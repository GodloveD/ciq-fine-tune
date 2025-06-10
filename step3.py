import os
import argparse
import json
import datetime
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

def parse_arguments():
    """
    Parse command line arguments from FuzzBall workflow
    """
    parser = argparse.ArgumentParser(description='T5 Fine-tuning')
    
    parser.add_argument('--cpu_cores', type=int, default=8,
                       help='Number of CPU cores available')
    parser.add_argument('--memory_gb', type=int, default=32,
                       help='Memory in GB')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--train_split', type=float, default=0.9,
                       help='Training split ratio')
    
    return parser.parse_args()

def load_model_and_tokenizer():
    """
    Load the model and tokenizer from previous job
    """
    print("Loading model and tokenizer from previous job...")
    
    if not os.path.exists("/scratch/model"):
        raise FileNotFoundError("Model not found. Make sure model-loading job completed successfully.")
    
    if not os.path.exists("/scratch/tokenizer"):
        raise FileNotFoundError("Tokenizer not found. Make sure model-loading job completed successfully.")
    
    model = T5ForConditionalGeneration.from_pretrained("/scratch/model")
    tokenizer = T5Tokenizer.from_pretrained("/scratch/tokenizer")
    
    print(f"Model loaded with {model.num_parameters():,} parameters")
    return model, tokenizer

def load_tokenized_data(train_split):
    """
    Load tokenized data and split into train/validation
    """
    print("Loading tokenized data...")
    
    if not os.path.exists("/scratch/tokenized_data"):
        raise FileNotFoundError("Tokenized data not found. Make sure model-loading job completed successfully.")
    
    dataset = Dataset.load_from_disk("/scratch/tokenized_data")
    print(f"Loaded {len(dataset)} tokenized examples")
    
    # Split into train and validation
    train_size = int(train_split * len(dataset))
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, len(dataset)))
    
    print(f"Train dataset: {len(train_dataset)} examples")
    print(f"Validation dataset: {len(val_dataset)} examples")
    
    return train_dataset, val_dataset

def fine_tune_model(model, tokenizer, train_dataset, val_dataset, args):
    """
    Fine-tune the model using the prepared data
    """
    print("Starting model fine-tuning...")
    
    output_dir = "/scratch/fine_tuned_model"
    
    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"Effective batch size: {effective_batch_size}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        dataloader_num_workers=min(args.cpu_cores, 8),  # Use available cores but cap at 8
        warmup_steps=100,
        logging_steps=50,
        eval_steps=200,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        remove_unused_columns=False,
        report_to=None,  # Disable wandb/tensorboard for simplicity
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print(f"Training with:")
    print(f"  - {len(train_dataset)} training examples")
    print(f"  - {len(val_dataset)} validation examples")
    print(f"  - {args.num_epochs} epochs")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print()
    
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")
    return output_dir

def save_training_summary(args, output_dir, train_dataset, val_dataset):
    """
    Save a summary of the training configuration and results
    """
    summary = {
        "model_name": "google/flan-t5-base",  # Could parameterize this
        "cpu_cores": args.cpu_cores,
        "memory_gb": args.memory_gb,
        "training_config": {
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "train_split": args.train_split
        },
        "dataset_info": {
            "training_examples": len(train_dataset),
            "validation_examples": len(val_dataset),
            "total_examples": len(train_dataset) + len(val_dataset)
        },
        "output_directory": output_dir,
        "completed_at": str(datetime.datetime.now())
    }
    
    summary_path = f"{output_dir}/training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Training summary saved to {summary_path}")
    return summary

def main():
    print("=" * 50)
    print("T5 SQUAD FINE-TUNING")
    print("=" * 50)
    
    args = parse_arguments()
    
    print(f"Configuration:")
    print(f"  CPU cores: {args.cpu_cores}")
    print(f"  Memory: {args.memory_gb} GB")
    print(f"  Epochs: {args.num_epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Train split: {args.train_split}")
    print()
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer()
        
        # Load and split tokenized data
        train_dataset, val_dataset = load_tokenized_data(args.train_split)
        
        # Fine-tune the model
        output_dir = fine_tune_model(model, tokenizer, train_dataset, val_dataset, args)
        
        # Save training summary
        summary = save_training_summary(args, output_dir, train_dataset, val_dataset)
        
        print("\n" + "=" * 50)
        print("FINE-TUNING COMPLETE!")
        print("=" * 50)
        print(f"Fine-tuned model saved to: {output_dir}")
        print(f"Training examples: {summary['dataset_info']['training_examples']}")
        print(f"Validation examples: {summary['dataset_info']['validation_examples']}")
        print(f"Epochs completed: {args.num_epochs}")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)