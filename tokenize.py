import os
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_from_disk

def parse_arguments():
    """
    Parse command line arguments from FuzzBall workflow
    """
    parser = argparse.ArgumentParser(description='Model Loading and Tokenization')
    
    parser.add_argument('--cpu_cores', type=int, default=8,
                       help='Number of CPU cores available')
    parser.add_argument('--memory_gb', type=int, default=32,
                       help='Memory in GB')
    parser.add_argument('--model_name', type=str, default='google/flan-t5-base',
                       help='Pre-trained model to load')
    parser.add_argument('--max_input_length', type=int, default=512,
                       help='Maximum input sequence length')
    parser.add_argument('--max_target_length', type=int, default=128,
                       help='Maximum target sequence length')
    
    return parser.parse_args()

def load_model_and_tokenizer(model_name):
    """
    Load T5 model and tokenizer
    """
    print(f"Loading model: {model_name}")
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Handle padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model and tokenizer loaded successfully")
    print(f"Model parameters: ~{model.num_parameters() / 1e6:.1f}M")
    
    return model, tokenizer

def tokenize_dataset(tokenizer, max_input_length, max_target_length):
    """
    Load processed data and tokenize it
    """
    print("Loading processed SQuAD data...")
    
    # Load the dataset saved from previous job
    dataset = load_from_disk("/scratch/processed_squad_data")
    print(f"Loaded {len(dataset)} examples")
    
    def tokenize_function(examples):
        # Tokenize inputs
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=max_input_length,
            truncation=True,
            padding=True,
            return_tensors=None
        )
        
        # Tokenize targets
        targets = tokenizer(
            examples["target_text"],
            max_length=max_target_length,
            truncation=True,
            padding=True,
            return_tensors=None
        )
        
        model_inputs["labels"] = targets["input_ids"]
        return model_inputs
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=["input_text", "target_text"]
    )
    
    print("Tokenization complete!")
    print(f"Sample tokenized input shape: {len(tokenized_dataset[0]['input_ids'])}")
    print(f"Sample tokenized label shape: {len(tokenized_dataset[0]['labels'])}")
    
    # Save tokenized dataset
    tokenized_dataset.save_to_disk("/scratch/tokenized_squad_data")
    print("Saved tokenized data to /scratch/tokenized_squad_data")
    
    # Save model and tokenizer for next stage
    model.save_pretrained("/scratch/loaded_model")
    tokenizer.save_pretrained("/scratch/loaded_tokenizer")
    print("Saved model and tokenizer to /scratch/")
    
    return tokenized_dataset

def main():
    print("=" * 50)
    print("MODEL LOADING AND TOKENIZATION")
    print("=" * 50)
    
    args = parse_arguments()
    
    print(f"Configuration:")
    print(f"  CPU cores: {args.cpu_cores}")
    print(f"  Memory: {args.memory_gb} GB")
    print(f"  Model: {args.model_name}")
    print(f"  Max input length: {args.max_input_length}")
    print(f"  Max target length: {args.max_target_length}")
    print()
    
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model_name)
        
        # Tokenize the dataset
        tokenized_dataset = tokenize_dataset(
            tokenizer, 
            args.max_input_length, 
            args.max_target_length
        )
        
        print(f"\nModel loading and tokenization completed successfully!")
        print(f"Ready for training with {len(tokenized_dataset)} tokenized examples")
        return True
        
    except Exception as e:
        print(f"Model loading/tokenization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)