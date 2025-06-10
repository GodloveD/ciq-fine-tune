import os
import argparse
from datasets import load_dataset, Dataset

def parse_arguments():
    """
    Parse command line arguments from FuzzBall workflow
    """
    parser = argparse.ArgumentParser(description='SQuAD Data Preparation')
    
    parser.add_argument('--cpu_cores', type=int, default=8,
                       help='Number of CPU cores available')
    parser.add_argument('--memory_gb', type=int, default=32,
                       help='Memory in GB')
    parser.add_argument('--max_examples', type=int, default=5000,
                       help='Maximum number of examples to process')
    parser.add_argument('--max_input_length', type=int, default=512,
                       help='Maximum input sequence length')
    
    return parser.parse_args()

def prepare_squad_data(max_examples, max_input_length):
    """
    Load and format SQuAD data for T5 fine-tuning
    """
    print(f"Loading SQuAD 2.0 dataset - using {max_examples} examples")
    
    # Load the dataset
    dataset = load_dataset("squad_v2")
    train_data = dataset["train"].shuffle(seed=42).select(range(max_examples))
    
    input_texts = []
    target_texts = []
    
    print("Converting SQuAD examples to T5 format...")
    for i, example in enumerate(train_data):
        if i % 1000 == 0:
            print(f"Processed {i} examples...")
            
        question = example['question']
        context = example['context'] 
        answers = example['answers']['text']
        
        # Format for T5: "answer question: {question} context: {context}"
        input_text = f"answer question: {question} context: {context}"
        
        # Handle unanswerable questions in SQuAD 2.0
        if len(answers) > 0:
            target_text = answers[0]
        else:
            target_text = "This question cannot be answered based on the given context."
        
        # Basic length filtering
        if len(input_text.split()) <= max_input_length:
            input_texts.append(input_text)
            target_texts.append(target_text)
    
    # Create dataset
    processed_dataset = Dataset.from_dict({
        "input_text": input_texts,
        "target_text": target_texts
    })
    
    print(f"Data preparation complete!")
    print(f"Total examples processed: {len(processed_dataset)}")
    print(f"Sample input: {input_texts[0][:100]}...")
    print(f"Sample target: {target_texts[0]}")
    
    # Save processed data for next stage
    processed_dataset.save_to_disk("/scratch/processed_squad_data")
    print("Saved processed data to /scratch/processed_squad_data")
    
    return processed_dataset

def main():
    print("=" * 50)
    print("SQUAD DATA PREPARATION")
    print("=" * 50)
    
    args = parse_arguments()
    
    print(f"Configuration:")
    print(f"  CPU cores: {args.cpu_cores}")
    print(f"  Memory: {args.memory_gb} GB")
    print(f"  Max examples: {args.max_examples}")
    print(f"  Max input length: {args.max_input_length}")
    print()
    
    try:
        dataset = prepare_squad_data(args.max_examples, args.max_input_length)
        print("\nData preparation completed successfully!")
        return True
        
    except Exception as e:
        print(f"Data preparation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)