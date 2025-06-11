"""
SQuAD Model Testing & Comparison Script - Cluster Optimized
Compares original vs fine-tuned model performance for console output
"""

import os
import json
import torch
import time
import warnings
import argparse
from typing import List, Dict, Tuple, Any
from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

def parse_arguments():
    """Parse command line arguments from FuzzBall workflow"""
    parser = argparse.ArgumentParser(description='SQuAD Model Testing & Comparison')
    
    parser.add_argument('--cpu_cores', type=int, default=8,
                       help='Number of CPU cores available')
    parser.add_argument('--memory_gb', type=int, default=32,
                       help='Memory in GB')
    parser.add_argument('--max_test_examples', type=int, default=50,
                       help='Maximum number of test examples')
    parser.add_argument('--original_model', type=str, default='google/flan-t5-base',
                       help='Original model name')
    parser.add_argument('--finetuned_model_path', type=str, default='/scratch/fine_tuned_model',
                       help='Path to fine-tuned model')
    
    return parser.parse_args()

class SQuADModelTester:
    """Test and compare SQuAD model performance before/after fine-tuning"""
    
    def __init__(self, original_model_name, finetuned_model_path):
        self.original_model_name = original_model_name
        self.finetuned_model_path = finetuned_model_path
        
        # Models and tokenizer (shared)
        self.original_model = None
        self.finetuned_model = None
        self.tokenizer = None  # Same tokenizer for both models
        
        # Test data
        self.test_data = None
        
        print("🧪 SQuAD Model Tester initialized")
        print(f"   Original model: {original_model_name}")
        print(f"   Fine-tuned model: {finetuned_model_path}")
    
    def load_models(self):
        """Load both original and fine-tuned models"""
        print("\n🔄 Loading models...")
        
        # Load tokenizer (same for both models since we didn't fine-tune it)
        try:
            print(f"📥 Loading tokenizer: {self.original_model_name}")
            self.tokenizer = T5Tokenizer.from_pretrained(self.original_model_name)
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            print("✅ Tokenizer loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            return False
        
        # Load original model
        try:
            print(f"📥 Loading original model: {self.original_model_name}")
            self.original_model = T5ForConditionalGeneration.from_pretrained(self.original_model_name)
            print("✅ Original model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading original model: {e}")
            return False
        
        # Load fine-tuned model
        try:
            print(f"📥 Loading fine-tuned model: {self.finetuned_model_path}")
            
            if not os.path.exists(self.finetuned_model_path):
                print(f"❌ Fine-tuned model not found at {self.finetuned_model_path}")
                print("💡 Make sure you've run the fine-tuning pipeline first!")
                return False
            
            self.finetuned_model = T5ForConditionalGeneration.from_pretrained(self.finetuned_model_path)
            print("✅ Fine-tuned model loaded successfully")
            print("💡 Note: Using same tokenizer for both models (only model weights were fine-tuned)")
            
        except Exception as e:
            print(f"❌ Error loading fine-tuned model: {e}")
            return False
        
        return True
    
    def load_test_data(self, num_test_examples=50):
        """Load SQuAD test data"""
        print(f"\n📚 Loading {num_test_examples} test examples...")
        
        try:
            # Load SQuAD validation set for testing
            dataset = load_dataset("squad_v2")
            validation_data = dataset["validation"]
            
            # Use subset for testing
            if num_test_examples < len(validation_data):
                self.test_data = validation_data.shuffle(seed=42).select(range(num_test_examples))
            else:
                self.test_data = validation_data
            
            print(f"✅ Loaded {len(self.test_data)} test examples")
            
            # Show data distribution
            answerable = sum(1 for example in self.test_data if len(example['answers']['text']) > 0)
            impossible = len(self.test_data) - answerable
            
            print(f"   Answerable questions: {answerable}")
            print(f"   Impossible questions: {impossible}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
            return False
    
    def generate_answer(self, model, tokenizer, question: str, context: str) -> Tuple[str, float]:
        """Generate answer using T5 model"""
        # Format input for T5
        input_text = f"answer question: {question} context: {context}"
        
        # Tokenize
        inputs = tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Generate answer
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=64,
                num_beams=3,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode answer
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer.strip(), generation_time
    
    def evaluate_model(self, model, tokenizer, model_name: str, max_examples: int = 50) -> Dict[str, Any]:
        """Evaluate model on test data"""
        print(f"\n🧪 Evaluating {model_name}...")
        
        results = {
            "model_name": model_name,
            "total_examples": 0,
            "answerable_examples": 0,
            "impossible_examples": 0,
            "correct_answers": 0,
            "correct_impossible": 0,
            "total_generation_time": 0.0,
            "examples": []
        }
        
        # Process test examples
        for i, example in enumerate(self.test_data):
            if i >= max_examples:
                break
                
            question = example['question']
            context = example['context']
            ground_truth_answers = example['answers']['text']
            is_impossible = len(ground_truth_answers) == 0
            
            # Generate answer
            try:
                predicted_answer, gen_time = self.generate_answer(model, tokenizer, question, context)
                
                # Evaluate correctness
                is_correct = False
                
                if is_impossible:
                    # For impossible questions, check if model indicates no answer
                    impossible_indicators = [
                        "cannot be answered", "no answer", "not provided", 
                        "impossible", "unanswerable", "not found", "not mentioned"
                    ]
                    is_correct = any(indicator in predicted_answer.lower() for indicator in impossible_indicators)
                    results["impossible_examples"] += 1
                    if is_correct:
                        results["correct_impossible"] += 1
                else:
                    # For answerable questions, check if prediction matches any ground truth
                    predicted_lower = predicted_answer.lower().strip()
                    for gt_answer in ground_truth_answers:
                        gt_lower = gt_answer.lower().strip()
                        # Simple exact match or substring match
                        if (predicted_lower == gt_lower or 
                            predicted_lower in gt_lower or 
                            gt_lower in predicted_lower):
                            is_correct = True
                            break
                    
                    results["answerable_examples"] += 1
                    if is_correct:
                        results["correct_answers"] += 1
                
                # Store example result
                example_result = {
                    "question": question,
                    "context": context[:100] + "..." if len(context) > 100 else context,
                    "ground_truth": ground_truth_answers[0] if ground_truth_answers else "No answer",
                    "predicted": predicted_answer,
                    "is_impossible": is_impossible,
                    "is_correct": is_correct,
                    "generation_time": gen_time
                }
                
                results["examples"].append(example_result)
                results["total_generation_time"] += gen_time
                results["total_examples"] += 1
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"   Processed {i + 1}/{min(max_examples, len(self.test_data))} examples...")
                    
            except Exception as e:
                print(f"Warning: Error processing example {i}: {e}")
                continue
        
        # Calculate metrics
        if results["answerable_examples"] > 0:
            results["answerable_accuracy"] = results["correct_answers"] / results["answerable_examples"]
        else:
            results["answerable_accuracy"] = 0.0
            
        if results["impossible_examples"] > 0:
            results["impossible_accuracy"] = results["correct_impossible"] / results["impossible_examples"]
        else:
            results["impossible_accuracy"] = 0.0
        
        if results["total_examples"] > 0:
            results["overall_accuracy"] = (results["correct_answers"] + results["correct_impossible"]) / results["total_examples"]
            results["avg_generation_time"] = results["total_generation_time"] / results["total_examples"]
        else:
            results["overall_accuracy"] = 0.0
            results["avg_generation_time"] = 0.0
        
        print(f"✅ {model_name} evaluation complete!")
        
        return results
    
    def compare_models(self, max_examples: int = 50) -> Dict[str, Any]:
        """Compare original vs fine-tuned model performance"""
        print("\n" + "="*80)
        print("🥊 MODEL COMPARISON")
        print("="*80)
        
        # Evaluate original model
        original_results = self.evaluate_model(
            self.original_model, 
            self.tokenizer, 
            "Original Model",
            max_examples
        )
        
        # Evaluate fine-tuned model
        finetuned_results = self.evaluate_model(
            self.finetuned_model, 
            self.tokenizer, 
            "Fine-tuned Model",
            max_examples
        )
        
        # Create comparison
        comparison = {
            "original": original_results,
            "finetuned": finetuned_results,
            "improvements": {},
            "test_examples": max_examples
        }
        
        # Calculate improvements
        comparison["improvements"] = {
            "overall_accuracy_improvement": finetuned_results["overall_accuracy"] - original_results["overall_accuracy"],
            "answerable_accuracy_improvement": finetuned_results["answerable_accuracy"] - original_results["answerable_accuracy"],
            "impossible_accuracy_improvement": finetuned_results["impossible_accuracy"] - original_results["impossible_accuracy"],
            "speed_improvement": original_results["avg_generation_time"] - finetuned_results["avg_generation_time"]
        }
        
        return comparison
    
    def print_detailed_results(self, comparison: Dict[str, Any]):
        """Print comprehensive comparison results"""
        print("\n" + "="*100)
        print("📊 DETAILED COMPARISON RESULTS")
        print("="*100)
        
        original = comparison["original"]
        finetuned = comparison["finetuned"]
        improvements = comparison["improvements"]
        
        # Performance Summary Table
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print("=" * 85)
        print(f"{'Metric':<30} {'Original':<15} {'Fine-tuned':<15} {'Improvement':<15}")
        print("=" * 85)
        print(f"{'Overall Accuracy':<30} {original['overall_accuracy']:<15.4f} {finetuned['overall_accuracy']:<15.4f} {improvements['overall_accuracy_improvement']:+<15.4f}")
        print(f"{'Answerable Questions':<30} {original['answerable_accuracy']:<15.4f} {finetuned['answerable_accuracy']:<15.4f} {improvements['answerable_accuracy_improvement']:+<15.4f}")
        print(f"{'Impossible Questions':<30} {original['impossible_accuracy']:<15.4f} {finetuned['impossible_accuracy']:<15.4f} {improvements['impossible_accuracy_improvement']:+<15.4f}")
        print(f"{'Avg Generation Time (s)':<30} {original['avg_generation_time']:<15.4f} {finetuned['avg_generation_time']:<15.4f} {improvements['speed_improvement']:+<15.4f}")
        print("=" * 85)
        
        # Detailed Breakdown
        print(f"\n📋 DETAILED BREAKDOWN:")
        print("-" * 60)
        print(f"Total Examples Tested: {original['total_examples']}")
        print(f"Answerable Questions: {original['answerable_examples']}")
        print(f"Impossible Questions: {original['impossible_examples']}")
        print()
        print("ORIGINAL MODEL RESULTS:")
        print(f"  • Correct answerable: {original['correct_answers']}/{original['answerable_examples']}")
        print(f"  • Correct impossible: {original['correct_impossible']}/{original['impossible_examples']}")
        print()
        print("FINE-TUNED MODEL RESULTS:")
        print(f"  • Correct answerable: {finetuned['correct_answers']}/{finetuned['answerable_examples']}")
        print(f"  • Correct impossible: {finetuned['correct_impossible']}/{finetuned['impossible_examples']}")
        
        # Improvement Analysis
        print(f"\n🎯 IMPROVEMENT ANALYSIS:")
        print("-" * 60)
        
        overall_improvement = improvements['overall_accuracy_improvement']
        if overall_improvement > 0:
            print(f"✅ OVERALL IMPROVEMENT: +{overall_improvement:.4f} ({overall_improvement*100:.2f}%)")
            print("   The fine-tuned model performs BETTER than the original model!")
        elif overall_improvement < 0:
            print(f"⚠️ OVERALL DECLINE: {overall_improvement:.4f} ({overall_improvement*100:.2f}%)")
            print("   The fine-tuned model performs worse than the original model.")
        else:
            print("➡️ NO CHANGE: The models perform equally well.")
        
        answerable_improvement = improvements['answerable_accuracy_improvement']
        if answerable_improvement > 0:
            print(f"✅ ANSWERABLE QUESTIONS: +{answerable_improvement:.4f} improvement")
        elif answerable_improvement < 0:
            print(f"⚠️ ANSWERABLE QUESTIONS: {answerable_improvement:.4f} decline")
        else:
            print("➡️ ANSWERABLE QUESTIONS: No change")
        
        impossible_improvement = improvements['impossible_accuracy_improvement']
        if impossible_improvement > 0:
            print(f"✅ IMPOSSIBLE QUESTIONS: +{impossible_improvement:.4f} improvement")
        elif impossible_improvement < 0:
            print(f"⚠️ IMPOSSIBLE QUESTIONS: {impossible_improvement:.4f} decline")
        else:
            print("➡️ IMPOSSIBLE QUESTIONS: No change")
        
        speed_improvement = improvements['speed_improvement']
        if speed_improvement > 0:
            print(f"⚡ SPEED IMPROVEMENT: {speed_improvement:.4f}s faster per answer")
        elif speed_improvement < 0:
            print(f"🐌 SPEED DECLINE: {abs(speed_improvement):.4f}s slower per answer")
        else:
            print("➡️ SPEED: No change")
    
    def show_example_comparisons(self, comparison: Dict[str, Any], num_examples: int = 3):
        """Show side-by-side example comparisons"""
        print(f"\n🔍 EXAMPLE COMPARISONS (First {num_examples})")
        print("=" * 120)
        
        original_examples = comparison["original"]["examples"]
        finetuned_examples = comparison["finetuned"]["examples"]
        
        for i in range(min(num_examples, len(original_examples), len(finetuned_examples))):
            orig = original_examples[i]
            fine = finetuned_examples[i]
            
            print(f"\n📝 EXAMPLE {i+1}:")
            print("-" * 120)
            print(f"❓ Question: {orig['question']}")
            print(f"📚 Context: {orig['context']}")
            print(f"✅ Ground Truth: {orig['ground_truth']}")
            print()
            print(f"🤖 Original Model:")
            print(f"   Answer: '{orig['predicted']}'")
            print(f"   Correct: {'✅ YES' if orig['is_correct'] else '❌ NO'}")
            print(f"   Time: {orig['generation_time']:.3f}s")
            print()
            print(f"🎯 Fine-tuned Model:")
            print(f"   Answer: '{fine['predicted']}'")
            print(f"   Correct: {'✅ YES' if fine['is_correct'] else '❌ NO'}")
            print(f"   Time: {fine['generation_time']:.3f}s")
            
            # Show improvement/decline
            if orig['is_correct'] != fine['is_correct']:
                if fine['is_correct'] and not orig['is_correct']:
                    print("🎉 IMPROVEMENT: Fine-tuned model got this RIGHT while original got it WRONG!")
                elif not fine['is_correct'] and orig['is_correct']:
                    print("⚠️ DECLINE: Fine-tuned model got this WRONG while original got it RIGHT!")
            elif orig['is_correct'] and fine['is_correct']:
                print("✅ BOTH CORRECT: Both models answered correctly")
            else:
                print("❌ BOTH WRONG: Both models answered incorrectly")
            
            print("-" * 120)
    
    def print_final_summary(self, comparison: Dict[str, Any]):
        """Print final summary and conclusions"""
        print("\n" + "="*100)
        print("🎉 FINE-TUNING EFFECTIVENESS SUMMARY")
        print("="*100)
        
        improvements = comparison["improvements"]
        overall_improvement = improvements["overall_accuracy_improvement"]
        
        print(f"📊 TEST RESULTS:")
        print(f"   • Test examples: {comparison['test_examples']}")
        print(f"   • Overall accuracy change: {overall_improvement:+.4f} ({overall_improvement*100:+.2f}%)")
        
        if overall_improvement > 0.01:  # Significant improvement
            print(f"\n🚀 SUCCESS! Fine-tuning was EFFECTIVE!")
            print(f"   ✅ The HPC cluster resources successfully improved model performance")
            print(f"   ✅ Fine-tuned model accuracy: {comparison['finetuned']['overall_accuracy']:.4f}")
            print(f"   ✅ Original model accuracy: {comparison['original']['overall_accuracy']:.4f}")
            print(f"   ✅ Improvement: +{overall_improvement:.4f}")
        elif overall_improvement < -0.01:  # Significant decline
            print(f"\n⚠️ Fine-tuning resulted in performance decline")
            print(f"   📉 This might indicate overfitting or insufficient training data")
            print(f"   💡 Consider adjusting hyperparameters or training longer")
        else:  # Minimal change
            print(f"\n➡️ Fine-tuning resulted in minimal change")
            print(f"   📊 The model performance remained approximately the same")
            print(f"   💡 This might indicate the model was already well-suited for the task")
        
        print(f"\n🔬 TECHNICAL DETAILS:")
        print(f"   • Original model: {comparison['original']['model_name']}")
        print(f"   • Fine-tuned model: {comparison['finetuned']['model_name']}")
        print(f"   • Answerable questions improvement: {improvements['answerable_accuracy_improvement']:+.4f}")
        print(f"   • Impossible questions improvement: {improvements['impossible_accuracy_improvement']:+.4f}")
        print(f"   • Speed change: {improvements['speed_improvement']:+.4f}s per answer")
        
        print(f"\n💡 CONCLUSION:")
        if overall_improvement > 0:
            print("   The HPC cluster fine-tuning process successfully enhanced model performance!")
            print("   This demonstrates the value of dedicated compute resources for model improvement.")
        else:
            print("   The fine-tuning process has been completed and evaluated.")
            print("   Further optimization may be needed to achieve better performance gains.")
        
        print("="*100)
    
    def run_complete_evaluation(self, max_examples: int = 50):
        """Run complete evaluation pipeline"""
        print("🚀 COMPLETE MODEL EVALUATION")
        print("="*80)
        print("🎯 Demonstrating HPC cluster fine-tuning effectiveness")
        print("="*80)
        
        # Load models
        if not self.load_models():
            print("❌ Failed to load models!")
            return False
        
        # Load test data
        if not self.load_test_data(max_examples):
            print("❌ Failed to load test data!")
            return False
        
        # Compare models
        comparison = self.compare_models(max_examples)
        
        # Show comprehensive results
        self.print_detailed_results(comparison)
        self.show_example_comparisons(comparison, num_examples=3)
        self.print_final_summary(comparison)
        
        return True

def main():
    """Main testing execution"""
    print("🧪 SQuAD MODEL TESTING & COMPARISON")
    print("="*80)
    print("🎯 Evaluating HPC cluster fine-tuning effectiveness")
    print("="*80)
    
    args = parse_arguments()
    
    print(f"⚙️ Test Configuration:")
    print(f"   CPU cores: {args.cpu_cores}")
    print(f"   Memory: {args.memory_gb} GB")
    print(f"   Test examples: {args.max_test_examples}")
    print(f"   Original model: {args.original_model}")
    print(f"   Fine-tuned model: {args.finetuned_model_path}")
    
    # Initialize tester
    tester = SQuADModelTester(args.original_model, args.finetuned_model_path)
    
    # Run evaluation
    try:
        success = tester.run_complete_evaluation(args.max_test_examples)
        
        if success:
            print("\n🎯 Testing completed successfully!")
            print("The comparison demonstrates the impact of HPC cluster fine-tuning on model performance.")
        else:
            print("\n❌ Testing failed. Check error messages above.")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)