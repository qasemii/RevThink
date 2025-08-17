import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
import logging
from collections import defaultdict
import re
import sys
import os

from huggingface_hub import login


login(os.getenv("HF_TOKEN"))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoTMCQEvaluator:
    def __init__(self, model_name: str, device: str = "auto", max_length: int = 512):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.max_length = max_length

        logger.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None
        )

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        logger.info(f"Model loaded on device: {self.device}")

    def _setup_device(self, device: str) -> torch.device:
        """Setup the appropriate device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def format_question_standard(self, question: str, choices: List[str]) -> str:
        """Format question for standard (non-CoT) evaluation."""
        formatted = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            formatted += f"{chr(65 + i)}. {choice}\n"
        formatted += "Answer:"
        return formatted

    def format_question_cot(self, question: str, choices: List[str], cot_style: str = "basic") -> str:
        """
        Format question with Chain-of-Thought prompting.

        Args:
            question: The question text
            choices: List of answer choices
            cot_style: Style of CoT prompting ("basic", "detailed", "step_by_step", "few_shot")
        """
        if cot_style == "basic":
            formatted = f"Question: {question}\n"
            for i, choice in enumerate(choices):
                formatted += f"{chr(65 + i)}. {choice}\n"
            formatted += "\nLet me think through this step by step:\n"

        elif cot_style == "detailed":
            formatted = f"Please solve this multiple-choice question step by step.\n\n"
            formatted += f"Question: {question}\n\nOptions:\n"
            for i, choice in enumerate(choices):
                formatted += f"{chr(65 + i)}. {choice}\n"
            formatted += f"\nLet me analyze each option carefully and explain my reasoning:\n"

        elif cot_style == "step_by_step":
            formatted = f"Question: {question}\n"
            for i, choice in enumerate(choices):
                formatted += f"{chr(65 + i)}. {choice}\n"
            formatted += f"\nTo solve this, I need to:\n1. Understand what the question is asking\n2. Analyze each option\n3. Choose the best answer\n\nStep-by-step analysis:\n"

        elif cot_style == "few_shot":
            # Include few-shot examples for CoT reasoning
            formatted = self._get_few_shot_examples()
            formatted += f"\n\nNow let me solve this question:\n\n"
            formatted += f"Question: {question}\n"
            for i, choice in enumerate(choices):
                formatted += f"{chr(65 + i)}. {choice}\n"
            formatted += f"\nLet me think through this step by step:\n"

        return formatted

    def _get_few_shot_examples(self) -> str:
        """Generate few-shot examples for CoT prompting."""
        examples = """Here are some examples of how to solve multiple-choice questions with step-by-step reasoning:

                      Example 1:
                      Question: What is the capital of France?
                      A. London
                      B. Berlin
                      C. Paris
                      D. Madrid

                      Let me think through this step by step:
                      This is asking for the capital city of France. I need to recall my knowledge of European geography.
                      - London is the capital of the United Kingdom
                      - Berlin is the capital of Germany
                      - Paris is the capital of France
                      - Madrid is the capital of Spain

                      Therefore, the answer is C. Paris

                      Example 2:
                      Question: If a book costs $12 and is on sale for 25% off, what is the sale price?
                      A. $8
                      B. $9
                      C. $10
                      D. $11

                      Let me think through this step by step:
                      First, I need to calculate 25% of $12:
                      25% = 0.25
                      0.25 × $12 = $3

                      The discount is $3, so the sale price is:
                      $12 - $3 = $9

                      Therefore, the answer is B. $9"""

        return examples

    def generate_reasoning(self, prompt: str, max_new_tokens: int = 200) -> str:
        """
        Generate the model's reasoning for a given prompt.

        Args:
            prompt: The formatted question prompt
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            Generated reasoning text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low temperature for more consistent reasoning
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        reasoning = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return reasoning.strip()

    def extract_answer_from_reasoning(self, reasoning: str, num_choices: int) -> Optional[str]:
        """
        Extract the final answer from the generated reasoning.

        Args:
            reasoning: Generated reasoning text
            num_choices: Number of available choices

        Returns:
            Extracted answer letter or None if not found
        """
        # Look for common answer patterns
        patterns = [
            r'(?:answer is|Answer:|the answer is)\s*([A-E])',
            r'(?:Therefore|So|Thus),?\s*(?:the answer is)?\s*([A-E])',
            r'(?:correct answer|right answer).*?([A-E])',
            r'\b([A-E])\s*(?:is correct|is right|is the answer)',
            r'choose\s*([A-E])',
            r'select\s*([A-E])',
        ]

        valid_choices = [chr(65 + i) for i in range(num_choices)]

        # Try each pattern
        for pattern in patterns:
            matches = re.findall(pattern, reasoning, re.IGNORECASE)
            for match in matches:
                if match.upper() in valid_choices:
                    return match.upper()

        # Fallback: look for any letter that appears near the end
        words = reasoning.split()[-20:]  # Look at last 20 words
        for word in reversed(words):
            for char in word:
                if char.upper() in valid_choices:
                    return char.upper()

        # Last resort: find the most frequent valid letter in the reasoning
        letter_counts = defaultdict(int)
        for char in reasoning.upper():
            if char in valid_choices:
                letter_counts[char] += 1

        if letter_counts:
            return max(letter_counts.keys(), key=lambda k: letter_counts[k])

        return None

    def get_answer_probabilities(self, prompt: str, choices: List[str]) -> Dict[str, float]:
        """Get probability-based predictions (for non-CoT evaluation)."""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]

        choice_probs = {}
        choice_letters = [chr(65 + i) for i in range(len(choices))]

        for letter in choice_letters:
            token_id = self.tokenizer.encode(f" {letter}", add_special_tokens=False)
            if token_id:
                choice_probs[letter] = torch.softmax(logits, dim=-1)[token_id[0]].item()
            else:
                token_id = self.tokenizer.encode(letter, add_special_tokens=False)
                if token_id:
                    choice_probs[letter] = torch.softmax(logits, dim=-1)[token_id[0]].item()
                else:
                    choice_probs[letter] = 0.0

        return choice_probs

    def predict_answer_cot(self, question: str, choices: List[str], cot_style: str = "basic") -> Dict:
        """
        Predict answer using Chain-of-Thought reasoning.

        Returns:
            Dictionary containing prediction, reasoning, and metadata
        """
        prompt = self.format_question_cot(question, choices, cot_style)
        reasoning = self.generate_reasoning(prompt)
        predicted_letter = self.extract_answer_from_reasoning(reasoning, len(choices))

        return {
            "predicted_answer": predicted_letter,
            "reasoning": reasoning,
            "prompt": prompt,
            "extraction_successful": predicted_letter is not None
        }

    def predict_answer_standard(self, question: str, choices: List[str]) -> Dict:
        """Predict answer using standard probability-based approach."""
        prompt = self.format_question_standard(question, choices)
        choice_probs = self.get_answer_probabilities(prompt, choices)
        predicted_letter = max(choice_probs.keys(), key=lambda k: choice_probs[k])

        return {
            "predicted_answer": predicted_letter,
            "choice_probabilities": choice_probs,
            "prompt": prompt,
            "extraction_successful": True
        }

    def evaluate_dataset(self, dataset_name: str, split: str = "validation",
                        max_samples: Optional[int] = None,
                        use_cot: bool = True, cot_style: str = "basic") -> Dict:
        """
        Evaluate the model on a dataset with or without CoT reasoning.

        Args:
            dataset_name: Name of the dataset
            split: Dataset split to evaluate on
            max_samples: Maximum number of samples to evaluate
            use_cot: Whether to use Chain-of-Thought reasoning
            cot_style: Style of CoT prompting if use_cot=True
        """
        logger.info(f"Loading dataset: {dataset_name}")

        # Dataset configurations (same as before)
        if "commonsense_qa" in dataset_name.lower():
            dataset = load_dataset("tau/commonsense_qa", split=split)
            question_col = "question"
            choices_col = "choices"
            answer_col = "answerKey"

            def extract_choices(item):
                return item[choices_col]["text"]

            def extract_answer(item):
                return item[answer_col]

        elif "arc" in dataset_name.lower():
            dataset = load_dataset("ai2_arc", "ARC-Challenge", split=split)
            question_col = "question"
            choices_col = "choices"
            answer_col = "answerKey"

            def extract_choices(item):
                return item[choices_col]["text"]

            def extract_answer(item):
                return item[answer_col]

        elif "hellaswag" in dataset_name.lower():
            dataset = load_dataset("hellaswag", split=split)
            question_col = "ctx"
            choices_col = "endings"
            answer_col = "label"

            def extract_choices(item):
                return item[choices_col]

            def extract_answer(item):
                return chr(65 + int(item[answer_col]))

        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")

        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        results = {
            "total": len(dataset),
            "correct": 0,
            "accuracy": 0.0,
            "predictions": [],
            "confusion_matrix": defaultdict(lambda: defaultdict(int)),
            "evaluation_method": "CoT" if use_cot else "Standard",
            "cot_style": cot_style if use_cot else None,
            "extraction_failures": 0
        }

        method_desc = f"CoT ({cot_style})" if use_cot else "Standard probability-based"
        logger.info(f"Evaluating on {len(dataset)} samples using {method_desc} method...")

        for i, item in enumerate(tqdm(dataset)):
            question = item[question_col]
            choices = extract_choices(item)
            correct_answer = extract_answer(item)

            try:
                if use_cot:
                    pred_result = self.predict_answer_cot(question, choices, cot_style)
                else:
                    pred_result = self.predict_answer_standard(question, choices)

                predicted_answer = pred_result["predicted_answer"]

                # Handle extraction failures
                if not pred_result["extraction_successful"] or predicted_answer is None:
                    results["extraction_failures"] += 1
                    # For failed extractions, randomly guess
                    predicted_answer = chr(65)  # Default to 'A'

                is_correct = predicted_answer == correct_answer
                if is_correct:
                    results["correct"] += 1

                # Store prediction details
                prediction_record = {
                    "question": question,
                    "choices": choices,
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "extraction_successful": pred_result["extraction_successful"]
                }

                if use_cot:
                    prediction_record["reasoning"] = pred_result["reasoning"]
                else:
                    prediction_record["choice_probabilities"] = pred_result["choice_probabilities"]

                results["predictions"].append(prediction_record)

                # Update confusion matrix
                results["confusion_matrix"][correct_answer][predicted_answer] += 1

            except Exception as e:
                logger.error(f"Error processing sample {i}: {e}")
                continue

        results["accuracy"] = results["correct"] / results["total"] if results["total"] > 0 else 0.0
        results["extraction_failure_rate"] = results["extraction_failures"] / results["total"] if results["total"] > 0 else 0.0

        return results

    def print_results(self, results: Dict):
        """Print evaluation results in a formatted way."""
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - {results['evaluation_method']} METHOD")
        print(f"{'='*60}")
        print(f"Model: {self.model_name}")
        print(f"Total samples: {results['total']}")
        print(f"Correct predictions: {results['correct']}")
        print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")

        if results['evaluation_method'] == 'CoT':
            print(f"CoT Style: {results['cot_style']}")
            print(f"Answer extraction failures: {results['extraction_failures']} ({results['extraction_failure_rate']*100:.2f}%)")

        # Print confusion matrix
        if results["confusion_matrix"]:
            print(f"\nConfusion Matrix:")
            all_labels = sorted(set(list(results["confusion_matrix"].keys()) +
                                  [k for v in results["confusion_matrix"].values() for k in v.keys()]))

            print(f"{'Actual/Pred':<12}", end="")
            for label in all_labels:
                print(f"{label:>6}", end="")
            print()

            for true_label in all_labels:
                print(f"{true_label:<12}", end="")
                for pred_label in all_labels:
                    count = results["confusion_matrix"][true_label][pred_label]
                    print(f"{count:>6}", end="")
                print()

        # Show example predictions
        print(f"\nSample Predictions:")
        for i, pred in enumerate(results["predictions"][:3]):
            print(f"\nExample {i+1}:")
            print(f"Question: {pred['question'][:100]}...")
            print(f"Choices: {pred['choices']}")
            print(f"Correct: {pred['correct_answer']} | Predicted: {pred['predicted_answer']} | {'✓' if pred['is_correct'] else '✗'}")

            if results['evaluation_method'] == 'CoT':
                print(f"Reasoning: {pred['reasoning'][:200]}...")
                print(f"Extraction successful: {pred['extraction_successful']}")
            else:
                print(f"Probabilities: {pred['choice_probabilities']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate language models on MCQ benchmarks with CoT reasoning")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="HuggingFace model name") # model
    parser.add_argument("--dataset", type=str, default="tau/commonsense_qa",
                       help="Dataset name (tau/commonsense_qa, ai2_arc, hellaswag)") # dataset
    parser.add_argument("--split", type=str, default="validation",
                       help="Dataset split to evaluate on") # split
    parser.add_argument("--max_samples", type=int, default=100,
                       help="Maximum number of samples to evaluate") # max number of samples
    parser.add_argument("--use_cot", action="store_true", default=True,
                       help="Use Chain-of-Thought reasoning") # CoT enable
    parser.add_argument("--cot_style", type=str, default="basic",
                       choices=["basic", "detailed", "step_by_step", "few_shot"],
                       help="Style of CoT prompting") # CoT mode
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to run on (auto, cuda, cpu)")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file to save results (JSON format)")
    parser.add_argument("--compare_methods", action="store_true", default=False,
                       help="Run both standard and CoT evaluation for comparison")

    # Parse only known arguments, ignoring others
    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning(f"Ignoring unknown arguments: {unknown}")

    # Initialize evaluator
    evaluator = CoTMCQEvaluator(args.model, args.device, args.max_length)

    if args.compare_methods:
        # Run both methods for comparison
        print("Running Standard Evaluation...")
        results_standard = evaluator.evaluate_dataset(
            args.dataset, args.split, args.max_samples, use_cot=False
        )

        print("\nRunning CoT Evaluation...")
        results_cot = evaluator.evaluate_dataset(
            args.dataset, args.split, args.max_samples, use_cot=True, cot_style=args.cot_style
        )

        # Print comparison
        evaluator.print_results(results_standard)
        evaluator.print_results(results_cot)

        print(f"\n{'='*60}")
        print(f"COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Standard Method Accuracy: {results_standard['accuracy']:.4f} ({results_standard['accuracy']*100:.2f}%)")
        print(f"CoT Method Accuracy: {results_cot['accuracy']:.4f} ({results_cot['accuracy']*100:.2f}%)")
        print(f"Improvement: {(results_cot['accuracy'] - results_standard['accuracy'])*100:.2f} percentage points")

        if args.output:
            combined_results = {
                "standard": results_standard,
                "cot": results_cot,
                "comparison": {
                    "standard_accuracy": results_standard['accuracy'],
                    "cot_accuracy": results_cot['accuracy'],
                    "improvement": results_cot['accuracy'] - results_standard['accuracy']
                }
            }
            with open(args.output, 'w') as f:
                json.dump(combined_results, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    else:
        # Run single evaluation
        results = evaluator.evaluate_dataset(
            args.dataset, args.split, args.max_samples, args.use_cot, args.cot_style
        )

        evaluator.print_results(results)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()