import json
import time
from typing import List, Dict, Any
import openai
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import os

class CommonsenseQAReasoningGenerator:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize the reasoning generator.
        
        Args:
            api_key: OpenAI API key
            model: OpenAI model to use (e.g., "gpt-3.5-turbo", "gpt-4")
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.together.xyz/v1",
            )
        self.model = model
        
    def load_commonsenseqa_dataset(self, split: str = "validation") -> List[Dict]:
        """
        Load CommonsenseQA dataset from HuggingFace.
        
        Args:
            split: Dataset split to load ("train", "validation", or "test")
            
        Returns:
            List of dataset examples
        """
        try:
            dataset = load_dataset("tau/commonsense_qa", split=split)
            return list(dataset)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
    
    def format_question_with_choices(self, example: Dict) -> str:
        """
        Format the question with multiple choice options.
        
        Args:
            example: Single example from CommonsenseQA dataset
            
        Returns:
            Formatted question string
        """
        question = example["question"]
        choices = example["choices"]
        
        formatted_question = f"Question: {question}\n\nChoices:\n"
        for i, choice in enumerate(choices["text"]):
            label = choices["label"][i]
            formatted_question += f"{label}. {choice}\n"
        
        return formatted_question
    
    def generate_reasoning(self, example: Dict) -> Dict[str, Any]:
        """
        Generate reasoning for why the correct answer is right.
        
        Args:
            example: Single example from CommonsenseQA dataset
            
        Returns:
            Dictionary with original data plus generated reasoning
        """
        formatted_question = self.format_question_with_choices(example)
        correct_answer = example["answerKey"]
        
        # Find the correct answer text
        correct_answer_text = ""
        for i, label in enumerate(example["choices"]["label"]):
            if label == correct_answer:
                correct_answer_text = example["choices"]["text"][i]
                break
        
        prompt = f"""
            You are an expert at commonsense reasoning. Given the following multiple-choice question, provide a clear, step-by-step explanation for why the correct answer is right.

            {formatted_question}

            The correct answer is: {correct_answer}. {correct_answer_text}

            Please provide a reasoning explaining:
            1. Why this answer makes the most sense
            2. Why the other options are less suitable

            Keep your explanation clear, concise, and as brief as possible, focusing on the logical reasoning process.
            
            """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that excels at explaining commonsense reasoning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            reasoning = response.choices[0].message.content.strip()
            
            return {
                "id": example.get("id", ""),
                "question": example["question"],
                "choices": example["choices"],
                "correct_answer": correct_answer,
                "correct_answer_text": correct_answer_text,
                "reasoning": reasoning,
                "model_used": self.model
            }
            
        except Exception as e:
            print(f"Error generating reasoning for question: {e}")
            return {
                "id": example.get("id", ""),
                "question": example["question"],
                "choices": example["choices"],
                "correct_answer": correct_answer,
                "correct_answer_text": correct_answer_text,
                "reasoning": f"Error generating reasoning: {str(e)}",
                "model_used": self.model
            }
    
    def process_dataset(self, 
                       split: str = "validation", 
                       max_examples: int = None,
                       output_file: str = "commonsenseqa_with_reasoning.json",
                       delay_seconds: float = 1.0) -> List[Dict]:
        """
        Process the entire dataset and generate reasoning for all examples.
        
        Args:
            split: Dataset split to process
            max_examples: Maximum number of examples to process (None for all)
            output_file: File to save results
            delay_seconds: Delay between API calls to respect rate limits
            
        Returns:
            List of processed examples with reasoning
        """
        print(f"Loading CommonsenseQA {split} dataset...")
        dataset = self.load_commonsenseqa_dataset(split)
        
        if not dataset:
            print("Failed to load dataset")
            return []
        
        if max_examples:
            dataset = dataset[:max_examples]
        
        print(f"Processing {len(dataset)} examples...")
        
        results = []
        for i, example in enumerate(tqdm(dataset, desc="Generating reasoning")):
            result = self.generate_reasoning(example)
            results.append(result)
            
            # Save progress periodically
            if (i + 1) % 10 == 0:
                self.save_results(results, output_file)
            
            # Respect rate limits
            time.sleep(delay_seconds)
        
        # Final save
        self.save_results(results, output_file)
        print(f"Results saved to {output_file}")
        
        return results
    
    def save_results(self, results: List[Dict], filename: str):
        """Save results to JSON file."""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    def load_results(self, filename: str) -> List[Dict]:
        """Load results from JSON file."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading results: {e}")
            return []
    
    def create_summary_report(self, results: List[Dict]) -> Dict:
        """Create a summary report of the processing results."""
        if not results:
            return {}
        
        total_examples = len(results)
        successful_reasoning = len([r for r in results if not r["reasoning"].startswith("Error")])
        
        summary = {
            "total_examples_processed": total_examples,
            "successful_reasoning_generated": successful_reasoning,
            "success_rate": successful_reasoning / total_examples * 100,
            "model_used": results[0]["model_used"],
            "sample_questions": [
                {
                    "question": r["question"][:100] + "...",
                    "correct_answer": r["correct_answer"],
                    "reasoning_preview": r["reasoning"][:200] + "..."
                }
                for r in results[:3]
            ]
        }
        
        return summary

# Example usage
def main():
    # Set your OpenAI API key
    # api_key = os.getenv("OPENAI_API_KEY")  # Make sure to set this environment variable
    api_key = "80f3a660c2bd5c3902101d1d1977c951c8659d55b2b949ae42e47f8b6d42c6d7"
        
    # Initialize the generator
    generator = CommonsenseQAReasoningGenerator(
        api_key=api_key,
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo"  # or "gpt-4" for better reasoning
    )
    
    # Process a small subset first (remove max_examples for full dataset)
    results = generator.process_dataset(
        split="validation",
        max_examples=10,  # Start with 5 examples for testing
        output_file="commonsenseqa_reasoning_sample.json",
        delay_seconds=1.0
    )
    
    # Generate summary report
    summary = generator.create_summary_report(results)
    print("\nSummary Report:")
    print(f"Total examples: {summary.get('total_examples_processed', 0)}")
    print(f"Success rate: {summary.get('success_rate', 0):.1f}%")
    
    # Save summary
    with open("summary_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nSample reasoning generated:")
    if results:
        print(f"Question: {results[0]['question']}")
        print(f"Correct Answer: {results[0]['correct_answer']} - {results[0]['correct_answer_text']}")
        print(f"Reasoning: {results[0]['reasoning']}")


if __name__ == "__main__":
    main()