import json
import time
from typing import List, Dict, Any
import openai
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import os
import argparse
from openai import OpenAI


def verbalize_question(example: Dict) -> str:
    question = example["question"]
    choices = example["choices"]
    
    formatted_question = f"Question: {question}\n\nChoices:\n"
    for i, choice in enumerate(choices["text"]):
        label = choices["label"][i]
        formatted_question += f"{label}. {choice}\n"
    
    return formatted_question


def generate_reasoning(example, client, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):

    formatted_question = verbalize_question(example)
    correct_answer = example["answerKey"]

    # Find the correct answer text
    correct_answer_text = ""
    for i, label in enumerate(example["choices"]["label"]):
        if label == correct_answer:
            correct_answer_text = example["choices"]["text"][i]
            break
    
    prompt = f"""
        You are an expert at commonsense reasoning. Given the following multiple-choice question, provide a clear explanation for why the correct answer is right and other answers are wrong.

        {formatted_question}

        The correct answer is: {correct_answer}. {correct_answer_text}

        Keep your explanation clear, concise, and as brief as possible, focusing on the logical reasoning process.

        """

    generation_configs = {
        "temperature": 0.8,
        # "top_p": 1,
        # "frequency_penalty": 0,
        # "presence_penalty": 0,
        "max_tokens": 1000,
    }

    response = client.chat.completions.create(
          model=model,
          messages=[
              {"role": "system", "content": "You are a helpful assistant that excels at explaining commonsense reasoning."},
              {"role": "user", "content": prompt}
          ],
          **generation_configs
      )

    reasoning = response.choices[0].message.content.strip()
    
    return {
        "id": example.get("id", ""),
        "question": example["question"],
        "choices": example["choices"],
        "correct_answer": correct_answer,
        "correct_answer_text": correct_answer_text,
        "reasoning": reasoning,
    }


def save_results(results: List[Dict], filename: str):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def load_results(filename: str) -> List[Dict]:
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return []



# Example usage
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train', type=str)
    parser.add_argument('--max_examples', default=None, type=int)
    parser.add_argument('--output_file', default=None, type=str)
    
    args = parser.parse_args()

    api_key = "80f3a660c2bd5c3902101d1d1977c951c8659d55b2b949ae42e47f8b6d42c6d7"

    dataset = load_dataset("tau/commonsense_qa", split=args.split)
    if args.max_examples:
        dataset = dataset.select(range(args.max_examples))

    client = OpenAI(
        api_key = api_key or os.getenv("OPENAI_API_KEY"),
        base_url="https://api.together.xyz/v1",
    )

    results = []
    for i, example in enumerate(tqdm(dataset, desc="Generating reasoning")):
        result = generate_reasoning(example, client)
        results.append(result)

    if args.output_file is None:
        args.output_file = f'output.json'
    save_results(results, args.output_file)


    print("\nSample reasoning generated:")
    if results:
        print(f"Question: {results[0]['question']}")
        print(f"Correct Answer: {results[0]['correct_answer']} - {results[0]['correct_answer_text']}")
        print(f"Reasoning: {results[0]['reasoning']}")


if __name__ == "__main__":
    main()