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

from utils import generate_reasoning, save_results

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

    if output_file is None:
        output_file = f'output.json'
    save_results(results, output_file)


    print("\nSample reasoning generated:")
    if results:
        print(f"Question: {results[0]['question']}")
        print(f"Correct Answer: {results[0]['correct_answer']} - {results[0]['correct_answer_text']}")
        print(f"Reasoning: {results[0]['reasoning']}")


if __name__ == "__main__":
    main()