# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



import torch
from typing import List, Dict, Any

import os
import json


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
        You are an expert at commonsense reasoning. Given the following multiple-choice question, provide a clear, step-by-step explanation for why the correct answer is right.

        {formatted_question}

        The correct answer is: {correct_answer}. {correct_answer_text}

        Please provide a reasoning explaining:
        1. Why this answer makes the most sense
        2. Why the other options are less suitable

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

    output = response.choices[0].message.content.strip()
    
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