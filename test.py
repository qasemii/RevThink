import argparse
import os
import shutil
import tempfile
import logging
from typing import Optional, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import peft
from datasets import load_dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_merge_peft(base_model_name: str, peft_checkpoint_dir: str, tmp_dir: Optional[str] = None) -> str:
    """Load base model, apply PEFT adapters from checkpoint, merge, and save to a temp dir.

    Returns the path to the merged model directory.
    """
    if not os.path.isdir(peft_checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {peft_checkpoint_dir}")

    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    logger.info(f"Applying PEFT adapters from: {peft_checkpoint_dir}")
    model = peft.PeftModel.from_pretrained(base_model, peft_checkpoint_dir)

    # Merge LoRA weights into the base model for faster inference
    logger.info("Merging LoRA adapters into base model and unloading PEFT wrappers...")
    try:
        model = model.merge_and_unload()
    except AttributeError:
        logger.warning("merge_and_unload not available; proceeding without merging.")

    # Save merged model to a temporary directory
    save_dir = tmp_dir or tempfile.mkdtemp(prefix="merged_model_")
    os.makedirs(save_dir, exist_ok=True)

    logger.info(f"Saving merged model to: {save_dir}")
    model.save_pretrained(save_dir)

    # Save tokenizer as well so downstream loader finds special tokens
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(save_dir)

    return save_dir


def main():
    parser = argparse.ArgumentParser(description="Load a trained PEFT model and evaluate it on a dataset.")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to the PEFT checkpoint directory saved by train.py (e.g., ./checkpoints/gemma-7b_CSQA_sft_all)")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base HF model name that was fine-tuned (e.g., google/gemma-7b-it)")
    parser.add_argument("--dataset", type=str, default="tau/commonsense_qa",
                        help="Dataset name (tau/commonsense_qa, ai2_arc, hellaswag)")
    parser.add_argument("--split", type=str, default="validation",
                       help="Dataset split to evaluate on") # split
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate") # max number of samples
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to run on (auto, cuda, cpu)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum input length")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON path to save evaluation results")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Optional HF token (or set HF_TOKEN env var)")

    args, unknown = parser.parse_known_args()
    if unknown:
        logger.warning(f"Ignoring unknown arguments: {unknown}")

    # Authenticate to HF if needed
    login(os.getenv("HF_TOKEN") or args.hf_token)

    merged_dir = None
    try:
        # Merge PEFT adapters into base model
        merged_dir = load_and_merge_peft(args.base_model, args.checkpoint_dir)

        # Load merged model and tokenizer for inference
        device = (
            torch.device("cuda") if (args.device in ["auto", "cuda"] and torch.cuda.is_available())
            else torch.device("cpu")
        )
        logger.info(f"Loading merged model from {merged_dir} on device: {device}")
        tokenizer = AutoTokenizer.from_pretrained(merged_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
        )
        model.eval()

        # Dataset loading (no CoT)
        logger.info(f"Loading dataset: {args.dataset} [{args.split}]")
        if "commonsense_qa" in args.dataset.lower():
            dataset = load_dataset("tau/commonsense_qa", split=args.split)

            def extract_question(example: Dict) -> str:
                # Simple prompt: question and lettered choices; no CoT
                q = "Answer the following question:\n"
                q += f"Question: {example['question']}\n"
                for i, choice in enumerate(example['choices']['text']):
                    q += f"{example['choices']['label'][i]}. {choice}\n"
                q += "Answer:"
                return q

            def extract_choices(example: Dict) -> List[str]:
                return example['choices']['label']

            def extract_answer(example: Dict) -> str:
                return example['answerKey']

        elif "ai2_arc" in args.dataset.lower() or "arc" in args.dataset.lower():
            dataset = load_dataset("ai2_arc", "ARC-Challenge", split=args.split)

            def extract_question(example: Dict) -> str:
                q = f"Question: {example['question']}\n"
                for i, choice in enumerate(example['choices']['text']):
                    label = example['choices']['label'][i]
                    q += f"{label}. {choice}\n"
                q += "Answer:"
                return q

            def extract_choices(example: Dict) -> List[str]:
                return example['choices']['label']

            def extract_answer(example: Dict) -> str:
                return example['answerKey']

        elif "hellaswag" in args.dataset.lower():
            dataset = load_dataset("hellaswag", split=args.split)

            def extract_question(example: Dict) -> str:
                q = f"Context: {example['ctx']}\n"
                for i, ending in enumerate(example['endings']):
                    q += f"{chr(65 + i)}. {ending}\n"
                q += "Answer:"
                return q

            def extract_choices(example: Dict) -> List[str]:
                # Use letters A-D
                return [chr(65 + i) for i in range(len(example['endings']))]

            def extract_answer(example: Dict) -> str:
                return chr(65 + int(example['label']))

        else:
            raise ValueError(f"Dataset {args.dataset} not supported.")

        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))

        def choice_probabilities(prompt: str, choice_letters: List[str]) -> Dict[str, float]:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)
            scores: Dict[str, float] = {}
            for letter in choice_letters:
                # Try with space prefix, then without
                token_ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
                if not token_ids:
                    token_ids = tokenizer.encode(letter, add_special_tokens=False)
                if token_ids:
                    scores[letter] = probs[token_ids[0]].item()
                else:
                    scores[letter] = 0.0
            return scores

        # Evaluate
        total = len(dataset)
        correct = 0
        predictions = []
        logger.info(f"Evaluating {total} samples (no CoT)...")
        for i, ex in enumerate(dataset):
            prompt = extract_question(ex)
            choice_letters = extract_choices(ex)
            probs = choice_probabilities(prompt, choice_letters)
            pred = max(probs.keys(), key=lambda k: probs[k])
            gold = extract_answer(ex)
            is_correct = pred == gold
            if is_correct:
                correct += 1
            predictions.append({
                "question": prompt[:200],
                "predicted": pred,
                "gold": gold,
                "is_correct": is_correct,
                "probs": probs,
            })

        accuracy = correct / total if total > 0 else 0.0
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS - No CoT")
        print(f"{'='*60}")
        print(f"Model: merged ({args.base_model})")
        print(f"Dataset: {args.dataset} [{args.split}] | Samples: {total}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        if args.output:
            import json
            results = {
                "model": args.base_model,
                "dataset": args.dataset,
                "split": args.split,
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "predictions": predictions[:50],  # limit size
            }
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {args.output}")

    finally:
        # Clean up temporary merged model directory if we created one
        if merged_dir and os.path.isdir(merged_dir):
            try:
                shutil.rmtree(merged_dir)
            except Exception as e:
                logger.warning(f"Could not remove temporary directory {merged_dir}: {e}")


if __name__ == "__main__":
    main()