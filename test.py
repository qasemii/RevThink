import argparse
import os
import shutil
import tempfile
import logging
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import peft

from evaluate import CoTMCQEvaluator


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
                        help="Dataset split (validation/test)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max number of samples to evaluate")
    parser.add_argument("--use_cot", action="store_true", default=True,
                        help="Use Chain-of-Thought reasoning prompts")
    parser.add_argument("--cot_style", type=str, default="basic",
                        choices=["basic", "detailed", "step_by_step", "few_shot"],
                        help="CoT prompting style")
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

    tmp_dir = None
    merged_dir = None
    try:
        # Merge PEFT adapters into base model
        merged_dir = load_and_merge_peft(args.base_model, args.checkpoint_dir)

        # Reuse existing evaluator pipeline by pointing it to the merged model directory
        evaluator = CoTMCQEvaluator(merged_dir, device=args.device, max_length=args.max_length)
        results = evaluator.evaluate_dataset(
            dataset_name=args.dataset,
            split=args.split,
            max_samples=args.max_samples,
            use_cot=args.use_cot,
            cot_style=args.cot_style,
        )
        evaluator.print_results(results)

        if args.output:
            import json
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