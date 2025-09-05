import argparse
import json
import random
import re
import pathlib
import torch
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from unittest.mock import patch

import sys
sys.path.append("/home/ke_wang/assignment5-alignment") 
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

sys.path.append("/home/ke_wang/assignment5-alignment/tests/")
from adapters import (
    run_tokenize_prompt_and_output,
    run_get_response_log_probs,
    run_sft_microbatch_train_step,
)

from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

# --- vLLM HELPER FUNCTIONS ---
def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """Initializes the vLLM instance on a separate GPU."""
    vllm_set_random_seed(seed)
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            tensor_parallel_size=1,
            device=device,
            dtype='bfloat16',
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )

def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """Loads the current training model's weights into the vLLM instance."""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())

# --- Main Experiment Script ---
def main(args):
    # --- 1. Setup ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy_device = torch.device(f"cuda:{args.policy_device_id}" if torch.cuda.is_available() else "cpu")
    vllm_device = f"cuda:{args.vllm_device_id}"

    run = wandb.init(
        project="cs336-assignment5-sft",
        config=vars(args),
        name=f"sft_size_{args.dataset_size}_lr_{args.learning_rate}"
    )
    wandb.define_metric("train/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("eval/*", step_metric="eval/step")

    # --- 2. Load Model and Tokenizer ---
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(policy_device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Initializing vLLM for evaluation on {vllm_device}...")
    vllm_instance = init_vllm(model_id=args.model_path, device=vllm_device, seed=args.seed)

    # --- 3. Load and Prepare Data ---
    print("Loading and preparing data...")
    train_questions, train_answers = [], []
    with open(args.train_data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if args.filter_correct:
                reward_info = r1_zero_reward_fn(data["response"], data["response"])
                if reward_info["answer_reward"] == 1.0:
                    train_questions.append(data["prompt"])
                    train_answers.append(data["response"])
            else:
                train_questions.append(data["prompt"])
                train_answers.append(data["response"])
    
    if args.dataset_size != -1:
        train_questions = train_questions[:args.dataset_size]
        train_answers = train_answers[:args.dataset_size]
    print(f"Using {len(train_questions)} training examples.")

    valid_questions, valid_answers = [], []
    with open(args.valid_data_path, "r") as f:
        for line in f:
            data = json.loads(line)
            valid_questions.append(data["prompt"])
            valid_answers.append(data["response"])
    
    train_indices = list(range(len(train_questions)))

    # --- 4. Initialize Optimizer ---
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # --- 5. Main Training Loop ---
    print("Starting SFT training...")
    for step in tqdm(range(args.num_train_steps), desc="Training Steps"):
        model.train()
        for micro_step in range(args.gradient_accumulation_steps):
            sample_indices = random.choices(train_indices, k=args.micro_batch_size)
            prompt_strs = [train_questions[i] for i in sample_indices]
            output_strs = [train_answers[i] for i in sample_indices]

            encoded = run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
            for k, v in encoded.items():
                encoded[k] = v.to(policy_device)

            log_probs_info = run_get_response_log_probs(model, encoded["input_ids"], encoded["labels"], False)
            
            loss, meta_data = run_sft_microbatch_train_step(
                log_probs_info["log_probs"],
                encoded["response_mask"],
                args.gradient_accumulation_steps,
            )

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        
        train_metrics = {f"train/{k}": v for k, v in meta_data.items()}
        train_metrics["train/step"] = step
        run.log(train_metrics)
        
        # --- 7. Periodic Evaluation ---
        if (step + 1) % args.eval_interval == 0:
            print(f"\n--- Evaluating at step {step} ---")
            model.eval()

            load_policy_into_vllm_instance(policy=model, llm=vllm_instance)
            
            eval_indices = random.sample(range(len(valid_questions)), k=args.num_eval_samples)
            prompts_for_eval = [valid_questions[i] for i in eval_indices]
            ground_truths_for_eval = [valid_answers[i] for i in eval_indices]
            
            sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

            outputs = vllm_instance.generate(prompts_for_eval, sampling_params, use_tqdm=False)
            
            total_correct = 0
            for i in range(len(outputs)):
                response = outputs[i].outputs[0].text
                ground_truth = ground_truths_for_eval[i]
                # CHANGED: Use r1_zero_reward_fn for evaluation
                reward_info = r1_zero_reward_fn(response, ground_truth)
                if reward_info["answer_reward"] == 1.0:
                    total_correct += 1
            
            accuracy = total_correct / args.num_eval_samples
            
            eval_metrics = {"eval/accuracy": accuracy, "eval/step": step}
            run.log(eval_metrics)
            print(f"Validation Accuracy: {accuracy:.4f}")
            
            model.train()
            
    # --- 8. Save Final Model ---
    print("Training finished. Saving final model...")
    output_dir = pathlib.Path(args.output_dir) / run.name
    model.save_pretrained(save_directory=output_dir)
    tokenizer.save_pretrained(save_directory=output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str, default="data/math/train.jsonl")
    parser.add_argument("--valid_data_path", type=str, default="data/math/test.jsonl")
    parser.add_argument("--output_dir", type=str, default="/data/ke_wang/sft_checkpoints")
    parser.add_argument("--dataset_size", type=int, default=-1, help="Number of examples to use. -1 for full dataset.")
    parser.add_argument("--filter_correct", action="store_true", help="Filter dataset to only correct examples.")
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--num_eval_samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--policy_device_id", type=int, default=0, help="GPU ID for the training policy model.")
    parser.add_gument("--vllm_device_id", type=int, default=1, help="GPU ID for the vLLM evaluation model.")

    args = parser.parse_args()
    
    assert args.batch_size % args.gradient_accumulation_steps == 0
    args.micro_batch_size = args.batch_size // args.gradient_accumulation_steps
    
    main(args)