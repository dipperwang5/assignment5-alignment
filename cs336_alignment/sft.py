import argparse
import json
import random
from typing import List
from unittest.mock import patch
import os

from tqdm import tqdm
import wandb
import torch
import torch.nn.functional as F
from einops import rearrange, reduce, repeat

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

from cs336_alignment.baseline import evaluate_vllm, r1_zero_reward_fn, load_and_format_prompts

QWEN_BASE_PATH = "Qwen/Qwen2.5-Math-1.5B"

def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.25):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/
    # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/
    22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def setup_wandb(experiment_name: str):
    # Setup wandb metrics
    wandb.init(project="alignment", name=experiment_name)
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    # the x‑axis for evaluation
    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")
    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")


def load_model_and_tokenizer(model_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    return model, tokenizer


def save_model_tokenizer(model, tokenizer, output_dir: str, suffix: str = ""):
    model.save_pretrained(save_directory=f"{output_dir}/{suffix}" if suffix else output_dir)
    tokenizer.save_pretrained(save_directory=f"{output_dir}/{suffix}" if suffix else output_dir)


def tokenize_prompt_and_output(prompt_strs: List[str], output_strs: List[str], tokenizer: PreTrainedTokenizer):
    input_ids_list = []
    response_mask_list = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_enc = tokenizer(prompt, add_special_tokens=False)
        output_enc = tokenizer(output, add_special_tokens=False)
        full_input = prompt_enc['input_ids'] + output_enc['input_ids']
        response_mask = [0] * len(prompt_enc['input_ids']) + [1] * len(output_enc['input_ids'])
        input_ids_list.append(torch.tensor(full_input, dtype=torch.long))
        response_mask_list.append(torch.tensor(response_mask, dtype=torch.long))

    batch_size = len(input_ids_list)
    max_len = max(len(ids) for ids in input_ids_list)
    input_ids_batch = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long)
    response_mask_batch = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i, (ids, mask) in enumerate(zip(input_ids_list, response_mask_list)):
        seq_len = len(ids)
        input_ids_batch[i, :seq_len] = ids
        response_mask_batch[i, :seq_len] = mask

    return {
        "input_ids": input_ids_batch[:, :-1],               # (batch, max_len-1)
        "labels": input_ids_batch[:, 1:],                   # (batch, max_len-1)
        "response_mask": response_mask_batch[:, 1:]         # (batch, max_len-1)
    }


# compute the per-token entropy of the logits
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_normalize_factor = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_p = logits - log_normalize_factor
    entropy = -reduce(log_p * torch.exp(log_p), "batch_size seq_len vocab_size -> batch_size seq_len", "sum")
    return entropy


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    tensor_masked = tensor * mask #batch_size seq_len
    tensor_masked_sum = torch.sum(tensor_masked, dim=dim) # batch_size

    return tensor_masked_sum / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    normalized_loss = masked_normalize(-policy_log_probs, response_mask, normalize_constant, dim=-1)
    adjusted_normalized_loss = normalized_loss.mean() / gradient_accumulation_steps
    adjusted_normalized_loss.backward()

    return adjusted_normalized_loss, {"normalized_loss": normalized_loss}