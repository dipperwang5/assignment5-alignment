from __future__ import annotations

import os
from typing import Any, Callable, Literal
import sys
sys.path.append("assignment5-alignment/cs336_alignment")
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

import torch
from torch import Tensor
from torch.utils.data import Dataset
import torch.nn.functional as F

import pandas as pd
import statistics
# import mlflow
from transformers import PreTrainedTokenizerBase

from einops import rearrange, reduce, repeat

import pdb

def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
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



def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    # compute rewards
    rewards_list = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        rewards_list.append(reward_fn(response, ground_truth)["reward"])

    # compute advantage function
    normalized_advantages_list = []
    unnormalized_advantages_list = []
    
    # pdb.set_trace()
    for idx in range(len(rollout_responses) // group_size):
        group_rewards_list = rewards_list[idx*group_size: (idx+1)*group_size]
        mean_group_reward = statistics.mean(group_rewards_list)
        std_group_reward = statistics.stdev(group_rewards_list)

        unnormalized_advantages = [reward - mean_group_reward for reward in group_rewards_list]
        normalized_advantages = [(reward - mean_group_reward) / (std_group_reward + advantage_eps) for reward in group_rewards_list]
        
        unnormalized_advantages_list.extend(unnormalized_advantages)
        normalized_advantages_list.extend(normalized_advantages)

    # metadata
    metadata = {
                "mean_reward": statistics.mean(rewards_list),
                "std_reward": statistics.stdev(rewards_list),
                "max_reward": max(rewards_list),
                "min_reward": min(rewards_list),
                }

    if normalize_by_std:
        return torch.Tensor(normalized_advantages_list), torch.Tensor(rewards_list), metadata
    else:
        return torch.Tensor(unnormalized_advantages_list), torch.Tensor(rewards_list), metadata

def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    # logits, batch_size seq_len vocab_size
    log_normalize_factor = torch.logsumexp(logits, dim=-1, keepdim=True)
    log_p = logits - log_normalize_factor
    entropy = -reduce(log_p * torch.exp(log_p), "batch_size seq_len vocab_size -> batch_size seq_len", "sum")
    return entropy


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    outputs = model(input_ids)
    logits = outputs.logits #batch_size seq_len vocab_size
    log_probs = F.log_softmax(logits, dim=-1) #batch_size seq_len vocab_size

    labels_unsqueeze = rearrange(labels, "batch_size seq_len -> batch_size seq_len 1")
    log_probs_labels = torch.gather(log_probs, dim=-1, index=labels_unsqueeze)
    log_probs_labels = rearrange(log_probs_labels, "batch_size seq_len 1 -> batch_size seq_len")

    output = {"log_probs": log_probs_labels}
    if return_token_entropy:
        entropy = run_compute_entropy(logits)
        output["token_entropy"] = entropy
        
    return output

def log_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_strs: list[str],
    ground_truth_strs: list[str],
    max_token_len: int = 128,
    temp: float = 0.7,
) -> None:
    """
    Generates responses from a model for given prompts and logs detailed information.
    """
    model.eval()
    with torch.no_grad():
        # tokenize the prompt for generation
        inputs = tokenizer(prompt_strs, return_tensors="pt", padding=True, truncation=True).to(model.device)
        prompt_token_lens = inputs["attention_mask"].sum(dim=1)

        # generate the response
        generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=max_token_len,
                        temperature=temp,
                        do_sample=True
                        )        

        # decode the generated responses
        response_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        generated_responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # calculate the entropy for generated responses
        log_prob_info = run_get_response_log_probs(model, input_ids=generated_ids, labels=generated_ids, return_token_entropy=True)
        token_entropies = log_prob_info["token_entropy"]

        generations_data = []
        all_response_lengths, correct_response_lengths, incorrect_response_lengths = [], [], []

        for i in range(len(prompt_strs)):
            prompt = prompt_strs[i]
            response = generated_responses[i]
            ground_truth = ground_truth_strs[i]
            prompt_len = prompt_token_lens[i]
            
            reward_info = r1_zero_reward_fn(response, ground_truth)
            
            response_len = len(response_ids[i])
            response_entropies = token_entropies[i, prompt_len : prompt_len+prompt_len]
            avg_entropy = response_entropies.mean().item() if len(response_entropies) > 0 else 0.0

            # Add the row of data to our list
            generations_data.append({
                "Prompt": prompt,
                "Generated Response": response,
                "Ground Truth": ground_truth,
                "Format Reward": reward_info["format_reward"],
                "Answer Reward": reward_info["answer_reward"],
                "Total Reward": reward_info["reward"],
                "Avg Token Entropy": avg_entropy
            })

            all_response_lengths.append(response_len)
            if reward_info["answer_reward"] > 0:
                correct_response_lengths.append(response_len)
            else:
                incorrect_response_lengths.append(response_len)

        # log the detailed generation data as a CSV artifact
        df = pd.DataFrame(generations_data)
        df.to_csv("generations.csv", index=False)
        # mlflow.log_artifact("generations.csv", artifact_path="generations")

        # log the aggregate metrics
        scalar_metrics = {
            "avg_response_length": np.mean(all_response_lengths) if all_response_lengths else 0,
            "avg_correct_response_length": np.mean(correct_response_lengths) if correct_response_lengths else 0,
            "avg_incorrect_response_length": np.mean(incorrect_response_lengths) if incorrect_response_lengths else 0,
        }
        # mlflow.log_metrics(scalar_metrics)

    model.train()


def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    raise NotImplementedError


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    raise NotImplementedError


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    raise NotImplementedError


def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    raise NotImplementedError

def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    batch_size, seq_length = policy_log_probs.shape

    ce_loss = -policy_log_probs
    loss_sum = run_masked_normalize(ce_loss, response_mask, 
                                normalize_constant=normalize_constant)
    
    loss = loss_sum / batch_size / gradient_accumulation_steps
    loss.backward()
    
    n_tokens = response_mask.sum()
    avg_token_ce = loss_sum / (n_tokens + 1e-8)
    meta_data = {
        "log_sum": loss_sum.detach(),
        "n_tokens": n_tokens.detach(),
        "avg_ce_per_token": avg_token_ce.detach(),
        "mean_log_prob": (policy_log_probs * response_mask).sum() / (n_tokens + 1e-8)
    }

    return loss.detach(), meta_data


    
def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    raise NotImplementedError


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    tensor_masked = tensor * mask #batch_size seq_len
    tensor_masked_sum = torch.sum(tensor_masked, dim=dim) # batch_size

    return tensor_masked_sum / normalize_constant



"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    raise NotImplementedError


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    raise NotImplementedError


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    raise NotImplementedError


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    raise NotImplementedError
