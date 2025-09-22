from collections import defaultdict
import argparse
import numpy as np
import statistics
import torch
from typing import Literal
from cs336_alignment.sft import get_response_log_probs, setup_wandb, load_model_and_tokenizer, \
                                load_and_format_prompts, init_vllm, load_policy_into_vllm_instance, log_generations, \
                                tokenize_prompt_and_output, masked_normalize
from vllm import SamplingParams
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, question_only_reward_fn
from cs336_alignment.baseline import run_vllm
import random
from tqdm import tqdm
import wandb

def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):
    metadata = defaultdict(list)
    raw_rewards = []
    advantages = []

    rollout_batch_size = len(rollout_responses)
    for i in range(0, rollout_batch_size, group_size):
        group_responses = rollout_responses[i : i + group_size]
        group_ground_truths = repeated_ground_truths[i : i + group_size]

        group_rewards = []
        for response, ground_truth in zip(group_responses, group_ground_truths):
            reward = reward_fn(response, ground_truth)
            group_rewards.append(reward["reward"])
            raw_rewards.append(reward["reward"])

        avg_reward = sum(group_rewards) / group_size
        std_reward = statistics.stdev(group_rewards)
        metadata["avg_reward"].append(avg_reward)
        metadata["std_reward"].append(std_reward)

        for reward in group_rewards:
            advantage = reward - avg_reward
            if normalize_by_std:
                advantage /= (std_reward + advantage_eps)

            advantages.append(advantage)

    return advantages, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metadata = defaultdict(list)

    importance_ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_importance_ratio = torch.clamp(importance_ratio, 1 - cliprange, 1 + cliprange)
    raw_policy_weight = advantages * importance_ratio
    clipped_policy_weight = advantages * clipped_importance_ratio

    metadata["clipped"].append(clipped_policy_weight < raw_policy_weight)

    return -torch.min(raw_policy_weight, clipped_policy_weight), metadata

def compute_grpo_unclipped_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    metadata = defaultdict(list)

    importance_ratio = torch.exp(policy_log_probs - old_log_probs)
    return -advantages * importance_ratio, {}

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_unclipped"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    elif loss_type == "reinforce_with_baseline":
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    elif loss_type == "grpo_clip":
        return compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    elif loss_type == "grpo_unclipped":
        return compute_grpo_unclipped_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None= None,
) -> torch.Tensor:
    return torch.sum(torch.where(mask, tensor, torch.zeros_like(tensor)), dim=dim) / mask.sum(dim=dim)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip", "grpo_unclipped"],
    raw_rewards: torch.Tensor | None= None,
    advantages: torch.Tensor | None= None,
    old_log_probs: torch.Tensor | None= None,
    cliprange: float | None= None,
    use_mask_normalize: bool | None= None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    policy_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange
    )

    if use_mask_normalize:
        response_policy_loss = masked_normalize(policy_loss, response_mask, response_mask.shape[0])
    else:
        # mean over all dimensions wherever we have a response token
        response_policy_loss = masked_mean(policy_loss, response_mask)

    scaled_response_policy_loss = response_policy_loss / gradient_accumulation_steps
    scaled_response_policy_loss.backward()

    return scaled_response_policy_loss, metadata

def grpo_train_loop(n_grpo_steps, learning_rate, advantage_eps,
                    group_size, rollout_sampling_params, eval_sampling_params,
                    n_train_steps_per_rollout_batch, train_batch_size, micro_train_batch_size,
                    gradient_accumulation_steps, n_prompts_per_rollout_batch, epochs_per_rollout_batch,
                    loss_type, use_std_normalization,
                    reward_fn, model, tokenizer, llm, train_prompts,
                    train_answers, output_dir, cliprange, eval_steps,
                    eval_prompts, eval_answers, max_grad_norm, use_mask_normalize):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_train_steps = 0
    for i in range(n_grpo_steps):
        # get train_batch_size prompts
        print(f"GRPO iteration {i}")
        load_policy_into_vllm_instance(model, llm)
        # 32 prompts
        batch_indices = random.sample(range(len(train_prompts)), n_prompts_per_rollout_batch)
        batch_prompts = [train_prompts[i] for i in batch_indices]
        batch_answers = [train_answers[i] for i in batch_indices]

        # now these are size rollout_batch_size (n_prompts_per_rollout_batch * group_size) = 256
        batch_responses = run_vllm(llm, batch_prompts, rollout_sampling_params)

        print("num prompts:", len(batch_prompts))
        print("num generated:", len(batch_responses))

        repeated_ground_truths = []
        repeated_batch_prompts = []
        for answer, prompt in zip(batch_answers, batch_prompts):
            repeated_ground_truths.extend([answer] * group_size)
            repeated_batch_prompts.extend([prompt] * group_size)

        advantages, raw_rewards, metadata = compute_group_normalized_rewards(
            reward_fn, batch_responses, repeated_ground_truths, group_size, advantage_eps, use_std_normalization
        )

        info_dict = tokenize_prompt_and_output(repeated_batch_prompts, batch_responses, tokenizer)
        input_ids = info_dict["input_ids"].to(device)
        labels = info_dict["labels"].to(device)
        response_mask = info_dict["response_mask"].to(device)

        model.eval()
        with torch.no_grad():
            old_log_probs = torch.cat([
                get_response_log_probs(
                    model,
                    input_ids[idx:idx+micro_train_batch_size],
                    labels[idx:idx+micro_train_batch_size],
                    return_token_entropy=True)["log_probs"].to(device)
                for idx in range(0, len(batch_responses), micro_train_batch_size)])
        old_log_probs = old_log_probs.detach()

        model.train()

        for epoch in range(epochs_per_rollout_batch):
            print(f"GRPO epoch {epoch}")
            # # this is 1 for on-policy, > 1 for off-policy --> this is unrelated to train batch size
            # for j in range(n_train_steps_per_rollout_batch):
            # micro_train_batch_size = 256/128 = 2
            # now we have rollout_batch_size for rewards, advantages, etc
            num_accumulated = 0
            for microbatch_idx in tqdm(range(0, len(batch_responses), micro_train_batch_size)):
                start_idx = microbatch_idx
                end_idx = microbatch_idx + micro_train_batch_size

                microbatch_responses = batch_responses[start_idx:end_idx]
                microbatch_advantages = torch.tensor(advantages[start_idx:end_idx]).unsqueeze(-1).to(device)
                microbatch_raw_rewards = torch.tensor(raw_rewards[start_idx:end_idx]).unsqueeze(-1).to(device)
                microbatch_response_mask = response_mask[start_idx:end_idx]

                microbatch_log_probs = get_response_log_probs(
                    model,
                    input_ids[start_idx:end_idx],
                    labels[start_idx:end_idx],
                    return_token_entropy=True)

                microbatch_policy_log_probs = microbatch_log_probs["log_probs"].to(device)
                microbatch_token_entropy = microbatch_log_probs["token_entropy"].to(device)
                microbatch_old_log_probs = old_log_probs[start_idx:end_idx]

                loss, metadata = grpo_microbatch_train_step(
                    microbatch_policy_log_probs,
                    microbatch_response_mask, gradient_accumulation_steps,
                    loss_type, microbatch_raw_rewards, microbatch_advantages,
                    microbatch_old_log_probs, cliprange, use_mask_normalize
                )

                wandb.log({
                    "train/loss": loss.item(),
                    "train/token_entropy": microbatch_token_entropy.mean().item(),
                    "train_step": total_train_steps,
                },
                step=total_train_steps)
                total_train_steps += 1
                num_accumulated += 1

                if num_accumulated % gradient_accumulation_steps == 0 or microbatch_idx == len(batch_responses) - 1:
                    if max_grad_norm:
                        preclipped_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    # also want to log the gradient norm
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.detach().data.pow(2).sum().item()
                    total_norm = total_norm ** 0.5

                    wandb.log({
                        "train/grad_norm": preclipped_norm,
                        "train/grad_norm_post_clip": total_norm,
                        "train_step": total_train_steps,
                    })

                    optimizer.step()
                    optimizer.zero_grad()

        if i % eval_steps == 0 or i == n_grpo_steps - 1:
            avg_answer_reward, avg_format_reward = log_generations(
                model, llm, eval_prompts, eval_answers, reward_fn,
                eval_sampling_params, total_train_steps=i, num_samples=1024)

            print(f"Avg eval reward (total/format): {avg_answer_reward}, {avg_format_reward}")

if __name__ == "__main__":
    # add relevant args
    parser = argparse.ArgumentParser()
    # hyperparameters
    parser.add_argument("--n_grpo_steps", type=int, default=75)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--advantage_eps", type=float, default=1e-6)
    parser.add_argument("--rollout_batch_size", type=int, default=256)
    parser.add_argument("--epochs_per_rollout_batch", type=int, default=1)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--sampling_temperature", type=float, default=1.0)
    parser.add_argument("--sampling_min_tokens", type=int, default=4)
    parser.add_argument("--sampling_max_tokens", type=int, default=1024)
    parser.add_argument("--n_train_steps_per_rollout_batch", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=256)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=128)
    parser.add_argument("--loss_type", type=str, default="reinforce_with_baseline")
    parser.add_argument("--use_std_normalization", type=bool, default=True)
    parser.add_argument("--n_prompts_per_batch", type=int, default=32)
    parser.add_argument("--cliprange", type=float, default=0.2)
    # experiment args
    parser.add_argument("--experiment_name", type=str, default="grpo")
    parser.add_argument("--model_path", type=str, default="models/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path", type=str, default="data/MATH/train.jsonl")
    parser.add_argument("--eval_data_path", type=str, default="data/MATH/validation.jsonl")
    parser.add_argument("--data_amount", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default="results/grpo")
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_mask_normalize", action="store_true")
    parser.add_argument("--question_only", action="store_true")
    args = parser.parse_args()

    assert args.train_batch_size % args.gradient_accumulation_steps == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    assert args.rollout_batch_size % args.group_size == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = args.rollout_batch_size // args.group_size
    assert args.train_batch_size >= args.group_size, (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = args.rollout_batch_size // micro_train_batch_size

    setup_wandb(args.experiment_name)

    # load model and tokenizer
    device = "cuda:0"
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    model.to(device)

    rollout_sampling_params = SamplingParams(
        temperature=args.sampling_temperature,
        max_tokens=args.sampling_max_tokens,
        min_tokens=args.sampling_min_tokens,
        n=args.group_size,
        seed=42,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    eval_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )

    # eval vllm
    prompt_path = "prompts/question_only.prompt" if args.question_only else "prompts/r1_zero.prompt"
    eval_fn = question_only_reward_fn if args.question_only else r1_zero_reward_fn
    train_prompts, train_answers = load_and_format_prompts(args.train_data_path, prompt_path)
    eval_prompts, eval_answers = load_and_format_prompts(args.eval_data_path, prompt_path)

    # vllm for generating the rollouts
    llm = init_vllm(args.model_path, device=device, seed=42)
    load_policy_into_vllm_instance(model, llm)

    grpo_train_loop(args.n_grpo_steps, args.learning_rate, args.advantage_eps,
                    args.group_size, rollout_sampling_params, eval_sampling_params,
                    args.n_train_steps_per_rollout_batch, args.train_batch_size,
                    micro_train_batch_size, args.gradient_accumulation_steps,
                    n_prompts_per_rollout_batch, args.epochs_per_rollout_batch,
                    args.loss_type, args.use_std_normalization, eval_fn,
                    model, tokenizer, llm, train_prompts, train_answers,
                    args.output_dir, args.cliprange, args.eval_steps,
                    eval_prompts, eval_answers, args.max_grad_norm, args.use_mask_normalize)