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
    tokenized_prompts = tokenizer(prompt_strs, padding=False, add_special_tokens=False)["input_ids"]
    tokenized_outputs = tokenizer(output_strs, padding=False, add_special_tokens=False)["input_ids"]

    concat_input_ids = []
    # range of prompt to output (for the labels, so we subtract 1)
    response_starts = []
    response_ends = []
    for tokenized_prompt, tokenized_output in zip(tokenized_prompts, tokenized_outputs):
        concat_input_ids.append(tokenized_prompt + tokenized_output)
        response_start = len(tokenized_prompt) - 1
        response_starts.append(response_start)
        response_ends.append(response_start + len(tokenized_output) - 1)

    max_len = max(len(input_ids) for input_ids in concat_input_ids)
    for i in range(len(concat_input_ids)):
        concat_input_ids[i] = concat_input_ids[i] + [tokenizer.pad_token_id] * (max_len - len(concat_input_ids[i]))

    concat_input_ids = torch.tensor(concat_input_ids)
    input_ids = concat_input_ids[:, :-1]
    labels = concat_input_ids[:, 1:]

    response_starts = torch.tensor(response_starts).unsqueeze(1)
    response_ends = torch.tensor(response_ends).unsqueeze(1)
    positions = torch.arange(max_len - 1).unsqueeze(0)
    response_mask = (positions >= response_starts) & (positions <= response_ends)

    return {"input_ids": input_ids, "labels": labels, "response_mask": response_mask}

# compute the per-token entropy of the logits
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_z = torch.logsumexp(logits, dim=-1)
    probs = torch.exp(logits - log_z.unsqueeze(-1))
    entropy = log_z - torch.sum(probs * logits, dim=-1)
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    output_logits = model(input_ids).logits
    log_probs = F.log_softmax(output_logits, dim=-1)
    gathered_log_probs = torch.gather(log_probs, index=labels.unsqueeze(-1), dim=-1).squeeze(-1)

    result_dict = {"log_probs": gathered_log_probs}

    # compute the log_probs
    if return_token_entropy:
        result_dict["token_entropy"] = compute_entropy(output_logits)

    return result_dict

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None= None,
) -> torch.Tensor:
    return torch.sum(torch.where(mask, tensor, torch.zeros_like(tensor)), dim=dim) / normalize_constant

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

def log_generations(model, vllm, eval_prompts, eval_answers, reward_fn,
                    sampling_params, total_train_steps: int, num_samples: int = 100,
                    full_dataset: bool = False):
    load_policy_into_vllm_instance(model, vllm)

    # randomly sample a set of indices
    if full_dataset:
        eval_prompts_sample = eval_prompts
        eval_answers_sample = eval_answers
    else:
        indices = random.sample(range(len(eval_prompts)), num_samples)
        eval_prompts_sample = [eval_prompts[i] for i in indices]
        eval_answers_sample = [eval_answers[i] for i in indices]

    eval_results = evaluate_vllm(vllm, reward_fn, eval_prompts_sample, eval_answers_sample, sampling_params)

    correct_lengths = []
    incorrect_lengths = []
    format_reward = 0
    answer_reward = 0
    for info_dict in eval_results:
        if info_dict["answer_reward"] == 1:
            correct_lengths.append(len(info_dict["response"]))
        else:
            incorrect_lengths.append(len(info_dict["response"]))
        format_reward += info_dict["format_reward"]
        answer_reward += info_dict["answer_reward"]

    correct_length = sum(correct_lengths) / len(correct_lengths) if correct_lengths else 0
    incorrect_length = sum(incorrect_lengths) / len(incorrect_lengths) if incorrect_lengths else 0
    average_length = sum(correct_lengths + incorrect_lengths) / len(correct_lengths + incorrect_lengths) if correct_lengths + incorrect_lengths else 0

    avg_format_reward = format_reward / len(eval_results)
    avg_answer_reward = answer_reward / len(eval_results)
    wandb.log({"eval/correct_length": correct_length,
               "eval/incorrect_length": incorrect_length,
               "eval/average_length": average_length,
               "eval/format_reward": avg_format_reward,
               "eval/answer_reward": avg_answer_reward,
               "eval_step": total_train_steps})

    # print one sample
    # print(f"Prompt: {eval_results[0]['prompt']}")
    # print(f"Answer: {eval_results[0]['answer']}")
    # print(f"Response: {eval_results[0]['response']}")
    # print(f"Format Reward: {eval_results[0]['format_reward']}")
    # print(f"Answer Reward: {eval_results[0]['answer_reward']}")

    return avg_answer_reward, avg_format_reward

# before this: set up wandb, model, tokenizer, slice data
def sft_training_loop(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    llm: LLM,
    sft_prompts: List[str],
    sft_cots: List[str],
    sft_answers: List[str],
    gradient_accumulation_steps: int,
    microbatch_size: int,
    device: str,
    eval_prompts: List[str],
    eval_answers: List[str],
    eval_steps: int,
    eval_sampling_params: SamplingParams,
    output_dir: str,
    learning_rate: float = 1e-5,
    max_grad_norm: float | None = 1.0,
    epochs: int = 10,
    half_dataset: bool = False,
    starting_step: int = 0,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_eval_reward = 0
    total_train_steps = starting_step
    running_loss = 0

    for epoch in range(epochs):
        # shuffle the data
        indices = list(range(len(sft_prompts)))
        random.shuffle(indices)
        sft_prompts = [sft_prompts[i] for i in indices]
        sft_cots = [sft_cots[i] for i in indices]
        sft_answers = [sft_answers[i] for i in indices]
        print(f"Epoch {epoch}")

        progress_bar = tqdm(range(0, len(sft_prompts), microbatch_size))
        for microbatch_idx, i in enumerate(progress_bar):
            microbatch_prompts = sft_prompts[i:i+microbatch_size]
            microbatch_cots = sft_cots[i:i+microbatch_size]
            microbatch_answers = sft_answers[i:i+microbatch_size]

            # log the epoch
            progress_bar.set_description(f"Epoch {epoch}")

            # tokenize the data
            tokenize_result = tokenize_prompt_and_output(microbatch_prompts, microbatch_cots, tokenizer)
            input_ids = tokenize_result["input_ids"].to(device)
            labels = tokenize_result["labels"].to(device)
            response_mask = tokenize_result["response_mask"].to(device)

            # model response
            model_output_dict = get_response_log_probs(model, input_ids, labels, return_token_entropy=True)
            policy_log_probs = model_output_dict["log_probs"].to(device)
            token_entropy = model_output_dict["token_entropy"].to(device)

            # loss and train step
            train_loss, _ = sft_microbatch_train_step(policy_log_probs, response_mask, gradient_accumulation_steps)
            running_loss += train_loss.item()
            if microbatch_idx % gradient_accumulation_steps == 0 or microbatch_idx == len(progress_bar) - 1:
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                total_train_steps += 1
                avg_loss = running_loss / gradient_accumulation_steps
                wandb.log({"train/loss": avg_loss,
                           "train/token_entropy": token_entropy.mean(),
                           "train_step": total_train_steps})

                progress_bar.set_postfix({"loss": avg_loss, "total_train_steps": total_train_steps})
                running_loss = 0

            # eval
            if microbatch_idx % eval_steps == 0:
                avg_answer_reward, avg_format_reward = log_generations(model, llm, eval_prompts, eval_answers, r1_zero_reward_fn, eval_sampling_params, total_train_steps)
                print(f"Avg eval reward (total/format): {avg_answer_reward}, {avg_format_reward}")
                if avg_answer_reward > best_eval_reward:
                    print("Saving best model")
                    best_eval_reward = avg_answer_reward
                    save_model_tokenizer(model, tokenizer, output_dir, "best")

    # eval vllm
    if half_dataset:
        avg_answer_reward, avg_format_reward = log_generations(model, llm, eval_prompts, eval_answers, r1_zero_reward_fn, eval_sampling_params, total_train_steps, full_dataset=False, num_samples=1000)
    else:
        avg_answer_reward, avg_format_reward = log_generations(model, llm, eval_prompts, eval_answers, r1_zero_reward_fn, eval_sampling_params, total_train_steps, full_dataset=True)

    print(f"Avg eval reward (total/format): {avg_answer_reward}, {avg_format_reward}")
    if avg_answer_reward > best_eval_reward:
        print("Saving best model")
        save_model_tokenizer(model, tokenizer, output_dir, "best")

    return total_train_steps

def load_sft_data(data_path: str, data_amount: int) -> tuple[List[str], List[str], List[str]]:
    prompts = []
    answers = []
    cots = []
    with open(data_path, "r") as data_file:
        for line in data_file:
            data = json.loads(line)
            prompts.append(data["prompt"])
            cots.append(data["response"])
            answers.append(data["ground_truth"])

    if data_amount == -1:
        return prompts, cots, answers
    else:
        return prompts[:data_amount], cots[:data_amount], answers[:data_amount]

def main(sft_data_path: str, eval_data_path: str, model_path: str, output_dir: str,
         microbatch_size: int, gradient_accumulation_steps: int, data_amount: int,
         eval_steps: int, epochs: int, learning_rate: float = 1e-5, max_grad_norm: float = 1.0, experiment_name: str = "sft",
         from_filtered: bool = False):
    setup_wandb(experiment_name)

    # load model and tokenizer
    train_device = "cuda:0"
    eval_device = "cuda:0"
    model, tokenizer = load_model_and_tokenizer(model_path)
    model.to(train_device)

    # optimize model with training data
    sft_prompts, sft_cots, sft_answers = load_sft_data(sft_data_path, data_amount)

    eval_sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )

    # eval vllm
    prompt_path = "prompts/r1_zero.prompt"
    eval_prompts, eval_answers = load_and_format_prompts(eval_data_path, prompt_path)

    print(f"sft_prompts: {len(sft_prompts)}, sft_answers: {len(sft_answers)}")
    print(f"eval_prompts: {len(eval_prompts)}, eval_answers: {len(eval_answers)}")

    llm = init_vllm(model_path, device=eval_device, seed=42)
    load_policy_into_vllm_instance(model, llm)
    sft_training_loop(model, tokenizer, llm, sft_prompts, sft_cots, sft_answers, gradient_accumulation_steps,
                      microbatch_size, train_device, eval_prompts, eval_answers,
                      eval_steps, eval_sampling_params, output_dir, learning_rate=learning_rate, max_grad_norm=max_grad_norm, epochs=epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--microbatch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--data_amount", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="results/sft")
    parser.add_argument("--eval_steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--experiment_name", type=str, default="sft_128")
    parser.add_argument("--from_filtered", action="store_true")
    args = parser.parse_args()
    print(args.epochs)

    if args.from_filtered:
        sft_data_path = "data/MATH/sft_filtered.jsonl"
    else:
        sft_data_path = "data/MATH/sft.jsonl"

    main(
        sft_data_path=sft_data_path,
        eval_data_path="data/MATH/validation.jsonl",
        model_path=QWEN_BASE_PATH,
        output_dir=args.output_dir,
        microbatch_size=args.microbatch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        data_amount=args.data_amount,
        eval_steps=args.eval_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        experiment_name=args.experiment_name,
        from_filtered=args.from_filtered
    )