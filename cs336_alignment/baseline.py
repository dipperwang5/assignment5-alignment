from vllm import LLM, SamplingParams
from typing import Callable, List, Tuple
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
import json
import os
from collections import Counter

QWEN_BASE_PATH = "models/Qwen2.5-Math-1.5B"
# LLAMA_8B_PATH = "models/Llama-3.1-8B"
# LLAMA_70B_PATH = "models/Llama-3.3-70B-Instruct"

def run_vllm(vllm_model, prompts, sampling_params) -> List[str]:
    outputs = vllm_model.generate(prompts, sampling_params)
    texts = [output.text for response in outputs for output in response.outputs]
    return texts

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    responses = run_vllm(vllm_model, prompts, eval_sampling_params)
    reward_dicts = [reward_fn(response, answer) for response, answer in zip(responses, answers)]

    info_dicts = reward_dicts
    for info_dict in info_dicts:
        info_dict["response"] = responses[info_dicts.index(info_dict)]
        info_dict["answer"] = answers[info_dicts.index(info_dict)]
        info_dict["prompt"] = prompts[info_dicts.index(info_dict)]

    return info_dicts

def load_and_format_prompts(data_path: str, prompt_path: str) -> List[str]:
    BASE_DIR = os.path.dirname(__file__)
    prompt_path = os.path.join(BASE_DIR, prompt_path)
    with open(prompt_path, "r") as prompt_file:
        prompt = prompt_file.read()

    prompts = []
    answers = []
    with open(data_path, "r") as data_file:
        for line in data_file:
            data = json.loads(line)
            prompts.append(prompt.format(question=data["problem"]))
            answers.append(data["answer"])

    return prompts, answers

def build_llm_and_params(model_path: str) -> Tuple[LLM, SamplingParams]:
    llm = LLM(model=model_path)
    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
    )

    return llm, sampling_params

def serialize_results(reward_dicts: List[dict[str, float]], output_path: str):
    with open(output_path, "w") as f:
        json.dump(reward_dicts, f)

def inspect_info_dicts(info_dicts: List[dict[str, float]]):
    counter = Counter()
    bad_formats = []
    bad_answers = []
    for info_dict in info_dicts:
        if info_dict["format_reward"] == 1.0 and info_dict["answer_reward"] == 1.0:
            counter["correct"] += 1
        elif info_dict["format_reward"] == 1.0 and info_dict["answer_reward"] == 0.0:
            counter["format_correct_answer_incorrect"] += 1
            bad_answers.append(info_dict["response"])
        else:
            bad_formats.append(info_dict["response"])
            counter["incorrect"] += 1
    print(counter)
    print(bad_formats[:10])
    print(bad_answers[:10])

def main(data_path: str, model_path: str, output_path: str, prompt_path: str):
    prompts, answers = load_and_format_prompts(data_path, prompt_path)
    llm, sampling_params = build_llm_and_params(model_path)
    info_dicts = evaluate_vllm(llm, r1_zero_reward_fn, prompts, answers, sampling_params)
    inspect_info_dicts(info_dicts)
    serialize_results(info_dicts, output_path)

if __name__ == "__main__":
    data_path = "data/MATH/validation.jsonl"
    model_path = QWEN_BASE_PATH

    prompt_path = "prompts/r1_zero.prompt"
    output_dir = "results/baseline"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_path.split('/')[-1]}_r1_zero.jsonl")
    main(data_path, model_path, output_path, prompt_path)