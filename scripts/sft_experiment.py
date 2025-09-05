from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import random

import sys
sys.path.append("/home/ke_wang/assignment5-alignment/tests/")
from adapters import (
    run_tokenize_prompt_and_output, 
    run_get_response_log_probs, 
    run_sft_microbatch_train_step,
)
import torch.optim as optim
 
random.seed(0)

# Initialize the policy model
model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Math-1.5B",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            ).to("cuda" if torch.cuda.is_available() else "cpu") # Move model to device

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")

n_sft_steps = 1000
batch_size = 16
gradient_accumulation_steps = 4
train_data_path = "/home/ke_wang/assignment5-alignment/data/math/train.jsonl"

# Note: This loads the entire dataset into memory, which may not scale to huge files.
train_data_questions = []
train_data_answers = []
with open(train_data_path, "r") as f:
    for line in f:
        line = json.loads(line)
        train_data_questions.append(line["problem"])
        train_data_answers.append(line["solution"])
train_dataset_indexes = list(range(len(train_data_questions)))

learning_rate = 1e-4
weight_decay_val = 0.01
betas_tuple = (0.9, 0.999)
optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay_val,
    betas=betas_tuple
)

for step in range(n_sft_steps):
    # Sample a batch of indices
    sample_indexes = random.choices(train_dataset_indexes, k=batch_size)
    
    prompt_strs = [train_data_questions[i] for i in sample_indexes]
    output_strs = [train_data_answers[i] for i in sample_indexes]

    # Tokenize the batch and move to the same device as the model
    encoded_input_output = run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)
    for k, v in encoded_input_output.items():
        encoded_input_output[k] = v.to(model.device)
    
    # Get log probabilities of the correct tokens
    log_probs = run_get_response_log_probs(
        model, 
        encoded_input_output["input_ids"], 
        encoded_input_output["labels"], 
        return_token_entropy=False
    )["log_probs"]
    
    # This function should perform loss.backward() internally
    loss, meta_data = run_sft_microbatch_train_step(
        log_probs, 
        encoded_input_output["response_mask"], 
        gradient_accumulation_steps
    )
    
    if (step + 1) % gradient_accumulation_steps == 0:
        # Update weights every `gradient_accumulation_steps` batches.
        optimizer.step()
        # Zero gradients every `gradient_accumulation_steps` batches.
        optimizer.zero_grad()
        
        print(f"Step {step}: Loss = {loss.item()}")