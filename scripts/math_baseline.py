
import sys
import os
sys.path.append("assignment5-alignment/cs336_alignment")
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from vllm import LLM, SamplingParams
import json
import pandas as pd

os.makedirs("/home/ubuntu/kw_test/assignment5-alignment/result/eval/", exist_ok=True)

prompt_base = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it.\
The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer.\
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, \
i.e., <think> reasoning process here </think> <answer> answer here </answer>.\
User: {question}\
Assistant: <think>"

test_data_questions = []
test_data_answers = []

with open("/home/ubuntu/kw_test/assignment5-alignment/data/math/test.jsonl", "r") as f:
    for line in f:
        line = json.loads(line)
        test_data_questions.append(prompt_base.format(question=line["problem"]))
        test_data_answers.append(line["solution"])
        # break

print(len(test_data_questions))
print(len(test_data_answers))

# Create a sampling params object, stopping generation on newline.
sampling_params = SamplingParams(
temperature=1.0, top_p=1.0, max_tokens=1024, stop=["</answer>"],
include_stop_str_in_output=True
)

# Create an LLM.
llm = LLM(model="/home/ubuntu/kw_test/model/Qwen2.5-Math-1.5B")

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(test_data_questions, sampling_params)

# calculate the evaluation metrics and save results
with open("/home/ubuntu/kw_test/assignment5-alignment/result/eval/math.jsonl", "w") as f:
    for idx, (output, question, ground_truth) in enumerate(zip(outputs, test_data_questions, test_data_answers)):
        generated_answer = output.outputs[0].text
        grade = r1_zero_reward_fn(generated_answer, ground_truth)
        
        record = {
            "id": idx,
            "question": question,
            "answer": ground_truth,
            "generated_answer": generated_answer,
            **grade
        }
        f.write(json.dumps(record) + '\n')

# read the saved jsonl file and analysis
df = pd.read_json("/home/ubuntu/kw_test/assignment5-alignment/result/eval/math.jsonl", lines=True)
df_format_answer = df[(df["format_reward"] == 1.0) & (df["answer_reward"] == 1.0)]
df_format = df[(df["format_reward"] == 1.0) & (df["answer_reward"] == 0.0)]
df_no = df[(df["format_reward"] == 0.0) & (df["answer_reward"] == 0.0)]

print(f"both format and answer correct {df_format_answer.shape[0] / df.shape[0]}")
print(df_format_answer.iloc[:10].values.tolist())

print(f"only format correct {df_format.shape[0] / df.shape[0]}")
print(df_format.iloc[:10].values.tolist())

print(f"none correct {df_no.shape[0] / df.shape[0]}")
print(df_no.iloc[:10].values.tolist())