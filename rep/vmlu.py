import json
import os
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # adjust if needed


def build_prompt(question: str, choices: List[str]) -> str:
    """Format question and choices into a simple prompt."""
    prompt = "Question:\n" + question.strip() + "\n\nChoices:\n"
    for c in choices:
        prompt += c.strip() + "\n"
    prompt += "\nAnswer: "
    return prompt


def load_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: str, records: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def evaluate(model_id: str,
             input_file: str,
             output_file: str,
             batch_size: int = 8,
             max_tokens: int = 64):

    # init model and tokenizer
    llm = LLM(model=model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=max_tokens, repetition_penalty=1.1)

    # load dataset
    examples = load_jsonl(input_file)

    # build prompts
    prompts = []
    for ex in examples:
        prompts.append(build_prompt(ex["question"], ex["choices"]))

    # generate in batches
    all_outputs = []
    
        
    outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
    for out in outputs:
        all_outputs.append(out.outputs[0].text.strip())

    # attach answers
    assert len(all_outputs) == len(examples)
    for ex, ans in zip(examples, all_outputs):
        ex["answer"] = ans

    save_jsonl(output_file, examples)
    print(f"Saved {len(examples)} results to {output_file}")


if __name__ == "__main__":
    model_id = "/home/sora/.cache/huggingface/DeepSeek-V2-Lite-Pruned"
    input_file = "/home/sora/llm/moe/data/test/vmlu.jsonl"
    output_file = "/home/sora/llm/moe/output/deepseek_pre/result.jsonl"

    evaluate(model_id, input_file, output_file,
             batch_size=8, max_tokens=512)
