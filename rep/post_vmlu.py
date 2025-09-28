import json
import os
import re
from typing import List, Dict
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # adjust if needed


def load_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: str, records: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def extract_choice_with_regex(text: str) -> str:
    """
    Extracts choice letter from model output.
    Matches both <answer> C </answer> and #### D.
    """
    if not text:
        return "None"

    match = re.search(r"(?:<answer>\s*([A-Z])\s*</answer>|####\s*([A-Z]))",
                      text, re.IGNORECASE)
    if match:
        return (match.group(1) or match.group(2)).upper()
    return "None"


def process_with_llm(samples: List[Dict],
                     model_id: str,
                     batch_size: int = 8,
                     max_tokens: int = 32) -> List[str]:
    """
    Run an eval model to map generated answers to choice letters.
    """
    print(f"Initializing LLM from model: {model_id}")
    llm = LLM(model=model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    sampling_params = SamplingParams(temperature=0.1, max_tokens=max_tokens)

    system_prompt = (
        "You will be given a question's choices and a generated answer. "
        "Your task is to extract from the generated answer the mentioned choice letter. You are not allowed randomly choose a letter. If the generated answer does not match any choice or empty, just reply None"
        "Respond ONLY in one of the following formats:\n"
        "- <answer> X </answer>\n"
        "- #### X\n"
        "where X is the correct choice letter (A, B, C, D, ...). "
        "Do not add explanations."
    )

    prompts = []
    for sample in samples:
        choices = sample.get("choices", [])
        gen_answer = sample.get("answer", "")
        choices_str = "\n".join(choices)

        user_prompt = f"Choices:\n{choices_str}\n\nGenerated answer:\n{gen_answer}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        prompts.append(messages)

    # Convert messages into plain prompts
    str_prompts = tokenizer.apply_chat_template(prompts,
                                                add_generation_prompt=True,
                                                tokenize=False)

    print(f"Processing {len(str_prompts)} samples with vLLM...")
    outputs = llm.generate(prompts=str_prompts,
                           sampling_params=sampling_params)

    raw_responses = [out.outputs[0].text.strip() for out in tqdm(outputs)]
    assert len(raw_responses) == len(samples)

    # Extract final choice letters
    return [extract_choice_with_regex(resp) for resp in raw_responses]


def main():
    MODEL_ID = "/home/sora/llm/moe/ckpt/Qwen2.5-32B-Instruct"
    INPUT_JSONL = "/home/sora/llm/moe/output/deepseek_pre/result.jsonl"   # from script 1
    OUTPUT_JSONL = "/home/sora/llm/moe/output/deepseek_pre/final_result.jsonl"

    print("Step 1: Loading input data...")
    samples = load_jsonl(INPUT_JSONL)
    print(f"Loaded {len(samples)} samples.")

    print("Step 2: Processing with eval LLM...")
    final_choices = process_with_llm(samples, MODEL_ID)

    print("Step 3: Keeping only id + final_choice...")
    stripped_results = []
    for sample, choice in zip(samples, final_choices):
        stripped_results.append({
            "id": sample.get("id"),
            "answer": choice
        })

    print("Step 4: Saving to file...")
    save_jsonl(OUTPUT_JSONL, stripped_results)
    print(f"Saved {len(stripped_results)} results to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
