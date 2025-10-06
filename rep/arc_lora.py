import json
import re
from typing import List, Tuple, Optional
import os
from tqdm import tqdm

# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # Adjust based on your setup
from datasets import load_dataset
from transformers import AutoTokenizer
try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except Exception as e:
    raise ImportError("vLLM import failed. Install with `pip install vllm` and ensure it's compatible with your environment.\n" + str(e))

def build_prompt(question: str, choices: List[str]) -> str:
    prompt = "Question:\n" + question.strip() + "\n\nChoices:\n"
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, c in enumerate(choices):
        prompt += f"{letters[i]}. {c.strip()}\n"
    return prompt

def normalize_pred(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"<answer>\s*([A-Z])\s*</answer>", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def gold_letter_from_example(example) -> Optional[str]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    if "answerKey" in example and isinstance(example["answerKey"], str):
        ans_key = example["answerKey"].strip().upper()
        if ans_key in letters:
            return ans_key
        if ans_key.isdigit():
            return letters[int(ans_key) - 1]

    if "answer" in example and isinstance(example["answer"], str):
        a = example["answer"].strip()
        if len(a) == 1 and a.upper() in letters:
            return a.upper()
        if "choices" in example:
            for i, c in enumerate(example["choices"]):
                if c.strip().lower() == a.lower():
                    return letters[i]
    if "answer" in example and isinstance(example["answer"], (int, float)):
        idx = int(example["answer"])
        return letters[idx]
    if "labels" in example and isinstance(example["labels"], (list, tuple)):
        try:
            return letters[int(example["labels"][0])]
        except Exception:
            pass
    for k in ["idx", "label", "answer_idx", "gold"]:
        if k in example and isinstance(example[k], (int, float)):
            return letters[int(example[k])]
    if "target" in example and "choices" in example:
        t = example["target"].strip().lower()
        for i, c in enumerate(example["choices"]):
            if c.strip().lower() == t:
                return letters[i]
    return None

def load_arc():
    """Loads the ARC-Challenge test split."""
    print("Loading ARC-Challenge dataset...")
    # use the 'test' split for evaluation
    ds = load_dataset("ai2_arc", "ARC-Challenge")
    examples = list(ds["test"])
    print(f"Loaded {len(examples)} examples from ARC-Challenge test set.")
    return examples

def prepare_examples(raw_examples):
    prepared = []
    print("Preparing examples...")
    for ex in tqdm(raw_examples):
        q = None
        choices = None

        for k in ["question", "query", "input", "stem"]:
            if k in ex:
                q = ex[k]
                break
        if not q and "prompt" in ex:
            q = ex["prompt"]

        if "choices" in ex and isinstance(ex["choices"], dict) and "text" in ex["choices"]:
            choices = ex["choices"]["text"]

        elif not choices:
            for k in ["choices", "options", "answers"]:
                if k in ex and isinstance(ex[k], (list, tuple)):
                    choices = ex[k]
                    break
        
        if not q or not choices:
            continue

        gold = gold_letter_from_example(ex)
        if gold: 
            prepared.append({"question": q, "choices": choices, "gold": gold})
    print(f"Prepared {len(prepared)} valid examples.")
    return prepared

def evaluate(model_id: str,
             tokenizer: Optional[str],
             examples: List[dict],
             batch_size: int = 8,
             max_tokens: int = 64,
             verbose: bool = True,
             adapter_path: str = None) -> Tuple[int, int, List[str]]:

    llm = LLM(model=model_id, trust_remote_code=True, enable_lora=True, max_loras=1, max_lora_rank=16)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer if tokenizer else model_id)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=max_tokens, repetition_penalty=1.1)

    total = 0
    correct = 0
    lora_request = LoRARequest(
        lora_name="expert_adapter",
        lora_int_id=1,
        lora_local_path=adapter_path
    )
    prompts = []
    for ex in examples:
        prompts.append([{"role": "user", "content": build_prompt(ex["question"], ex["choices"])}])
    
    prompt_token_ids = tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False)
    
    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = llm.generate(prompt_token_ids, sampling_params=sampling_params, lora_request=lora_request)
    
    text_results = [output.outputs[0].text for output in outputs]
    assert len(text_results) == len(prompts)
    assert len(text_results) == len(examples)
    
    return correct, total, text_results

if __name__ == "__main__":
    # --- Configuration ---
    model_id = "/home/sora/llm/moe/ckpt/DeepSeek-V2-Lite"
    batch_size = 8
    max_tokens = 512 # Keep this high enough for reasoning
    adapter_path = "/home/sora/llm/moe/ckpt/deepseek_presft_qkv/checkpoint-776"
    # --- Main Execution Logic ---
    raw = load_arc()
    prepared = prepare_examples(raw)

    correct, total, results = evaluate(model_id, model_id, prepared, batch_size=batch_size, max_tokens=max_tokens, adapter_path=adapter_path)
    
    combined = []
    for rec, ans in zip(prepared, results):
        item = rec.copy()      
        item["answer"] = ans
        combined.append(item)
    
    output_filename = "/home/sora/llm/moe/output/deepseek_presft_qkv/lora_arc_results_direct.json"
    print(f"Saving raw results to {output_filename}...")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    
    print("Script finished successfully.")