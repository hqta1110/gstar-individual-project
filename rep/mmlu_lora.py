import json
import re
from typing import List, Tuple, Optional
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

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
    prompt += "\nAnswer:"
    return prompt

def normalize_pred(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r"([A-Z])", text)
    if match:
        return match.group(1)
    return None

def gold_letter_from_example(example) -> Optional[str]:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if "answer" in example:
        idx = int(example["answer"])
        if 0 <= idx < len(letters):
            return letters[idx]
    return None

def load_mmlu():
    ds = load_dataset("cais/mmlu", "all", trust_remote_code=True)
    return list(ds["test"])

def prepare_examples(raw_examples):
    prepared = []
    for ex in raw_examples:
        q = ex.get("question")
        choices = ex.get("choices")
        if not q or not choices:
            continue
        gold = gold_letter_from_example(ex)
        if gold:
            prepared.append({"question": q, "choices": choices, "gold": gold, "subject": ex.get("subject")})
    return prepared

def evaluate(
    model_id: str,
    adapter_path: str,
    tokenizer_path: str,
    examples: List[dict],
    max_tokens: int = 512,
) -> Tuple[int, int, List[dict]]:

    print("🚀 Initializing vLLM with LoRA enabled...")
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=16
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, repetition_penalty=1.1)

    prompts = []
    for ex in examples:
        chat_prompt = [{"role": "user", "content": build_prompt(ex["question"], ex["choices"])}]
        prompts.append(tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True))

    lora_request = LoRARequest(
        lora_name="expert_adapter",
        lora_int_id=1,
        lora_local_path=adapter_path
    )

    print(f"Generating responses for {len(prompts)} prompts...")
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request
    )
    print("✅ Generation complete.")

    text_results = [output.outputs[0].text.strip() for output in outputs]
    assert len(text_results) == len(examples)

    correct = 0
    total = 0
    processed_results = []
    for i in tqdm(range(len(examples)), desc="Evaluating"):
        pred = normalize_pred(text_results[i])
        ex = examples[i]
        gold = ex.get("gold")

        item = ex.copy()
        item["answer"] = text_results[i]
        item["predicted_letter"] = pred
        item["is_correct"] = False

        if pred and gold:
            total += 1
            if pred == gold:
                correct += 1
                item["is_correct"] = True
        processed_results.append(item)

    return correct, total, processed_results

if __name__ == "__main__":
    base_model_path = "/home/sora/llm/moe/ckpt/DeepSeek-V2-Lite"
    adapter_path = "/home/sora/llm/moe/ckpt/deepseek_presft_qkv/checkpoint-776"
    output_file_path = "/home/sora/llm/moe/output/deepseek_presft_qkv/lora_mmlu_results.json"

    print("Loading and preparing MMLU dataset...")
    raw = load_mmlu()
    prepared = prepare_examples(raw)
    print(f"Prepared {len(prepared)} examples for evaluation.")

    correct, total, results = evaluate(
        model_id=base_model_path,
        adapter_path=adapter_path,
        tokenizer_path=base_model_path,
        examples=prepared,
        max_tokens=512
    )

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Detailed results saved to {output_file_path}")

    accuracy = (correct / total * 100) if total > 0 else 0
    print("\n--- MMLU Evaluation Summary ---")
    print(f"Correct: {correct}")
    print(f"Total:   {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("-----------------------------\n")
