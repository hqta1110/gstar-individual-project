import json
import re
from typing import List, Tuple, Optional
import os
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Make sure vllm is installed
try:
    from vllm import LLM, SamplingParams
    # 1. Import LoRARequest
    from vllm.lora.request import LoRARequest
except Exception as e:
    raise ImportError("vLLM import failed. Install with `pip install vllm` and ensure it's compatible with your environment.\n" + str(e))


# --- Helper functions (no changes needed here) ---

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
    # Extracts the first capital letter from the generated text
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
        if gold: # Only keep examples where we can find a gold answer
            prepared.append({"question": q, "choices": choices, "gold": gold, "subject": ex.get("subject")})
    return prepared

# --- Main Evaluation Function (Modified for LoRA) ---

def evaluate(
    model_id: str,
    adapter_path: str, # New argument for the adapter path
    tokenizer_path: str,
    examples: List[dict],
    max_tokens: int = 512,
) -> Tuple[int, int, List[dict]]:

    print("🚀 Initializing vLLM with LoRA enabled...")
    # 2. Initialize LLM with LoRA support
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        enable_lora=True,      # Enable LoRA
        max_loras=1,           # Max number of adapters to load at once
        max_lora_rank=16       # Set to your LoRA rank 'r'
    )
    print("✅ LLM Initialized.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, repetition_penalty=1.1)

    prompts = []
    for ex in examples:
        chat_prompt = [{"role": "user", "content": build_prompt(ex["question"], ex["choices"])}]
        prompts.append(tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True))

    # 3. Create a LoRARequest object
    # This tells vLLM which adapter to use for this generation batch.
    lora_request = LoRARequest(
        lora_name="expert_adapter",   # An arbitrary name for your adapter
        lora_int_id=1,                # A unique integer ID for this adapter
        lora_local_path=adapter_path
    )

    print(f"Generating responses for {len(prompts)} prompts...")
    # 4. Pass the lora_request to the generate method
    outputs = llm.generate(
        prompts,
        sampling_params,
        lora_request=lora_request
    )
    print("✅ Generation complete.")

    text_results = [output.outputs[0].text.strip() for output in outputs]
    assert len(text_results) == len(examples)

    # Process results and calculate accuracy
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


# --- Main execution block ---

if __name__ == "__main__":
    # Set GPU visibility
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # --- CONFIGURATION ---
    # Path to the base model (from your adapter_config.json)
    base_model_path = "/home/sora/llm/moe/ckpt/DeepSeek-V2-Lite"

    # Path to your trained adapter checkpoint directory
    # This is the directory containing 'adapter_config.json' and 'adapter_model.safetensors'
    adapter_path = "/home/sora/llm/moe/ckpt/deepseek_presft_qkv/checkpoint-776"

    # Path to save the final results
    output_file_path = "/home/sora/llm/moe/output/deepseek_presft_qkv/lora_mmlu_results.json"
    # --- END CONFIGURATION ---

    print("Loading and preparing MMLU dataset...")
    raw = load_mmlu()
    prepared = prepare_examples(raw)
    print(f"Prepared {len(prepared)} examples for evaluation.")

    correct, total, results = evaluate(
        model_id=base_model_path,
        adapter_path=adapter_path,
        tokenizer_path=base_model_path, # Use the base model's tokenizer
        examples=prepared,
        max_tokens=512 # Enough tokens for the model to reason and provide an answer
    )

    # Save detailed results to a file
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Detailed results saved to {output_file_path}")

    # Print final accuracy score
    accuracy = (correct / total * 100) if total > 0 else 0
    print("\n--- MMLU Evaluation Summary ---")
    print(f"Correct: {correct}")
    print(f"Total:   {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("-----------------------------\n")