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
    # 1. Import LoraRequest for direct adapter loading
    from vllm.lora.request import LoRARequest
except Exception as e:
    raise ImportError("vLLM import failed. Install with `pip install vllm` and ensure it's compatible with your environment.\n" + str(e))


# --- Helper functions (no changes needed) ---

def build_prompt(question: str, choices: str) -> str:
    """Creates a clear prompt for the model."""
    prompt = "Question:\n" + question.strip() + "\n\nChoices:\n" + choices.strip()
    prompt += "\n\nAnswer:"
    return prompt

def normalize_pred(text: str) -> Optional[str]:
    """Extracts the first capital letter (A, B, C, D) from the model's output."""
    if not text:
        return None
    match = re.search(r"([A-D])", text)
    if match:
        return match.group(1)
    return None

def load_med():
    """Loads the MedQA test set."""
    ds = load_dataset("lavita/medical-qa-datasets", "med-qa-en-4options-source", trust_remote_code=True)
    return list(ds["test"])

def prepare_examples(raw_examples):
    """Formats the raw dataset into a clean structure for evaluation."""
    prepared = []
    for ex in raw_examples:
        question = ex['question']
        gold = ex['answer_idx']
        options_str = ""
        for pair in ex['options']:
            options_str += f"{pair['key']}. {pair['value']}\n"
        prepared.append({"question": question, "choices": options_str, "gold": gold})
    return prepared

# --- Main Evaluation Function (Modified for Direct LoRA Loading) ---

def evaluate(
    model_id: str,
    adapter_path: str, # New argument for the adapter path
    tokenizer_path: str,
    examples: List[dict],
    max_tokens: int = 512,
) -> Tuple[int, int, List[dict]]:
    """
    Evaluates the model by loading the LoRA adapter directly at runtime.
    """
    print("🚀 Initializing vLLM with LoRA enabled...")
    # 2. Initialize LLM with LoRA support enabled
    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        enable_lora=True,      # Enable LoRA
        max_loras=1,           # Max number of adapters to load
        max_lora_rank=16       # Set to your LoRA rank 'r'
    )
    print("✅ LLM Initialized.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=max_tokens, repetition_penalty=1.1)
    
    prompts = []
    for ex in examples:
        chat_prompt = [{"role": "user", "content": build_prompt(ex["question"], ex["choices"])}]
        prompts.append(tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True))

    # 3. Create a LoRARequest object pointing to your adapter
    lora_request = LoRARequest(
        lora_name="expert_adapter",
        lora_int_id=1,
        lora_local_path=adapter_path
    )
    
    print(f"Generating responses for {len(prompts)} prompts with LoRA adapter...")
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
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    # --- CONFIGURATION ---
    # Path to the original base model
    BASE_MODEL_PATH = "/home/sora/llm/moe/ckpt/DeepSeek-V2-Lite"

    # Path to your trained adapter checkpoint directory
    ADAPTER_PATH = "/home/sora/llm/moe/ckpt/deepseek_presft_qkv/checkpoint-776"

    # Path to save the final results
    OUTPUT_FILE_PATH = "/home/sora/llm/moe/output/deepseek_presft_qkv/lora_medqa_results_direct.json"
    # --- END CONFIGURATION ---

    print("Loading and preparing MedQA dataset...")
    raw = load_med()
    prepared = prepare_examples(raw)
    print(f"Prepared {len(prepared)} examples for evaluation.")

    # Call the evaluate function with paths to both the base model and the adapter
    correct, total, results = evaluate(
        model_id=BASE_MODEL_PATH,
        adapter_path=ADAPTER_PATH,
        tokenizer_path=BASE_MODEL_PATH,
        examples=prepared,
        max_tokens=512
    )

    # Save detailed results to a JSON file
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Detailed results saved to {OUTPUT_FILE_PATH}")

    # Print the final accuracy score
    accuracy = (correct / total * 100) if total > 0 else 0
    print("\n--- MedQA Evaluation Summary ---")
    print(f"Correct: {correct}")
    print(f"Total:   {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("----------------------------\n")