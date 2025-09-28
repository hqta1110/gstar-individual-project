import json
import re
from typing import List, Tuple, Optional, Dict
import os
from tqdm import tqdm

# Ensure you have the correct GPU device(s) visible
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from datasets import load_dataset
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError as e:
    raise ImportError("vLLM import failed. Install with `pip install vllm` and ensure it's compatible with your environment.\n" + str(e))


system_prompt = """You are a helpful assistant that answers True/False questions based on the given passage.
"""

# Loader for the BoolQ dataset
def load_boolq() -> List[Dict]:
    """Loads the BoolQ validation split."""
    print("Loading BoolQ dataset...")
    ds = load_dataset("boolq")
    examples = list(ds["validation"])
    print(f"Loaded {len(examples)} examples from BoolQ validation set.")
    return examples

# Extract True/False answer from text
def extract_bool_from_text(text: str) -> Optional[bool]:
    """Extracts the final boolean answer after '####'."""
    if not text:
        return None
    match = re.search(r'####\s*(true|false)', text, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true"
    return None

# Build prompt
def build_boolq_prompt(passage: str, question: str) -> str:
    return f"Passage: {passage}\nQuestion: {question}\nAnswer:"

# Evaluation
def evaluate_boolq(
    model_id: str,
    examples: List[dict],
    adapter_path: str = None
) -> Tuple[int, int, List[Dict]]:
    llm = LLM(model=model_id, trust_remote_code=True, enable_lora=True, max_loras=1, max_lora_rank=16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=512, repetition_penalty=1.1)
    lora_request = LoRARequest(
        lora_name="expert_adapter",
        lora_int_id=1,
        lora_local_path=adapter_path
    )
    total = 0
    correct = 0

    # Build prompts
    prompts = [build_boolq_prompt(ex["passage"], ex["question"]) for ex in examples]
    final_prompts = []
    for prompt in prompts:
        final_prompts.append([
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt}
        ])
    chat_prompts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in final_prompts
    ]
    outputs = llm.generate(chat_prompts, sampling_params, lora_request=lora_request)

    results = []

    print("Evaluating results...")
    for i, output in tqdm(enumerate(outputs), total=len(outputs)):
        generated_text = output.outputs[0].text
        gold_answer_bool = bool(examples[i]["answer"])

        predicted_bool = extract_bool_from_text(generated_text)

        results.append({
            "passage": examples[i]["passage"],
            "question": examples[i]["question"],
            "gold": gold_answer_bool,
            "answer": generated_text,
            "predicted_answer": predicted_bool,
        })

        if predicted_bool is not None:
            total += 1
            if predicted_bool == gold_answer_bool:
                correct += 1

    return correct, total, results


if __name__ == "__main__":
    model_id = "/home/sora/llm/moe/ckpt/DeepSeek-V2-Lite"

    dataset_examples = load_boolq()
    adapter_path = "/home/sora/llm/moe/ckpt/deepseek_presft_qkv/checkpoint-776"

    correct_count, total_count, detailed_results = evaluate_boolq(model_id, dataset_examples, adapter_path=adapter_path)

    output_filename = "/home/sora/llm/moe/output/deepseek_presft_qkv/raw_result_boolq.json"
    print(f"\nSaving detailed results to {output_filename}...")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)

    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print("\n--- Evaluation Complete ---")
        print(f"Correct: {correct_count}")
        print(f"Total:   {total_count}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("---------------------------")
    else:
        print("Evaluation could not be completed. No valid answers found.")
