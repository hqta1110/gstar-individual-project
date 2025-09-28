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

system_prompt = """Answer the question'.
"""

# NEW FUNCTION: Loader for the GSM8K dataset.
def load_gsm8k() -> List[Dict]:
    """Loads the GSM8K test split."""
    print("Loading GSM8K dataset...")
    # We use the 'test' split for evaluation
    ds = load_dataset("gsm8k", "main")
    examples = list(ds["test"])
    print(f"Loaded {len(examples)} examples from GSM8K test set.")
    return examples

# NEW FUNCTION: Extracts the final numerical answer from text.
def extract_answer_from_text(text: str) -> Optional[float]:
    """Extracts the numerical answer after '####'."""
    if not text:
        return None
    
    # Find the last occurrence of '####' and get the text after it
    match = re.search(r'####\s*([\d,.-]+)', text)
    if match:
        answer_str = match.group(1)
        # Remove commas and convert to float
        try:
            cleaned_str = answer_str.replace(",", "")
            return float(cleaned_str)
        except ValueError:
            return None
    return None

# NEW FUNCTION: Builds a few-shot prompt for GSM8K.
def build_gsm8k_prompt(question: str) -> str:
    """Creates a few-shot prompt to guide the model's reasoning."""
    
    # Few-shot examples to show the desired Chain-of-Thought format
    
    return f"Question: {question}\nAnswer:"


# MODIFIED: The core evaluation function adapted for GSM8K.
def evaluate_gsm8k(
    model_id: str,
    examples: List[dict],
    adapter_path: str = None,
) -> Tuple[int, int, List[Dict]]:
    """
    Evaluates the model on the GSM8K dataset.
    """
    llm = LLM(model=model_id, trust_remote_code=True, enable_lora=True, max_loras=1, max_lora_rank=16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # # For reasoning, a lower temperature is better. max_tokens needs to be sufficient for steps.
    sampling_params = SamplingParams(temperature=0.1, max_tokens=512, repetition_penalty=1.1)
    lora_request = LoRARequest(
        lora_name="expert_adapter",
        lora_int_id=1,
        lora_local_path=adapter_path
    )
    total = 0
    correct = 0
    
    # Build all prompts
    prompts = [build_gsm8k_prompt(ex["question"]) for ex in examples]
    final_prompts = []
    for prompt in prompts:
        final_prompts.append([{'role': 'user', 'content': prompt}])
    chat_prompts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in final_prompts
    ]
    outputs = llm.generate(chat_prompts, sampling_params, lora_request=lora_request)

    
    results_with_answers = []
    
    print("Evaluating results...")
    for i, output in tqdm(enumerate(outputs), total=len(outputs)):
        generated_text = output.outputs[0].text
        gold_answer_text = examples[i]["answer"]
        
        # Extract the numerical answers
        predicted_answer = extract_answer_from_text(generated_text)
        gold_answer = extract_answer_from_text(gold_answer_text)
        
        # Store results for analysis
        results_with_answers.append({
            "question": examples[i]["question"],
            "gold": gold_answer_text,
            "gold_answer_extracted": gold_answer,
            "answer": generated_text,
            "predicted_answer_extracted": predicted_answer,
        })
        
        # Check for correctness
        if predicted_answer is not None and gold_answer is not None:
            total += 1
            if abs(predicted_answer - gold_answer) < 1e-4: # Compare floats
                correct += 1
                
    return correct, total, results_with_answers


if __name__ == "__main__":
    # --- Configuration ---
    model_id = "/home/sora/llm/moe/ckpt/DeepSeek-V2-Lite"
    
    # --- Main Execution Logic ---
    # 1. Load the dataset
    dataset_examples = load_gsm8k()
    ADAPTER_PATH = "/home/sora/llm/moe/ckpt/deepseek_presft_qkv/checkpoint-776"
    
    # 2. Run evaluation
    correct_count, total_count, detailed_results = evaluate_gsm8k(model_id, dataset_examples, adapter_path=ADAPTER_PATH)
    
    # 3. Save the detailed results to a file for inspection
    output_filename = "/home/sora/llm/moe/output/deepseek_presft_qkv/lora_gsm8k_results_direct.json"
    print(f"\nSaving detailed results to {output_filename}...")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)
        
    # 4. Print the final accuracy
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        print("\n--- Evaluation Complete ---")
        print(f"Correct: {correct_count}")
        print(f"Total:   {total_count}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("---------------------------")
    else:
        print("Evaluation could not be completed. No valid answers found.")