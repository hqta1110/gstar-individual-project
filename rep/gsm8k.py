import json
import re
from typing import List, Tuple, Optional, Dict
import os
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
except ImportError as e:
    raise ImportError("vLLM import failed. Install with `pip install vllm` and ensure it's compatible with your environment.\n" + str(e))

system_prompt = """Answer the question"""

def load_gsm8k() -> List[Dict]:
    print("Loading GSM8K dataset...")
    ds = load_dataset("gsm8k", "main")
    examples = list(ds["test"])
    print(f"Loaded {len(examples)} examples from GSM8K test set.")
    return examples

def extract_answer_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    match = re.search(r'####\s*([\d,.-]+)', text)
    if match:
        answer_str = match.group(1)
        try:
            cleaned_str = answer_str.replace(",", "")
            return float(cleaned_str)
        except ValueError:
            return None
    return None

def build_gsm8k_prompt(question: str) -> str:
    return f"Question: {question}"

def evaluate_gsm8k(
    model_id: str,
    examples: List[dict]
) -> Tuple[int, int, List[Dict]]:
    llm = LLM(model=model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=512, repetition_penalty=1.1)
    total = 0
    correct = 0
    prompts = [build_gsm8k_prompt(ex["question"]) for ex in examples]
    final_prompts = []
    for prompt in prompts:
        final_prompts.append([{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': prompt}])
    chat_prompts = [
        tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        for p in final_prompts
    ]
    outputs = llm.generate(chat_prompts, sampling_params)
    results_with_answers = []
    print("Evaluating results...")
    for i, output in tqdm(enumerate(outputs), total=len(outputs)):
        generated_text = output.outputs[0].text
        gold_answer_text = examples[i]["answer"]
        predicted_answer = extract_answer_from_text(generated_text)
        gold_answer = extract_answer_from_text(gold_answer_text)
        results_with_answers.append({
            "question": examples[i]["question"],
            "gold": gold_answer_text,
            "gold_answer_extracted": gold_answer,
            "answer": generated_text,
            "predicted_answer_extracted": predicted_answer,
        })
        if predicted_answer is not None and gold_answer is not None:
            total += 1
            if abs(predicted_answer - gold_answer) < 1e-4:
                correct += 1
    return correct, total, results_with_answers

if __name__ == "__main__":
    model_id = "/home/sora/.cache/huggingface/DeepSeek-V2-Lite-Pruned"
    dataset_examples = load_gsm8k()
    correct_count, total_count, detailed_results = evaluate_gsm8k(model_id, dataset_examples)
    output_filename = "/home/sora/llm/moe/output/deepseek_prunese_again_11/raw_result_gsm8k.json"
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
