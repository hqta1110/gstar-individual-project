import json
import re
from typing import List, Tuple, Optional, Dict
import os
from tqdm import tqdm


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from datasets import load_dataset
from transformers import AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
except ImportError as e:
    raise ImportError("vLLM import failed. Install with `pip install vllm` and ensure it's compatible with your environment.\n" + str(e))


system_prompt = """Answer the question.
"""

# Loader
def load_openbookqa() -> List[Dict]:
    """Loads the OpenBookQA validation split."""
    print("Loading OpenBookQA dataset...")
    ds = load_dataset("openbookqa", "main")
    examples = list(ds["validation"])
    print(f"Loaded {len(examples)} examples from OpenBookQA validation set.")
    return examples

# Extract predicted answer
def extract_choice_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    match = re.search(r'####\s*([A-Da-d])', text)
    if match:
        return match.group(1).upper()
    return None

# Build prompt
def build_openbookqa_prompt(question: str, choices: list[str], letters: list[str]) -> str:
    choices_str = "\n".join([f"{k}. {v}" for k, v in zip(choices, letters)])
    return f"Question: {question}\nOptions:\n{choices_str}\nYour answer: "

# Evaluation
def evaluate_openbookqa(
    model_id: str,
    examples: List[dict],
    adapter_path: str = None
) -> Tuple[int, int, List[Dict]]:
    llm = LLM(model=model_id, trust_remote_code=True, enable_lora=True, max_loras=1, max_lora_rank=16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=512, repetition_penalty=1.2)
    lora_request = LoRARequest(
        lora_name="expert_adapter",
        lora_int_id=1,
        lora_local_path=adapter_path
    )
    total = 0
    correct = 0

    prompts = [
        build_openbookqa_prompt(ex["question_stem"], ex["choices"]["text"], ex['choices']['label'])
        for ex in examples
    ]
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
        gold_label = examples[i]["answerKey"].upper()

        predicted_label = extract_choice_from_text(generated_text)

        results.append({
            "question": examples[i]["question_stem"],
            "choices": examples[i]["choices"]["text"],
            "gold": gold_label,
            "answer": generated_text,
            "predicted_label": predicted_label,
        })

        if predicted_label is not None:
            total += 1
            if predicted_label == gold_label:
                correct += 1

    return correct, total, results


if __name__ == "__main__":
    model_id = "/home/sora/llm/moe/ckpt/DeepSeek-V2-Lite"

    dataset_examples = load_openbookqa()
    adapter_path = "/home/sora/llm/moe/ckpt/deepseek_presft_qkv/checkpoint-776"
    correct_count, total_count, detailed_results = evaluate_openbookqa(model_id, dataset_examples, adapter_path=adapter_path)

    output_filename = "/home/sora/llm/moe/output/deepseek_presft_qkv/raw_result_openbookqa.json"
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
