import json
import re
from typing import List, Tuple, Optional
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # Adjust based on your setup
from datasets import load_dataset
from transformers import AutoTokenizer
try:
    from vllm import LLM, SamplingParams
except Exception as e:
    raise ImportError("vLLM import failed. Install with `pip install vllm` and ensure it's compatible with your environment.\n" + str(e))

def build_prompt(question: str, choices: str) -> str:
    
    prompt = "Question:\n" + question.strip() + "\n\nChoices:\n" + choices.strip() + "\nAnswer: "
    return prompt


def normalize_pred(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"<answer>\\s*([A-Z])\\s*</answer>", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return None


def load_med():
    ds = load_dataset("lavita/medical-qa-datasets", "med-qa-en-4options-source")
    examples = list(ds["test"])
    return examples


def prepare_examples(raw_examples):
    prepared = []
    for ex in raw_examples:
        question = ex['question']
        gold = ex['answer_idx']
        choices = []
        answer = ""
        for pair in ex['options']:
            answer += pair['key'] + ". " + pair['value'] + "\n"
        prepared.append({"question":question, "choices": answer, "gold":gold})
    return prepared


def chunkify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def evaluate(model_id: str,
             tokenizer: Optional[str],
             examples: List[dict],
             batch_size: int = 8,
             max_tokens: int = 64,
             verbose: bool = True) -> Tuple[int, int]:

    llm = LLM(model=model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer if tokenizer else model_id)
    sampling_params = SamplingParams(temperature=0.1, max_tokens=max_tokens, repetition_penalty=1.1)

    total = 0
    correct = 0

    prompts = []
    for batch in examples:
        prompts.append([{"role": "user", "content": build_prompt(batch["question"], batch["choices"])}])
    prompt_token_ids = [tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False) for message in prompts]
    outputs = llm.generate(prompt_token_ids, sampling_params=sampling_params)
    text_results = [output.outputs[0].text for output in outputs]
    assert len(text_results) == len(prompts)
    assert len(text_results) == len(examples)

    return correct, total, text_results


if __name__ == "__main__":
    model_id = "/home/sora/.cache/huggingface/DeepSeek-V2-Lite-Pruned"
    batch_size = 8
    max_tokens = 512

    raw = load_med()
    prepared = prepare_examples(raw)
    correct, total, results = evaluate(model_id, model_id, prepared, batch_size=batch_size, max_tokens=max_tokens)
    combined = []
    for rec, ans in zip(prepared, results):
        item = rec.copy()      
        item["answer"] = ans
        combined.append(item)
    with open("/home/sora/llm/moe/output/deepseek_prunese_again_11/raw_result_med.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    # print("RESULT: ", correct, "/", total, "accuracy=", (correct / total if total else 0.0))
