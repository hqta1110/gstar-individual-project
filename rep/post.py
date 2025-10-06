import json
import re
from typing import List, Optional, Dict
import os
from tqdm import tqdm

# Ensure you have the correct GPU device(s) visible
# os.environ['CUDA_VISIBLE_DEVICES'] = '2' 

# Import necessary libraries
try:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
except ImportError as e:
    raise ImportError("vLLM or Transformers import failed. Please ensure they are installed (`pip install vllm transformers`) and compatible with your environment.\n" + str(e))

def load_json_data(filepath: str) -> List[Dict]:
    """Loads a list of samples from a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return []

def extract_choice_with_regex(text: str) -> str:
    """
    Extracts the multiple-choice letter from the model's output using regex.
    Handles formats like <answer> A </answer> and <answer> A. {text} </answer>.
    """
    if not text:
        return "None"
    
    match = re.search(r"<answer>\s*(True|False|[A-Z]|-?\d+(?:\.\d+)?)(?:\s*\.?\s*.*)?</answer>", text, re.IGNORECASE)

    
    if match:
        return match.group(1).upper()
    else:
        # If the pattern is not found, return "None" as per the requirement.
        return "None"
def build_prompt(question: str, choices: List[str]) -> str:
    prompt = "Question:\n" + question.strip() + "\n\nChoices:\n" + choices.strip()
    return prompt

def process_with_llm(
    samples: List[Dict],
    model_id: str,
    tokenizer_path: Optional[str] = None
) -> List[str]:
    """
    Uses vLLM to process the 'answer' field of each sample and extract the choice.
    """
    print(f"Initializing LLM from model path: {model_id}")
    llm = LLM(model=model_id, trust_remote_code=True, tensor_parallel_size=1, max_model_len=29376)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path else model_id)

    sampling_params = SamplingParams(temperature=0.1, max_tokens=32)
    
    system_prompt = (
        "You will receive a generated answer."
        # "You need to respond true with the format {{<answer> True </answer>}} if the generated answer contains Yes, otherwise respond false with the format {{<answer> False </answer>}}."
        # "You will receive a ground truth answer and a generated answer."
        "Your task is to evaluate whether the generated answer matches the ground truth."
        # "It is a match if the generated answer mentioned about the final result in the ground truth, even briefly or detailed."
        # "If the ground truth is 'True' or 'False', it is a match if the generated answer agree or disagree with it."
        "It is a match if the final calculation result/final conclusion in the generated answer match with the ground truth."
        # "It is a match if the generated answer have the same idea as the ground truth. E.g: ground truth is true and the generated answer is 'Yes, ...', it is a match. Or ground truth is false and the generated answer is 'No, ...', it is a match."
        # "It is a match if the generated answer and the ground truth are both agree or disagree. For example, if the ground truth is 'True' and the generated answer is 'Yes, .....', it is a match."
        "If it matches, respond with the format {{<answer> True </answer>}} otherwise respond with {{<answer> False </answer>}}."
    )


    
    # Prepare all prompts for batch processing
    prompts = []
    golds = []
    for sample in samples:
        user_prompt = sample.get("answer", "")
        gold = sample.get("full_choice")
        # choices = sample.get("choices", "")
        golds.append(gold)
        # Create the chat message format
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Ground truth: " + str(gold) + "\nGenerated answer: " + user_prompt}
            # {"role": "user", "content": "\nGenerated answer: " + user_prompt}

        ]
        prompts.append(messages)

    # Apply the chat template to all prompts
    prompt_token_ids = tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False)

    print(f"Processing {len(prompts)} samples with vLLM...")
    outputs = llm.generate(prompt_token_ids, sampling_params=sampling_params)
    raw_responses = [output.outputs[0].text for output in tqdm(outputs, desc="Extracting responses")]
    assert len(golds) == len(raw_responses)
    
    # Use regex to get the final choice
    extracted_choices = [extract_choice_with_regex(resp) for resp in raw_responses]
    total = 0
    correct = 0
    # for i in range(len(extracted_choices)):
    #     if extracted_choices[i] and extracted_choices[i] != "None":
    #         total += 1
    #         if extracted_choices[i] == golds[i]:
    #         # for gsm8k, use the condition below
    #         # if float(extracted_choices[i]) - golds[i] < 1e-4:
    #             correct += 1
    for i in range(len(extracted_choices)):
        if extracted_choices[i] != "None":
            total += 1
            if extracted_choices[i].upper().strip() == "TRUE":
                correct += 1
    return extracted_choices, total, correct

def save_results_to_json(filepath: str, results: List[Dict]):
    """Saves the final list of results to a JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Successfully saved {len(results)} results to {filepath}")

if __name__ == "__main__":
    MODEL_ID = "/home/sora/llm/moe/ckpt/Qwen2.5-32B-Instruct" 
    
    # Specify the path to the JSON file generated by your first script
    INPUT_JSON_PATH = "/home/sora/llm/moe/output/deepseek_pre_nonse/raw_result_baseline_arc.json"
    
    # Specify the path for the final output
    OUTPUT_JSON_PATH = "/home/sora/llm/moe/output/deepseek_pre_nonse/final_choice_arc.json"
    
    # --- Main Execution Logic ---
    print("Step 1: Loading input data...")
    input_samples = load_json_data(INPUT_JSON_PATH)
    
    if input_samples:
        print(f"Loaded {len(input_samples)} samples.")
        
        print("\nStep 2: Processing answers with the LLM to extract choices...")
        model_choices, total, correct = process_with_llm(input_samples, MODEL_ID)
        print(f"Total: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {correct/total}")
        print("\nStep 3: Formatting final results...")
        final_results = []
        # Ensure we have the same number of choices as original samples
        if len(input_samples) == len(model_choices):
            i = 0
            for sample, choice in zip(input_samples, model_choices):
                final_results.append({
                    "index": i,
                    "gold": sample.get("gold", "N/A"), 
                    "choice": choice,
                    "raw": sample.get("answer", "")
                })
                i += 1
        else:
            print("Error: Mismatch between number of input samples and model outputs.")

        print("\nStep 4: Saving final results...")
        save_results_to_json(OUTPUT_JSON_PATH, final_results)
        
        print("\nScript finished successfully!")