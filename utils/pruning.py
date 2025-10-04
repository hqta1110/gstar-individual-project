import json
import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoConfig
from safetensors.torch import save_file
from tqdm import tqdm

# ===================================================================
# >> CONFIGURATION <<
# Please edit the variables in this section before running the script.
# ===================================================================

# The name or local path of the Hugging Face model you want to prune.
# Example: "deepseek-ai/deepseek-moe-16b-base"
MODEL_TO_PRUNE = "/home/sora/llm/moe/ckpt/DeepSeek-V2-Lite"

# The path to your JSON file containing the list of Super Experts.
# Example: "./my_super_experts.json"
SUPER_EXPERTS_FILE = "/home/sora/llm/moe/output_experts/deepseek_pre/deepseek_pre_wiki/super_experts_info/super_experts_report/Super Experts Report.json"

# The directory where the new, pruned model will be saved.
# Example: "./pruned_model_output"
OUTPUT_DIRECTORY = "/home/sora/.cache/huggingface/DeepSeek-V2-Lite-Pruned"

# ===================================================================
# Script Logic (No need to edit below this line)
# ===================================================================

def get_weight_key(model_architecture, layer_index, expert_index):
    """
    Constructs the correct weight key for the state dictionary based on the model architecture.
    This is crucial as different MoE models have different naming conventions.
    """
    if "MixtralForCausalLM" in model_architecture:
        # Mixtral uses w1, w2, w3 for its expert layers
        return f"model.layers.{layer_index}.block_sparse_moe.experts.{expert_index}.w3.weight"
    elif "DeepseekV2ForCausalLM" in model_architecture or "Qwen3MoeForCausalLM" in model_architecture or "OlmoeForCausalLM" in model_architecture:
        # DeepSeek, Qwen2-MoE, and OlmoE use a standard gate/up/down projection setup
        return f"model.layers.{layer_index}.mlp.experts.{expert_index}.down_proj.weight"
    else:
        # Add other model architectures here if needed
        raise NotImplementedError(
            f"Model architecture '{model_architecture}' is not supported. "
            "Please add the correct weight key naming convention to the `get_weight_key` function."
        )

def prune_and_save_model(model_name, super_experts_json_path, output_path):
    """
    Loads a model, prunes specified Super Experts by zeroing their weights,
    and saves the pruned model to a new directory.

    Args:
        model_name (str): The name or path of the Hugging Face model to prune.
        super_experts_json_path (str): Path to the JSON file containing the list of Super Experts.
        output_path (str): The directory where the pruned model will be saved.
    """
    print(f"Loading model: {model_name}...")
    # Load the model on the CPU to avoid large GPU memory usage during modification.
    # We use bfloat16 for consistency with the original scripts.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="cuda"
    )
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_architecture = model_config.architectures[0]
    print(f"Detected model architecture: {model_architecture}")

    print(f"Loading Super Experts list from: {super_experts_json_path}...")
    with open(super_experts_json_path, 'r') as f:
        super_experts = json.load(f)

    # Get the state dictionary of the model
    state_dict = model.state_dict()
    pruned_count = 0

    print("Pruning Super Experts...")
    for expert_info in tqdm(super_experts, desc="Pruning Experts"):
        layer_index = expert_info["layer_index"]
        expert_index = expert_info["expert_index"]

        # The paper identifies that nullifying the 'down_proj' (or equivalent)
        # is sufficient to disable the expert's contribution.
        try:
            key_to_prune = get_weight_key(model_architecture, layer_index, expert_index)
        except NotImplementedError as e:
            print(f"Error: {e}")
            return

        if key_to_prune in state_dict:
            # Zero out the weight tensor in-place
            weight_tensor = state_dict[key_to_prune]
            torch.nn.init.zeros_(weight_tensor)
            print(f"  - Pruned (zeroed out) weight: {key_to_prune}")
            pruned_count += 1
        else:
            print(f"  - WARNING: Weight key not found, skipping: {key_to_prune}")

    if pruned_count == 0:
        print("Warning: No experts were pruned. Please check your JSON file and model architecture.")
        return

    print(f"\nPruning complete. Total experts pruned: {pruned_count}")

    print(f"Saving pruned model to: {output_path}...")
    # The modern way to save is using save_pretrained, which handles the tokenizer,
    # config, and model weights (using safetensors by default).
    model.save_pretrained(output_path)
    # Also save the tokenizer for completeness
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    
    print("✅ Pruned model saved successfully.")


if __name__ == '__main__':
    # Ensure the configuration variables are not placeholder values before running
    if "path/to/your" in MODEL_TO_PRUNE or "path/to/your" in SUPER_EXPERTS_FILE:
        print("❌ ERROR: Please update the configuration variables at the top of the script before running.")
    else:
        prune_and_save_model(MODEL_TO_PRUNE, SUPER_EXPERTS_FILE, OUTPUT_DIRECTORY)