import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from SuperExpertLora import SuperExpertLora
from dataclasses import dataclass
from typing import Dict, List
import torch

@dataclass
class CausalDataCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        batch_input = self.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        )
        batch_labels = self.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
        )["input_ids"]

        # Mask out pad tokens in labels
        batch_labels[batch_labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": batch_input["input_ids"],
            "attention_mask": batch_input["attention_mask"],
            "labels": batch_labels,
        }

# --- Configuration ---
class TrainingConfig:
    # Environment
    cuda_visible_devices = "1"

    # Model & Data
    model_id = "/home/sora/llm/moe/ckpt/DeepSeek-V2-Lite"
    dataset_name = "/home/sora/llm/moe/data/train/full.json"  # JSON file with list of {question, answer, explain}
    output_dir = "/home/sora/llm/moe/ckpt/hehe"
    max_seq_length = 1024

    # LoRA
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05

    # Training
    epochs = 2
    batch_size = 16
    grad_accum_steps = 8
    learning_rate = 1e-4
    warmup_steps = 10
    logging_steps = 1
    save_steps = 10

# --- End Config ---
config = TrainingConfig()

# Set device
os.environ["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices

# --- 1. Load Model and Tokenizer ---
print(f"Loading base model: {config.model_id}")
tokenizer = AutoTokenizer.from_pretrained(config.model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    config.model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    use_cache=True,
    device_map="cuda",
)

# --- 2. Apply Custom LoRA ---
print("\nApplying Flexible LoRA...")
target_info = [
    {"type": "expert", "expert_index": "38", "layer_index": "1"},
    {"type": "expert", "expert_index": "54", "layer_index": "2"},
    # {"type": "attention", "layer_index": "all"},
]
lora_params = {
    "r": config.lora_r,
    "lora_alpha": config.lora_alpha,
    "lora_dropout": config.lora_dropout,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
lora_wrapper = SuperExpertLora(model=model, target_info=target_info, **lora_params)
peft_model = lora_wrapper.get_model()
peft_model.enable_input_require_grads()

print("Compiling the model... (may take a minute)")
# peft_model = torch.compile(peft_model)

print("\n--- Parameter Count ---")
peft_model.print_trainable_parameters()
print("-----------------------\n")

# --- 3. Load and Prepare Dataset ---
print(f"Loading dataset: {config.dataset_name}")

def load_custom_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)  
    return data

def format_and_tokenize_custom(examples):
    inputs, labels = [], []

    for q, a, e in zip(examples["question"], examples["answer"], examples["explain"]):
        instruction = f"Question:\n{q}\n\nAnswer and explanation:"
        response = f"{a}\n{e}"

        instr_ids = tokenizer.encode(instruction, add_special_tokens=False)
        resp_ids = tokenizer.encode(response, add_special_tokens=False)

        input_ids = instr_ids + resp_ids + [tokenizer.eos_token_id]
        label_ids = [-100] * len(instr_ids) + resp_ids + [tokenizer.eos_token_id]

        if len(input_ids) > config.max_seq_length:
            input_ids = input_ids[-config.max_seq_length:]
            label_ids = label_ids[-config.max_seq_length:]

        inputs.append(input_ids)
        labels.append(label_ids)

    return {"input_ids": inputs, "labels": labels}

raw_data = load_custom_json(config.dataset_name)
dataset = Dataset.from_list(raw_data)

tokenized_dataset = dataset.map(
    format_and_tokenize_custom,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Tokenizing dataset",
    num_proc=4,
)
print(f"Tokenized columns: {tokenized_dataset.column_names}")

# --- 4. Training Args ---
training_args = TrainingArguments(
    output_dir=config.output_dir,
    per_device_train_batch_size=config.batch_size,
    gradient_accumulation_steps=config.grad_accum_steps,
    learning_rate=config.learning_rate,
    num_train_epochs=config.epochs,
    lr_scheduler_type="cosine",
    warmup_steps=config.warmup_steps,
    logging_steps=config.logging_steps,
    save_strategy="steps",
    save_steps=config.save_steps,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    remove_unused_columns=False,
    save_total_limit=2,
)

# --- 5. Trainer ---
print("Initializing Trainer...")
data_collator = CausalDataCollator(tokenizer=tokenizer)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --- 6. Train ---
print("\n🚀 Starting training from scratch... 🚀")
trainer.train()

# --- 7. Save Final Model ---
final_save_path = f"{config.output_dir}/final_checkpoint"
print(f"Saving final LoRA adapter to {final_save_path}...")
trainer.save_model(final_save_path)
