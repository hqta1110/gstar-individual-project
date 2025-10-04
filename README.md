# Super Expert LoRA: Fine-Tuning and Evaluation

This project provides tools to implement and evaluate **Super Expert LoRA** and **Supervised Fine-Tuning (SFT)**. The experiments here were conducted using the **DeepSeek-V2-Lite 16B** model. The GPU required for run this project is 1 A100.

For the original research and usage for profiling Super Experts, please refer to the original project: [Super-Experts-Profilling](https://github.com/ZunhaiSu/Super-Experts-Profilling).

---

## 📂 Directory Structure

* `train/`: Contains the implementation for Super Expert LoRA and the Supervised Fine-Tuning (SFT) script.
* `rep/`: Contains scripts to evaluate model performance across six benchmarks: **MMLU**, **GSM8k**, **ARC-E**, **BoolQ**, **MedQA**, and **VMLU**.
* `utils/`: Contains utils file for pruning SE, comparing model's weight, etc.
---

## 🚀 Workflow

### Step 1: Profile Super Experts (Prerequisite)

Before using this repository, you must first run the profiling process from the original project to identify the Super Experts for your target model.

1. Go to the [Super-Experts-Profilling](https://github.com/ZunhaiSu/Super-Experts-Profilling) repository.
2. Follow their instructions to generate the Super Expert profile for your model.
3. (Optional) After obtaining the Super Experts, you can **prune them out from the base model** using the script below, then evaluate the pruned model:

   ```bash
   python utils/pruning.py
   ```

Alternatively, this repository also provides a built-in way to profile Super Experts across different datasets (math, legal, medical, multilingual, etc.). For example, you can run:

```bash
python run.py \
  --model_name deepseek-ai/DeepSeek-V2-Lite \
  --dataset_name math \
  --save_path destination \
  --profile_super_experts
```

**Arguments**:

* `model_name`: Path to the model you want to profile.
* `dataset_name`: Dataset for profiling (`math`, `medical`, `legal`, `wikitext2`, `c4`, `tinystories`, `multilingual`).
* `save_path`: Directory to save the profiling results.

---

### Step 2: Supervised Fine-Tuning (Optional)

You can fine-tune the model on a custom dataset, which is particularly useful for specializing in multiple-choice question (MCQ) formats.

1. **Prepare the Dataset**: The training dataset must be in JSON format with the following keys:

   * `question`: The input prompt or question.
   * `explain`: The detailed, expected output or reasoning.
   * `answer`: The final letter choice (e.g., `"A"`, `"B"`, `"C"`).

2. **Run Training**:

   ```bash
   python train/run_3.py
   ```

---

### Step 3: Evaluate Model Performance

Evaluate your base model or fine-tuned model on the supported benchmarks. Please change the path to the model & result in the code before running.

1. Navigate to the `rep/` directory.
2. Run the corresponding script:

   * **Base model**: `mmlu.py`
   * **LoRA model**: `mmlu_lora.py`

Example:

```bash
# Base model
python rep/mmlu.py

# Fine-tuned LoRA model
python rep/mmlu_lora.py
```

---

### Step 4: Post-Process Evaluation Results

Evaluation scripts produce raw text outputs. This final step extracts the precise answer choice for automated scoring.

1. This project uses **Qwen-2.5-32B-Instruct** for reliable answer extraction.
2. Use `rep/process.ipynb` to preprocess outputs for each benchmark.
3. Configure `post.py` with the paths to your evaluation output files.
4. Run:

   ```bash
   python post.py
   ```
5. (Optional) My results are stored in result.csv. Actual values may differ depending on the dataset and the experts chosen for fine-tuning/pruning, so consider this file as a reference.
---
