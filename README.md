# Super Expert LoRA: Fine-Tuning and Evaluation

This project provides tools to implement and evaluate **Super Expert LoRA** and **Supervised Fine-Tuning (SFT)**. The experiments here were conducted using the **DeepSeek-V2-Lite 16B** model.

For the original research and usage for profiling Super Experts, please refer to the original project: [Super-Experts-Profilling](https://github.com/ZunhaiSu/Super-Experts-Profilling).

-----

## 📂 Directory Structure

  * `train/`: Contains the implementation for Super Expert LoRA and the Supervised Fine-Tuning (SFT) script.
  * `rep/`: Contains scripts to evaluate model performance across six benchmarks: **MMLU**, **GSM8k**, **ARC-E**, **BoolQ**, **MedQA**, and **VMLU**.

-----

## 🚀 Workflow

### Step 1: Profile Super Experts (Prerequisite)

Before using this repository, you must first run the profiling process from the original project to identify the Super Experts for your target model.

1.  Go to the [Super-Experts-Profilling](https://github.com/ZunhaiSu/Super-Experts-Profilling) repository.
2.  Follow their instructions to generate the Super Expert profile for your model.

### Step 2: Supervised Fine-Tuning (Optional)

You can fine-tune the model on a custom dataset, which is particularly useful for specializing in multiple-choice question (MCQ) formats.

1.  **Prepare the Dataset**: The training dataset must be in a JSON format where each entry is an object with the following keys:

      * `question`: The input prompt or question.
      * `explain`: The detailed, expected output or reasoning.
      * `answer`: The final letter choice for an MCQ (e.g., "A", "B", "C").

2.  **Run Training**: Execute the `run_3.py` script to start the SFT process.

    ```bash
    python train/run_3.py
    ```

    Feel free to modify the training pipeline to suit your needs.

### Step 3: Evaluate Model Performance

Evaluate your base model or the fine-tuned model on the supported benchmarks.

1.  Navigate to the `rep/` directory.

2.  Run the corresponding script for the benchmark you wish to test.

      * To evaluate the **base model**, use the standard script (e.g., `mmlu.py`).
      * To evaluate a **fine-tuned LoRA model** from Step 2, use the `_lora` version of the script (e.g., `mmlu_lora.py`).

    **Example for MMLU:**

    ```bash
    # Evaluate base model
    python rep/mmlu.py

    # Evaluate fine-tuned LoRA model
    python rep/mmlu_lora.py
    ```

### Step 4: Post-Process Evaluation Results

The evaluation scripts produce raw text outputs. This final step extracts the precise answer choice (e.g., "A") from the raw text to allow for automated scoring.

1.  This project uses **Qwen-2.5-32B-Instruct** for reliable answer extraction.
2.  Refer to `rep/process.ipynb` with corresponding benchmark to post-process the raw text outputs before running the extraction.
3.  Configure the `post.py` script with the paths to your evaluation output files.
4.  Run the script to process the results.
    ```bash
    python post.py
    ```