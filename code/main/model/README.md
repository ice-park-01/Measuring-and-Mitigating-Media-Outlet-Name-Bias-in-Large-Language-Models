# Model Inference & XAI

This directory contains the core scripts for running model inference and performing Explainable AI (XAI) analyses.

## Files Description

- **`inference.py`**: The primary script for running model inference. It loads a model, processes the input prompts (with varying media biases), and saves the generated outputs.
- **`inference_xai.py`**: Specialized inference script that integrates with XAI libraries (like SHAP or Captum) to calculate feature attribution scores, helping to understand which parts of the input caused the bias.
- **`check_attention_mask.py`**: Utility to verify and debug attention masks, ensuring that padding and masking are handled correctly during batch processing.
- **`prompt_optimization.py`**: Scripts related to prompt engineering and optimization techniques used to potentially mitigate or study the bias.
- **`result_*.csv`**: Intermediate raw output files from inference runs are often saved here before being moved or analyzed.

## Running Inference

You typically won't run `inference.py` directly but rather through the shell scripts in the `shell/` directory, which handle argument passing for different experimental configurations.
