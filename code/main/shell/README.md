# Experiment Shell Scripts

This directory contains bash scripts to automate the execution of experiments. They wrap the `model/inference.py` calls with specific parameters for different models and datasets.

## Naming Convention

- **`run_allsides_*.sh`**: Runs experiments using the AllSides media bias dataset. The suffix indicates the model family (e.g., `gemma2`, `llama3`, `gpt4`).
- **`run_hyper_*.sh`**: Runs experiments using the Hyperpartisan dataset.
- **`summarize_*.sh`**: Runs summarization tasks specifically.
- **`rvt_*.sh`**: Scripts related to the Relevance Validation Test (RVT).

## How to Run

To execute an experiment, simply run the script from the project root:

```bash
# Example: Run AllSides experiment with Llama 3
bash shell/run_allsides_llama3.1.sh
```

**Note**: Ensure you have set up your environment variables (e.g., API keys for closed-source models) before running these scripts.
