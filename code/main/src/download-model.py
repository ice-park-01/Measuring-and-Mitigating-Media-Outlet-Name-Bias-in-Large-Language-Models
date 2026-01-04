import sys
print(sys.executable)

import torch
from transformers import AutoModel, AutoTokenizer


def main():
    # * meta-llama/Llama-3.3-70B-Instruct
    # * Qwen/Qwen2.5-72B-Instruct
    # * microsoft/phi-4
    #     * 14B
    # * mistralai/Mistral-Small-24B-Instruct-2501
    # * google/gemma-2-27b-it
    
    # set cache directory, hf access token
    cache_dir = "/nas/.cache"
    hf_token = "Your Huggingface Token"
    
    
    model_name = "meta-llama/Llama-3.3-70B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)

    print(f"Model {model_name} loaded")

    model_name = "Qwen/Qwen2.5-72B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)

    print(f"Model {model_name} loaded")
    
    model_name = "microsoft/phi-4"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)

    print(f"Model {model_name} loaded")
    
    model_name = "mistralai/Mistral-Small-24B-Instruct-2501"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)

    print(f"Model {model_name} loaded")
    
    model_name = "google/gemma-2-27b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, token=hf_token)

    print(f"Model {model_name} loaded")
    
    
if __name__ == "__main__":
    main()