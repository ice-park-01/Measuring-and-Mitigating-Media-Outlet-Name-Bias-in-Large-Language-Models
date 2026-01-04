import re

def extract_answer_from_logprobs(model_name: str, logprobs_dict: dict) -> str:
    if "llama" in model_name.lower():        
        # 가장 높은 확률의 항목부터 살펴봄
        for key, _ in logprobs_dict.items():
            # 'Ġ' 제거
            cleaned_key = key.replace('Ġ', '')
            # A, B, C 중 하나의 값이 나오면 반환
            if cleaned_key in ['A', 'B', 'C']:
                return cleaned_key
            
    elif "qwen" in model_name.lower():
        # 가장 높은 확률의 항목부터 살펴봄
        if "qwen-" in model_name.lower():
            for key, _ in logprobs_dict.items():
                # 'Ġ' 제거
                cleaned_key = key.decode('utf-8')
                cleaned_key = cleaned_key.strip()
                # A, B, C 중 하나의 값이 나오면 반환
                if cleaned_key in ['A', 'B', 'C']:
                    return cleaned_key
        else:
            for key, _ in logprobs_dict.items():
                # 'Ġ' 제거
                cleaned_key = key.replace('Ġ', '')
                # A, B, C 중 하나의 값이 나오면 반환
                if cleaned_key in ['A', 'B', 'C']:
                    return cleaned_key
            
    # 여기 아래로 검증 필요
    elif "phi" in model_name.lower():
        # 가장 높은 확률의 항목부터 살펴봄
        for key, _ in logprobs_dict.items():
            # 'Ġ' 제거
            cleaned_key = key.replace('Ġ', '')
            # A, B, C 중 하나의 값이 나오면 반환
            if cleaned_key in ['A', 'B', 'C']:
                return cleaned_key
            
    elif "mistral" in model_name.lower():
        # 가장 높은 확률의 항목부터 살펴봄
        for key, _ in logprobs_dict.items():
            # 'Ġ' 제거
            cleaned_key = key.replace('Ġ', '')
            # A, B, C 중 하나의 값이 나오면 반환
            if cleaned_key in ['A', 'B', 'C']:
                return cleaned_key
            
    elif "gemma" in model_name.lower():
        # 가장 높은 확률의 항목부터 살펴봄
        for key, _ in logprobs_dict.items():
            # 'Ġ' 제거
            cleaned_key = key.replace('Ġ', '')
            # A, B, C 중 하나의 값이 나오면 반환
            if cleaned_key in ['A', 'B', 'C']:
                return cleaned_key
            
    else:
        raise ValueError(f"Unsupported model: {model_name}")