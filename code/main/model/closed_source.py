import os
import sys

# 현재 파일의 디렉토리 경로를 기준으로 custom_dataset 폴더의 경로를 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
        
from custom_dataset.dataset import CustomDatasetAllsides
from src.tokens import openai_token

from openai import OpenAI


class ClosedSourceModel:
    def __init__(self, model_name: str):
        self.client = OpenAI(api_key=openai_token)
        self.model = model_name
        self.max_completion_tokens = 1000
        self.temperature = 0.0
        self.logprobs = True

    def process_question_natural(self, data, news_name: str):
        prompt_text = data.get_natural_prompt(news_name)
        
        outputs = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            logprobs=self.logprobs,
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ]
        )
            
        model_output = outputs.choices[0].message.content
        model_output = model_output.strip()
        
        return model_output, data
    
    def process_question_natural_for_order(self, data, news_name: str):
        prompt_text, order = data.get_natural_prompt_for_order(news_name)
        
        outputs = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            logprobs=self.logprobs,
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ]
        )
            
        model_output = outputs.choices[0].message.content
        model_output = model_output.strip()
        
        return model_output, data, order
    
    def process_question_cot(self, data, news_name: str):
        # 수정 예정
        prompt_text = data.get_cot_prompt(news_name)
                    
        inputs = self.tokenizer(prompt_text, return_tensors="pt", return_token_type_ids=False)
        inputs_token_len = len(inputs['input_ids'][0])
        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device)
        cot_outputs = self.model.generate(**inputs, max_new_tokens=inputs_token_len+200)
        cot_rationale = self.tokenizer.batch_decode(cot_outputs)
        
        prompt_text_with_rationale = cot_rationale[0] + ". Therefore, the answer is"     
        second_inputs = self.tokenizer(prompt_text_with_rationale, return_tensors="pt", return_token_type_ids=False)
        for k, v in second_inputs.items():
            second_inputs[k] = v.to(self.model.device)
        outputs = self.model(**second_inputs)
        outputs.logits = outputs.logits.to('cpu')
        logits = outputs.logits[0, -1]
        probs = logits.float().softmax(dim=-1)
            
        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }                
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:50]
        }

        # GPU 메모리 해제
        # del inputs, cot_outputs, second_inputs, outputs, logits, probs
        # torch.cuda.empty_cache()

        return logprobs_dict, data
    

    def process_question_summarization(self, data, reference_class: str, summary_length: int):
        prompt_text = data.get_summarization_prompt_reference(reference_class, summary_length)
        
        outputs = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ]
        )
        
        model_output = outputs.choices[0].message.content
        model_output = model_output.strip()
        
        return model_output, data


    
if __name__ == "__main__":
    hf_token = "Your HF Token"
    os.environ["HF_TOKEN"] = hf_token
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = OpenSourceModel("meta-llama/Llama-3.1-8B-Instruct", "/nas/.cache/huggingface/")
    dataset = CustomDatasetAllsides("../data/allsides/Article-Bias-Prediction/data/jsons", "meta-llama/Llama-3.1-8B-Instruct", "itself")
    print(model.predict(dataset[0]))