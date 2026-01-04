import os
import sys
import json
# 현재 파일의 디렉토리 경로를 기준으로 custom_dataset 폴더의 경로를 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np

# 시각화를 위한 matplotlib 사용
import matplotlib.pyplot as plt
import seaborn as sns

# XAI를 위한 라이브러리들
import shap
from captum.attr import Saliency, IntegratedGradients, LayerIntegratedGradients
from captum.attr import visualization as viz
        
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from custom_dataset.dataset import CustomDatasetAllsides

import numpy as np
import matplotlib.pyplot as plt
from src.functions import extract_answer_from_logprobs


  
class OpenSourceModel:
    def __init__(self, model_name: str, cache_dir: str, quantization: bool = True):
        self.model_name = model_name.lower()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                       cache_dir=cache_dir,
                                                       device_map="auto",
                                                       trust_remote_code=True
                                                       )
        
        # pad_token_id 설정
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # 공통 모델 매개변수
        model_params = {
            "cache_dir": cache_dir,
            "device_map": "auto",
            "trust_remote_code": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "attn_implementation": "eager"
        }
        
        if quantization:
            if "gemma" in model_name:
                quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) # , 
                model_params.update({
                    "quantization_config": quant_config,
                    "torch_dtype": torch.bfloat16
                })
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_params)
            elif "llama" in model_name:
                quant_config = BitsAndBytesConfig(load_in_4bit=True)
                model_params.update({
                    "quantization_config": quant_config,
                    "torch_dtype": torch.bfloat16
                })
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_params)
            else:
                quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                model_params.update({
                    "quantization_config": quant_config
                })
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_params)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_params)
            
        self.lbls_map = {v: k for k, v in self.tokenizer.get_vocab().items()}
        

    def process_question_natural(self, data, news_name: str):
        prompt_text = data.get_natural_prompt(news_name)
        
        inputs = self.tokenizer.encode_plus(prompt_text, return_tensors="pt", return_token_type_ids=False)
        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device)
            
        outputs = self.model(**inputs)
        # print(f"outputs.logits: {outputs.logits}")
        outputs.logits = outputs.logits.to('cpu')
        logits = outputs.logits[0, -1]
        
        probs = logits.float().softmax(dim=-1)

        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }
                
        # for i, (k, v) in enumerate(logprobs_dict.items()):
        #     if i >= 5:
        #         break
        #     print(k, v)
        
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:150]
        }
        
        # GPU 메모리 해제
        # del inputs, outputs, logits, probs
        # torch.cuda.empty_cache()
        
        return logprobs_dict, data
    
    def process_question_natural_for_order(self, data, news_name: str):
        prompt_text, order = data.get_natural_prompt_for_order(news_name)
        
        # print(f"prompt_text: \n\n{prompt_text[:500]}")
        
        inputs = self.tokenizer.encode_plus(prompt_text, return_tensors="pt", return_token_type_ids=False)
        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device)
            
        outputs = self.model(**inputs)
        # print(f"outputs.logits: {outputs.logits}")
        outputs.logits = outputs.logits.to('cpu')
        logits = outputs.logits[0, -1]
        
        probs = logits.float().softmax(dim=-1)

        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }
                
        # for i, (k, v) in enumerate(logprobs_dict.items()):
        #     if i >= 5:
        #         break
        #     print(k, v)
        
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:150]
        }
        
        # GPU 메모리 해제
        # del inputs, outputs, logits, probs
        # torch.cuda.empty_cache()
        
        return logprobs_dict, data, order
    
    def process_question_natural_optimization(self, data, prompt: str, news_name: str):
        prompt_text = data.get_natural_prompt_optimization(prompt, news_name)
        
        inputs = self.tokenizer.encode_plus(prompt_text, return_tensors="pt", return_token_type_ids=False)
        for k, v in inputs.items():
            inputs[k] = v.to(self.model.device)
            
        outputs = self.model(**inputs)
        # print(f"outputs.logits: {outputs.logits}")
        outputs.logits = outputs.logits.to('cpu')
        logits = outputs.logits[0, -1]
        
        probs = logits.float().softmax(dim=-1)

        logprobs_dict = {
            self.lbls_map[i]:
            np.log(probs[i].item()) for i in range(len(self.lbls_map))
        }
                
        # for i, (k, v) in enumerate(logprobs_dict.items()):
        #     if i >= 5:
        #         break
        #     print(k, v)
        
        # Reduce logprobs_dict to only keys with top 50 largest values
        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:150]
        }
        
        # GPU 메모리 해제
        # del inputs, outputs, logits, probs
        # torch.cuda.empty_cache()
        
        return logprobs_dict, data
    
    def process_question_summarization(self, data, news_name: str, summary_length: int):
        summary_length_conversion = {
            3: "three",
            5: "five",
            10: "ten"
        }
        prompt_text = data.get_summarization_prompt(news_name, summary_length_conversion[summary_length])
        
        if "qwen" in self.model_name:
            messages = [
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Follow the user's instructions carefully. Do not include any other text except the answer."},
                {"role": "user", "content": prompt_text}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        elif "phi" in self.model_name:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Follow the user's instructions carefully. Do not include any other text except the answer."},
                {"role": "user", "content": prompt_text}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        elif "gemma" in self.model_name:
            messages = [
                {"role": "user", "content": "You are a helpful assistant. Follow the user's instructions carefully. Do not include any other text except the answer.\n\n" + prompt_text}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        elif "llama" in self.model_name:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Follow the user's instructions carefully. Do not include any other text except the answer."},
                {"role": "user", "content": prompt_text}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # text = prompt_text + "Only output the summary.\nSummary:"
            text = prompt_text
            
        # print(f"Model input: \n{text}")
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **model_inputs, 
            max_new_tokens=512, 
            temperature=0.3,
            # repetition_penalty=1.1,
            do_sample=True,
            # pad_token_id=self.tokenizer.pad_token_id
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, outputs)
        ]
        response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(response)
        print("--------------------------------")
        
        return response, data
    
    
    def process_question_natural_xai(self, data, news_name: str, target_token: str = None):
        """
        Saliency를 사용하여 news_name이 모델 추론에 미치는 영향을 정량화
        (간소화된 안정적인 버전)
        
        Args:
            data: CustomDatasetAllsides 객체
            news_name: 뉴스 소스 이름
            target_token: 분석할 타겟 토큰 (None이면 가장 높은 확률의 토큰 사용)
        
        Returns:
            dict: XAI 분석 결과
        """
        # print(f"🔬 XAI 분석 시작: {news_name}")
        
        # 원본 프롬프트와 베이스라인 프롬프트 생성
        original_prompt = data.get_natural_prompt(news_name)
        baseline_prompt = data.get_natural_prompt("none")
        
        # 토큰화
        original_inputs = self.tokenizer.encode_plus(original_prompt, return_tensors="pt", return_token_type_ids=False)
        baseline_inputs = self.tokenizer.encode_plus(baseline_prompt, return_tensors="pt", return_token_type_ids=False)
        
        # 디바이스로 이동
        for k, v in original_inputs.items():
            original_inputs[k] = v.to(self.model.device)
        for k, v in baseline_inputs.items():
            baseline_inputs[k] = v.to(self.model.device)
        
        # 원본 모델 추론으로 확률 변화 계산
        with torch.no_grad():
            original_outputs = self.model(**original_inputs)
            original_logits = original_outputs.logits[0, -1].to('cpu')
            original_probs = original_logits.float().softmax(dim=-1)
            
            baseline_outputs = self.model(**baseline_inputs)
            baseline_logits = baseline_outputs.logits[0, -1].to('cpu')
            baseline_probs = baseline_logits.float().softmax(dim=-1)
        
        # 타겟 토큰 결정
        if target_token is None:
            target_token_id = torch.argmax(original_probs).item()
            target_token = self.lbls_map[target_token_id]
        else:
            target_token_id = self.tokenizer.convert_tokens_to_ids(target_token)
            

        logprobs_dict = {
            self.lbls_map[i]:
            np.log(original_probs[i].item()) for i in range(len(self.lbls_map))
        }

        logprobs_dict = {
            k: v for k, v in sorted(
                logprobs_dict.items(),
                key=lambda item: item[1],
                reverse=True
            )[:150]
        }
                
        target_token = extract_answer_from_logprobs(self.model_name, logprobs_dict)
        target_token = "Ġ" + target_token
        target_token_id = self.tokenizer.convert_tokens_to_ids(target_token)
        
        # print(f"🎯 분석 타겟 토큰: '{target_token}' (ID: {target_token_id})")
        # print(f"📊 타겟 토큰 확률 - 원본: {original_probs[target_token_id]:.4f}, 베이스라인: {baseline_probs[target_token_id]:.4f}")
        
        # Saliency 분석
        # 임베딩 준비
        original_embeds = self.model.get_input_embeddings()(original_inputs['input_ids'])
        original_embeds.requires_grad_(True)
        
        # Forward function
        def model_forward_saliency(input_embeds):
            outputs = self.model(inputs_embeds=input_embeds)
            return outputs.logits[:, -1, :]
        
        # Saliency 계산
        saliency = Saliency(model_forward_saliency)
        saliency_attr = saliency.attribute(
            original_embeds,
            target=target_token_id,
            abs=False
        )
        
        # 토큰 단위로 saliency 점수 요약 (L2 norm 사용)
        saliency_per_token = torch.norm(saliency_attr[0], dim=-1)  # [seq_len]
        # print("✅ Saliency 계산 완료")
        
        # 토큰 정보 준비
        prompt_tokens = self.tokenizer.tokenize(original_prompt)
        baseline_tokens = self.tokenizer.tokenize(baseline_prompt)
        news_name_start = None
        news_name_end = None
        
        # news_name 부분의 토큰 인덱스 찾기
        for i, (orig_token, base_token) in enumerate(zip(prompt_tokens, baseline_tokens)):
            if orig_token != base_token:
                if news_name_start is None:
                    news_name_start = i
                for j, token in enumerate(prompt_tokens[i:]):
                    if token == base_token:
                        news_name_end = i + j
                        break
                break
        
        # 영향력 계산
        total_saliency = torch.abs(saliency_per_token).sum().item()
        news_saliency = torch.abs(saliency_per_token[news_name_start:news_name_end+1]).sum().item()
        saliency_ratio = (news_saliency / total_saliency * 100) if total_saliency > 0 else 0
        baseline_saliency_ratio = (news_name_end-news_name_start+1)/len(prompt_tokens) * 100
        
        # print(f"\n📈 News Name 영향도 분석:")
        # print(f"   🔍 Saliency: {saliency_ratio:.2f}% ({news_saliency:.4f} / {total_saliency:.4f})")
        # print(f"   🔍 대조군 Saliency: {baseline_saliency_ratio:.2f}")
        # print(f"   📍 토큰 위치: {news_name_start}-{news_name_end}")
        # print(f"   📍 토큰 내용: {prompt_tokens[news_name_start:news_name_end+1]}")
        # print(f"   📍 전체 토큰 길이: {len(prompt_tokens)}")

        # 확률 변화량
        prob_change = original_probs[target_token_id].item() - baseline_probs[target_token_id].item()
        
        # 결과 정리
        xai_results = {
            'target_token': target_token,
            'target_token_id': target_token_id,
            'original_probability': original_probs[target_token_id].item(),
            'baseline_probability': baseline_probs[target_token_id].item(),
            'probability_change': prob_change,
            
            # Saliency scores
            'saliency_scores': saliency_per_token.cpu().numpy(),
            'saliency_abs_scores': torch.abs(saliency_per_token).cpu().numpy(),
            
            # news_name 토큰 위치
            'news_name_token_range': (news_name_start, news_name_end),
            
            # 토큰 정보
            'tokens': prompt_tokens,
            'token_ids': original_inputs['input_ids'][0].cpu().numpy(),
            
            # news_name 영향력 요약
            'news_name_impact': {
                'saliency_sum': float(news_saliency),
                'saliency_mean': float(news_saliency / (news_name_end - news_name_start + 1)) if news_name_start is not None and news_name_end is not None and news_name_start <= news_name_end else 0,
            },
            
            # 영향 비율
            'news_name_influence_ratio': {
                'saliency_ratio': saliency_ratio,
                'baseline_saliency_ratio': baseline_saliency_ratio,
                'probability_change_ratio': abs(prob_change) * 100,  # 확률 변화를 백분율로
            },
            
            # 전체 중요도 점수
            'total_importance': {
                'total_saliency': total_saliency,
            }
        }
        
        return xai_results, data


class OpenSourceReasoningModel:
    def __init__(self, model_name: str, cache_dir: str, quantization: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                                       cache_dir=cache_dir,
                                                       device_map="auto",
                                                       trust_remote_code=True
                                                       )
        
        # pad_token_id 설정
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        # 공통 모델 매개변수
        model_params = {
            "cache_dir": cache_dir,
            "device_map": "auto",
            "trust_remote_code": True,
            "pad_token_id": self.tokenizer.pad_token_id  # 여기에 pad_token_id 추가
        }
        
        if quantization:
            if "gemma" in model_name:
                quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                model_params.update({
                    "quantization_config": quant_config,
                    "torch_dtype": torch.bfloat16
                })
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_params)
            else:
                quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
                model_params.update({
                    "quantization_config": quant_config
                })
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_params)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_params)
            
        self.lbls_map = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.generation_config = GenerationConfig(
            max_new_tokens=32768,
            temperature=0.6,
            top_p=0.95,
            min_p=0.0,
            top_k=30,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id  # generation_config에도 pad_token_id 추가
        )
        

    def process_question_natural(self, data, news_name: str):
        prompt_text = data.get_reasoning_prompt(news_name)
        
        messages = [
            {"role": "user", "content": prompt_text}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, generation_config=self.generation_config)

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        extracted_answer = response.split("</think>")[1].strip()
        extracted_answer = json.loads(extracted_answer)["answer"]
        
        return extracted_answer, data


    
if __name__ == "__main__":
    hf_token = "Your HF Token"
    os.environ["HF_TOKEN"] = hf_token
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = OpenSourceModel("meta-llama/Llama-3.1-8B-Instruct", "/nas/.cache/huggingface/")
    dataset = CustomDatasetAllsides("../data/allsides/Article-Bias-Prediction/data/jsons", "meta-llama/Llama-3.1-8B-Instruct", "itself")
    print(model.predict(dataset[0]))