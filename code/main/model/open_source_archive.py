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
        print(f"🔬 XAI 분석 시작: {news_name}")
        
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
        
        print(f"🎯 분석 타겟 토큰: '{target_token}' (ID: {target_token_id})")
        print(f"📊 타겟 토큰 확률 - 원본: {original_probs[target_token_id]:.4f}, 베이스라인: {baseline_probs[target_token_id]:.4f}")
        
        # Saliency 분석
        try:
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
            print("✅ Saliency 계산 완료")
            
        except Exception as e:
            print(f"⚠️ Saliency 계산 중 오류: {e}")
            seq_len = original_inputs['input_ids'].shape[1]
            saliency_per_token = torch.zeros(seq_len)
        
        # 토큰 정보 준비
        prompt_tokens = self.tokenizer.tokenize(original_prompt)
        baseline_tokens = self.tokenizer.tokenize(baseline_prompt)
        
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
        
        # 범위 검증 및 제한
        if news_name_start is not None and news_name_end is not None:
            # 최대 8토큰으로 제한
            if news_name_end - news_name_start > 8:
                news_name_end = news_name_start + 8
            
            # saliency_per_token 길이 내로 제한
            max_idx = len(saliency_per_token) - 1
            news_name_start = min(news_name_start, max_idx)
            news_name_end = min(news_name_end, max_idx)
            
            print(f"🔍 News name 토큰 범위: {news_name_start}-{news_name_end}")
            print(f"🔍 토큰 내용: {prompt_tokens[news_name_start:news_name_end+1]}")
        else:
            print("⚠️ News name 토큰을 찾을 수 없습니다.")
        
        # 영향력 계산
        total_saliency = torch.abs(saliency_per_token).sum().item()
        
        if news_name_start is not None and news_name_end is not None and news_name_start <= news_name_end:
            news_saliency = torch.abs(saliency_per_token[news_name_start:news_name_end+1]).sum().item()
            saliency_ratio = (news_saliency / total_saliency * 100) if total_saliency > 0 else 0
            
            print(f"\n📈 News Name 영향도 분석:")
            print(f"   🔍 Saliency: {saliency_ratio:.2f}% ({news_saliency:.4f} / {total_saliency:.4f})")
            print(f"   📍 토큰 위치: {news_name_start}-{news_name_end}")
        else:
            saliency_ratio = 0
            news_saliency = 0
            print(f"⚠️ News name 영향력을 계산할 수 없습니다.")
        
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
                'probability_change_ratio': abs(prob_change) * 100,  # 확률 변화를 백분율로
                'average_ratio': (saliency_ratio + abs(prob_change) * 100) / 2,
            },
            
            # 전체 중요도 점수
            'total_importance': {
                'total_saliency': total_saliency,
            }
        }
        
        return xai_results, data
    
    def analyze_news_name_impact(self, xai_results, save_path: str = None):
        """
        XAI 분석 결과에서 news_name의 영향을 분석하고 시각화
        
        Args:
            xai_results: process_question_natural_xai의 결과
            save_path: 결과를 저장할 경로 (None이면 저장하지 않음)
        """
        # 1. 기본 정보 출력
        print("\n" + "="*50)
        print("🔬 XAI 분석 결과 요약")
        print("="*50)
        print(f"🎯 타겟 토큰: {xai_results['target_token']}")
        print(f"📊 원본 확률: {xai_results['original_probability']:.4f}")
        print(f"📊 베이스라인 확률: {xai_results['baseline_probability']:.4f}")
        print(f"📈 확률 변화: {xai_results['probability_change']:.4f}")
        
        # 2. news_name 영향 비율 출력 (핵심 정보)
        ratio = xai_results['news_name_influence_ratio']
        print("\n" + "="*50)
        print("📊 News Name이 전체 입력에서 차지하는 영향 비율")
        print("="*50)
        print(f"🔍 Saliency 방법: {ratio['saliency_ratio']:.2f}%")
        print(f"🔍 확률 변화: {ratio['probability_change_ratio']:.2f}%")
        print(f"📊 평균 영향 비율: {ratio['average_ratio']:.2f}%")
        
        # 3. news_name 절대 영향력 출력
        impact = xai_results['news_name_impact']
        print("\n" + "="*30)
        print("📈 News Name 절대 영향력")
        print("="*30)
        print(f"Saliency Sum: {impact['saliency_sum']:.4f}")
        print(f"Saliency Mean: {impact['saliency_mean']:.4f}")
        print(f"확률 변화: {xai_results['probability_change']:.4f}")
        
        # 4. 토큰별 중요도 시각화
        tokens = xai_results['tokens']
        news_start, news_end = xai_results['news_name_token_range']
        
        if news_start is not None and news_end is not None:
            # 토큰과 점수 배열의 길이 맞춤
            saliency_scores = xai_results['saliency_abs_scores']
            
            # 차원 확인 및 조정
            print(f"🔧 시각화 차원 정보: tokens={len(tokens)}, saliency_scores={saliency_scores.shape}")
            
            # 최소 길이로 맞춤
            min_len = min(len(tokens), len(saliency_scores))
            tokens_viz = tokens[:min_len]
            saliency_scores_viz = saliency_scores[:min_len]
            
            # news_name 범위도 조정
            news_start_viz = min(news_start, min_len - 1)
            news_end_viz = min(news_end, min_len - 1)
            
            print(f"🔧 조정된 길이: {min_len}, news_range: {news_start_viz}-{news_end_viz}")
            
            # 전체 토큰에 대한 중요도 플롯
            fig, axes = plt.subplots(2, 1, figsize=(18, 10))
            
            # Saliency scores
            axes[0].bar(range(len(tokens_viz)), saliency_scores_viz, alpha=0.7, color='blue')
            if news_start_viz <= news_end_viz and news_end_viz < len(saliency_scores_viz):
                axes[0].bar(range(news_start_viz, news_end_viz+1), saliency_scores_viz[news_start_viz:news_end_viz+1], 
                           alpha=0.9, color='red', label=f'News Name Tokens ({ratio["saliency_ratio"]:.1f}%)')
            axes[0].set_title('Saliency Scores by Token')
            axes[0].set_ylabel('Absolute Saliency Score')
            axes[0].legend()
            
            # 영향 비율 비교 차트
            methods = ['Saliency', 'Probability Change']
            ratios_vals = [ratio['saliency_ratio'], ratio['probability_change_ratio']]
            colors_ratio = ['blue', 'green']
            
            bars = axes[1].bar(methods, ratios_vals, color=colors_ratio, alpha=0.7)
            axes[1].set_title(f'News Name Influence Analysis (Average: {ratio["average_ratio"]:.1f}%)')
            axes[1].set_ylabel('Influence (%)')
            axes[1].set_ylim(0, max(max(ratios_vals), 1.0) * 1.2)
            
            # 막대 위에 수치 표시
            for bar, ratio_val in zip(bars, ratios_vals):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{ratio_val:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # x축 레이블 설정 (첫 번째 플롯만)
            axes[0].set_xticks(range(len(tokens_viz)))
            axes[0].set_xticklabels(tokens_viz, rotation=45, ha='right', fontsize=8)
            axes[0].set_xlabel('Token Index')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}_token_importance_with_ratio.png", dpi=300, bbox_inches='tight')
            
            plt.show()
        
        # 5. 종합 영향력 분석
        influence_score = self._calculate_simple_influence_score(xai_results)
        
        print("\n" + "="*40)
        print("🏆 종합 영향력 분석")
        print("="*40)
        print(f"📊 종합 영향력 점수: {influence_score:.4f}")
        print(f"📊 평균 영향 비율: {ratio['average_ratio']:.2f}%")
        
        # 영향도 해석
        if ratio['average_ratio'] > 15:
            interpretation = "🔴 매우 높은 영향"
        elif ratio['average_ratio'] > 8:
            interpretation = "🟡 높은 영향"
        elif ratio['average_ratio'] > 3:
            interpretation = "🟢 중간 영향"
        else:
            interpretation = "⚪ 낮은 영향"
        
        print(f"📝 영향도 해석: {interpretation}")
        print("="*40)
        
        return influence_score, ratio['average_ratio']
    
    def _calculate_simple_influence_score(self, xai_results):
        """
        간소화된 영향력 점수 계산 (Saliency 기반)
        """
        impact = xai_results['news_name_impact']
        prob_change = abs(xai_results['probability_change'])
        
        # Saliency 기반 점수 (정규화)
        saliency_score = impact['saliency_mean']
        max_saliency = max(xai_results['saliency_abs_scores']) if len(xai_results['saliency_abs_scores']) > 0 else 1e-8
        normalized_saliency = saliency_score / max_saliency if max_saliency > 0 else 0
        
        # 확률 변화 점수 (정규화)
        normalized_prob = min(prob_change * 10, 1.0)  # 확률 변화를 0-1로 정규화
        
        # 가중 평균으로 종합 점수 계산
        weights = {'saliency': 0.7, 'prob_change': 0.3}
        total_score = (
            weights['saliency'] * normalized_saliency +
            weights['prob_change'] * normalized_prob
        )
        
        return total_score
    
    def _calculate_influence_score(self, xai_results):
        """
        여러 XAI 방법의 결과를 종합하여 news_name의 영향력 점수 계산
        """
        impact = xai_results['news_name_impact']
        
        # 각 방법별 정규화된 점수 계산
        saliency_score = impact['saliency_mean']
        shap_score = impact['shap_mean']

        # 확률 변화도 고려
        prob_change = abs(xai_results['probability_change'])
        
        original_scores = {
            'saliency': saliency_score,
            'shap': shap_score,
            'prob_change': prob_change
        }
        
        # 가중 평균으로 종합 점수 계산
        # 각 방법의 중요도를 고려한 가중치 설정
        weights = {
            'saliency': 0.3,
            'shap': 0.3,
            'prob_change': 0.1
        }
        
        # 점수 정규화 (0-1 범위로)
        max_saliency = max(xai_results['saliency_abs_scores']) if len(xai_results['saliency_abs_scores']) > 0 else 1e-8
        max_shap = max(np.abs(xai_results['shap_values'])) if len(xai_results['shap_values']) > 0 else 1e-8
        
        normalized_saliency = saliency_score / max_saliency if max_saliency > 0 else 0
        normalized_shap = shap_score / max_shap if max_shap > 0 else 0
        normalized_prob = min(prob_change * 10, 1.0)  # 확률 변화를 0-1로 정규화
        
        # 종합 점수 계산
        total_score = (
            weights['saliency'] * normalized_saliency +
            weights['shap'] * normalized_shap +
            weights['prob_change'] * normalized_prob
        )
        
        return total_score, original_scores


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