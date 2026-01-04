import sys
import os
sys.path.append(os.path.abspath('.'))

from model.open_source import OpenSourceModel
from custom_dataset.dataset import CustomDatasetAllsides

def example_xai_analysis():
    """
    XAI 분석 기능 사용 예시
    """
    # 모델 초기화 (실제 모델 경로로 변경 필요)
    model_name = "microsoft/DialoGPT-medium"  # 예시 모델
    cache_dir = "./model_cache"
    
    try:
        model = OpenSourceModel(model_name, cache_dir, quantization=False)
        print("모델 로드 완료")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("실제 사용 가능한 모델 경로로 변경해주세요.")
        return
    
    # 샘플 데이터 생성
    sample_data = CustomDatasetAllsides(
        news_name="CNN",
        title="Sample News Title",
        date="2024-01-01",
        content="This is a sample news content for testing XAI analysis. The content discusses various political topics and provides different perspectives on current events."
    )
    
    # 1. 기본 XAI 분석
    print("=== 기본 XAI 분석 시작 ===")
    xai_results, data = model.process_question_natural_xai(sample_data, "CNN")
    
    # 2. 결과 분석 및 시각화
    print("=== 결과 분석 및 시각화 ===")
    influence_score = model.analyze_news_name_impact(xai_results, save_path="./xai_results")
    
    # 3. 다른 뉴스 소스와 비교
    print("\n=== 다른 뉴스 소스와 비교 ===")
    news_sources = ["CNN", "Fox News", "BBC", "none"]
    comparison_results = {}
    
    for news_source in news_sources:
        print(f"\n분석 중: {news_source}")
        try:
            xai_result, _ = model.process_question_natural_xai(sample_data, news_source)
            influence_score = model._calculate_influence_score(xai_result)
            comparison_results[news_source] = {
                'influence_score': influence_score,
                'probability_change': xai_result['probability_change'],
                'target_token': xai_result['target_token']
            }
            print(f"영향력 점수: {influence_score:.4f}")
            print(f"확률 변화: {xai_result['probability_change']:.4f}")
        except Exception as e:
            print(f"분석 실패: {e}")
    
    # 4. 비교 결과 요약
    print("\n=== 비교 결과 요약 ===")
    for news_source, result in comparison_results.items():
        print(f"{news_source}:")
        print(f"  - 영향력 점수: {result['influence_score']:.4f}")
        print(f"  - 확률 변화: {result['probability_change']:.4f}")
        print(f"  - 타겟 토큰: {result['target_token']}")
    
    # 5. 특정 토큰에 대한 분석
    print("\n=== 특정 토큰 분석 ===")
    target_tokens = ["A", "B", "C"]  # 정치적 편향 카테고리
    
    for token in target_tokens:
        try:
            xai_result, _ = model.process_question_natural_xai(sample_data, "CNN", target_token=token)
            influence_score = model._calculate_influence_score(xai_result)
            print(f"토큰 '{token}' 분석:")
            print(f"  - 영향력 점수: {influence_score:.4f}")
            print(f"  - 원본 확률: {xai_result['original_probability']:.4f}")
            print(f"  - 베이스라인 확률: {xai_result['baseline_probability']:.4f}")
        except Exception as e:
            print(f"토큰 '{token}' 분석 실패: {e}")

def batch_xai_analysis():
    """
    여러 데이터에 대한 배치 XAI 분석
    """
    # 모델 초기화
    model_name = "microsoft/DialoGPT-medium"  # 예시 모델
    cache_dir = "./model_cache"
    
    try:
        model = OpenSourceModel(model_name, cache_dir, quantization=False)
        print("모델 로드 완료")
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        return
    
    # 여러 샘플 데이터 생성
    sample_data_list = [
        CustomDatasetAllsides("CNN", "Title 1", "2024-01-01", "Content about liberal policies and social reforms."),
        CustomDatasetAllsides("Fox News", "Title 2", "2024-01-01", "Content about conservative values and traditional policies."),
        CustomDatasetAllsides("BBC", "Title 3", "2024-01-01", "Content about balanced reporting and neutral perspectives."),
    ]
    
    news_sources = ["CNN", "Fox News", "BBC", "none"]
    batch_results = {}
    
    for i, data in enumerate(sample_data_list):
        print(f"\n=== 데이터 {i+1} 분석 ===")
        batch_results[f"data_{i+1}"] = {}
        
        for news_source in news_sources:
            try:
                xai_result, _ = model.process_question_natural_xai(data, news_source)
                influence_score = model._calculate_influence_score(xai_result)
                
                batch_results[f"data_{i+1}"][news_source] = {
                    'influence_score': influence_score,
                    'probability_change': xai_result['probability_change'],
                    'target_token': xai_result['target_token']
                }
                
                print(f"{news_source}: 영향력 점수 {influence_score:.4f}, 확률 변화 {xai_result['probability_change']:.4f}")
                
            except Exception as e:
                print(f"{news_source} 분석 실패: {e}")
    
    # 배치 결과 요약
    print("\n=== 배치 분석 결과 요약 ===")
    for data_key, results in batch_results.items():
        print(f"\n{data_key}:")
        for news_source, result in results.items():
            print(f"  {news_source}: {result['influence_score']:.4f}")

if __name__ == "__main__":
    print("XAI 분석 예시 시작")
    
    # 기본 분석
    example_xai_analysis()
    
    # 배치 분석 (선택적)
    # batch_xai_analysis() 