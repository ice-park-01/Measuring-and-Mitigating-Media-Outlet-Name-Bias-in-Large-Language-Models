# -*- coding: utf-8 -*-
import os, json, datetime, itertools
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# 0. 데이터 로드 및 초기화
# -------------------------------------------------
# 모델 및 아티클 메타데이터 초기화
all_models = {}
article_meta = {}

# 데이터 로드 함수
def load_model_data(model_path, model_name):
    """모델 예측 결과 파일 로드"""
    with open(model_path, 'r') as f:
        model_data = json.load(f)
    all_models[model_name] = model_data
    
def load_article_metadata(metadata_path):
    """아티클 메타데이터 로드"""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    global article_meta
    article_meta = metadata

# 샘플 데이터 생성 (실제 데이터가 없는 경우 테스트용)
def create_sample_data():
    # 샘플 아티클 ID 목록
    article_ids = [f"art_{i}" for i in range(10)]
    
    # 아티클 메타데이터 샘플 생성
    for art_id in article_ids:
        # 무작위로 레이블 할당
        label = np.random.choice(['left', 'lean_left', 'center', 'lean_right', 'right'])
        article_meta[art_id] = {
            'label': label,
            'topic': np.random.choice(['정치', '경제', '사회', '국제']),
            'source': np.random.choice(['A신문', 'B방송', 'C매체', 'D포털']),
            'bias_text': np.random.choice(['hp', 'non-hp'])
        }
    
    # 모델 샘플 데이터 생성
    model_names = ['model_A', 'model_B', 'model_C']
    for model in model_names:
        model_data = {}
        for art_id in article_ids:
            # 기본 예측과 미디어 바이어스에 따른 예측 생성
            base_pred = np.random.choice(['left', 'lean_left', 'center', 'lean_right', 'right'])
            model_data[art_id] = {
                'baseline_pred': base_pred
            }
            
            for bias in ['left', 'lean_left', 'center', 'lean_right', 'right']:
                # 각 미디어 바이어스에 대한 예측 생성
                model_data[art_id][f'{bias}_pred'] = np.random.choice(['left', 'lean_left', 'center', 'lean_right', 'right'])
        
        all_models[model] = model_data

# 샘플 데이터 생성 (실제 데이터로 대체하세요)
create_sample_data()

# -------------------------------------------------
# 1. 준비: 순서척도 & 헬퍼
# -------------------------------------------------
ORDER = {'left':-2, 'lean_left':-1, 'center':0,
         'lean_right':1, 'right':2}

def classify_case(base, new, media):
    """Neutral / Confirmation / Amplify / Attenuate / Reversal"""
    Δ = ORDER[new] - ORDER[base]
    dir_media = np.sign(ORDER[media])

    if Δ == 0:
        return 'confirmation' if ORDER[base]*dir_media > 0 else 'neutral'
    if Δ*dir_media > 0:              # 같은 방향 이동
        return 'amplify'   if abs(Δ) >= 1 else 'attenuate'
    return 'reversal'

def correction_or_distortion(base, new, label):
    """정답(label) 방향으로 가까워졌는지"""
    dist_base = abs(ORDER[label] - ORDER[base])
    dist_new  = abs(ORDER[label] - ORDER[new])
    if dist_new < dist_base:
        return 'correction'
    if dist_new > dist_base:
        return 'distortion'
    return 'same'

# -------------------------------------------------
# 2. 메인 루프 – 기사 × 모델 × media-bias
# -------------------------------------------------
def analyze_bias_cases():
    bias_list   = ['left','lean_left','center','lean_right','right']
    records     = []             # 최종 행별 레코드

    for model, article_dict in all_models.items():
        for art_id, res in article_dict.items():
            base_pred = res['baseline_pred']
            meta       = article_meta[art_id]
            art_label  = meta['label']
            topic      = meta.get('topic','N/A')
            source     = meta.get('source','N/A')
            hyperflag  = meta.get('bias_text','non-hp')

            for media in bias_list:
                new_pred = res[f'{media}_pred']
                case     = classify_case(base_pred, new_pred, media)
                impact   = correction_or_distortion(base_pred, new_pred, art_label)
                records.append({
                    'Model': model,
                    'ArticleID': art_id,
                    'MediaBias': media,
                    'ArticleLabel': art_label,
                    'BaselinePred': base_pred,
                    'NewPred': new_pred,
                    'Case': case,
                    'Impact': impact,
                    'Topic': topic,
                    'Source': source,
                    'HP': hyperflag
                })

    df = pd.DataFrame(records)
    out_dir = '../analyze_result/bias_case_analysis'
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f'{out_dir}/case_level_long.csv', index=False)
    print("🔹 case_level_long.csv 저장 완료")
    
    return df

# -------------------------------------------------
# 3. 집계 테이블 – 모델 × MediaBias × Case
# -------------------------------------------------
def create_pivot_table(df):
    pivot = (df.groupby(['Model','MediaBias','Case'])
            .size()
            .reset_index(name='Count'))
    out_dir = '../analyze_result/bias_case_analysis'
    pivot.to_csv(f'{out_dir}/case_counts.csv', index=False)
    return pivot

# -------------------------------------------------
# 4. 비율 히트맵 (Amplify + Reversal) / Total
# -------------------------------------------------
def create_heatmap(pivot):
    effect_cols = ['amplify','reversal','attenuate','confirmation','neutral']
    
    ratio_df = (pivot.pivot_table(index=['Model','MediaBias'],
                                columns='Case',
                                values='Count',
                                fill_value=0)
                    .reset_index())
    ratio_df['Total'] = ratio_df[effect_cols].sum(axis=1)
    ratio_df['Amp+Rev_Ratio'] = (ratio_df['amplify'] + ratio_df['reversal']) / ratio_df['Total']
    heat = ratio_df.pivot(index='MediaBias', columns='Model', values='Amp+Rev_Ratio')

    plt.figure(figsize=(12,4))
    sns.heatmap(heat, annot=True, fmt='.2f', cmap='Reds', cbar_kws={'label':'Amp+Rev ratio'})
    plt.title('Bias-Sensitive Ratio (Amplify+Reversal) by Model / MediaBias')
    out_dir = '../analyze_result/bias_case_analysis'
    plt.savefig(f'{out_dir}/heatmap_amp_rev_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return ratio_df

# -------------------------------------------------
# 5. 주제·출처 Over-representation (Amplify vs Neutral)
# -------------------------------------------------
def top_overrep(df, field, top_n=5):
    bias_list = ['left','lean_left','center','lean_right','right']
    out_rows = []
    for model, media in itertools.product(df['Model'].unique(), bias_list):
        sub = df[(df.Model==model)&(df.MediaBias==media)]
        if sub.empty: continue
        # 분할
        aff  = sub[sub.Case=='amplify'][field].tolist()
        neu  = sub[sub.Case=='neutral'][field].tolist()
        if not aff or not neu: continue
        cnt_aff, cnt_neu = Counter(aff), Counter(neu)
        tot_aff, tot_neu = len(aff), len(neu)
        over_scores = {
            k: (cnt_aff.get(k,0)/tot_aff) /
               max(cnt_neu.get(k,1)/tot_neu, 1e-3)
            for k in set(cnt_aff)|set(cnt_neu)
        }
        for k, sc in sorted(over_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            out_rows.append({
                'Model': model, 'MediaBias': media,
                'Field': field, 'Value': k, 'Overrep': sc
            })
    return pd.DataFrame(out_rows)

def analyze_overrep(df):
    topic_over  = top_overrep(df, 'Topic',  3)
    source_over = top_overrep(df, 'Source', 3)
    out_dir = '../analyze_result/bias_case_analysis'
    topic_over .to_csv(f'{out_dir}/overrep_topic.csv',  index=False)
    source_over.to_csv(f'{out_dir}/overrep_source.csv', index=False)
    print("🔹 overrep_topic.csv / overrep_source.csv 저장 완료")
    return topic_over, source_over

# -------------------------------------------------
# 6. 샘플 추출: 최대 Amplify & Distortion
# -------------------------------------------------
def extract_samples(df):
    top_amp = (df[df.Case=='amplify']
              .assign(Move=lambda r: abs(ORDER[r.NewPred]-ORDER[r.BaselinePred]))
              .sort_values('Move', ascending=False)
              .head(20))
    out_dir = '../analyze_result/bias_case_analysis'
    top_amp.to_csv(f'{out_dir}/top20_amplify_cases.csv', index=False)

    worst_dist = df[df.Impact=='distortion'].head(20)
    worst_dist.to_csv(f'{out_dir}/worst20_distortion.csv', index=False)
    return top_amp, worst_dist

# -------------------------------------------------
# 메인 실행
# -------------------------------------------------
def main():
    # 1. 데이터 로드
    # 실제 데이터 로드 코드 (가능하다면 사용)
    # load_model_data("경로", "모델명")
    # load_article_metadata("경로")
    
    # 2. 분석 실행
    df = analyze_bias_cases()
    pivot = create_pivot_table(df)
    ratio_df = create_heatmap(pivot)
    topic_over, source_over = analyze_overrep(df)
    top_amp, worst_dist = extract_samples(df)
    
    print("✅ 분석 · 요약 · 시각화 완료 -> ../analyze_result/bias_case_analysis")

if __name__ == "__main__":
    main() 