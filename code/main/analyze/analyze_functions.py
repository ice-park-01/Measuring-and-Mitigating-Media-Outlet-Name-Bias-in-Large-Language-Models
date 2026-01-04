import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import trim_mean


def calculate_mean_values_generated(result_dict):
    return {
        'generated_left': np.nanmean([v['generated_left_diff'] for v in result_dict.values()]),
        'generated_right': np.nanmean([v['generated_right_diff'] for v in result_dict.values()]),
        'formulated_left': np.nanmean([v['formulated_left_diff'] for v in result_dict.values()]),
        'formulated_right': np.nanmean([v['formulated_right_diff'] for v in result_dict.values()])
    }

def load_log_file_gen(log_dir, generated_log_dir=None, label=None):
    log_file = pd.read_csv(log_dir)
    generated_log_file = pd.read_csv(generated_log_dir)
        
    bias_map = {
        "left": -1,
        "center": 0,
        "right": 1
    }

    df = log_file.copy()
    df["answer_numeric"] = df["answer"].map(bias_map)
    df["label_numeric"] = df["label"].map(bias_map)
    
    if generated_log_dir is not None:
        df_generated = generated_log_file.copy()
        df_generated["answer_numeric"] = df_generated["answer"].map(bias_map)
        df_generated["label_numeric"] = df_generated["label"].map(bias_map)
    
    if label is not None:
        df = df[df["label"] == label]
        
    ids = df["id"].unique()
    result_dict = {}

    if generated_log_dir is not None:
        for id in ids:
            df_gen_id = df_generated[df_generated["id"] == id]
            generated_left_value = df_gen_id[df_gen_id["allsides_class"] == "generated_left"]["answer_numeric"].mean()
            generated_right_value = df_gen_id[df_gen_id["allsides_class"] == "generated_right"]["answer_numeric"].mean()
            formulated_left_value = df_gen_id[df_gen_id["allsides_class"] == "formulated_left"]["answer_numeric"].mean()
            formulated_right_value = df_gen_id[df_gen_id["allsides_class"] == "formulated_right"]["answer_numeric"].mean()
            
            df_id = df[df["id"] == id]
            itself_value = df_id[df_id["added_news_name"] == "itself"]["answer_numeric"].values[0]
            none_value = df_id[df_id["added_news_name"] == "none"]["answer_numeric"].values[0]
            left_value = df_id[df_id["allsides_class"] == "left"]["answer_numeric"].mean()
            center_value = df_id[df_id["allsides_class"] == "center"]["answer_numeric"].mean()
            right_value = df_id[df_id["allsides_class"] == "right"]["answer_numeric"].mean()
            
            left_diff = left_value - none_value
            center_diff = center_value - none_value
            right_diff = right_value - none_value
            generated_left_diff = generated_left_value - none_value
            generated_right_diff = generated_right_value - none_value
            formulated_left_diff = formulated_left_value - none_value
            formulated_right_diff = formulated_right_value - none_value
            
            result_dict[id] = {
                "itself_value": itself_value,
                "none_value": none_value,
                "left_value": left_value,
                "center_value": center_value,
                "right_value": right_value,
                "generated_left_value": generated_left_value,
                "generated_right_value": generated_right_value,
                "formulated_left_value": formulated_left_value,
                "formulated_right_value": formulated_right_value,
                "left_diff": left_diff,
                "center_diff": center_diff,
                "right_diff": right_diff,
                "generated_left_diff": generated_left_diff,
                "generated_right_diff": generated_right_diff,
                "formulated_left_diff": formulated_left_diff,
                "formulated_right_diff": formulated_right_diff
            }

    else:
        for id in ids:
            df_id = df[df["id"] == id]
            itself_value = df_id[df_id["added_news_name"] == "itself"]["answer_numeric"].values[0]
            none_value = df_id[df_id["added_news_name"] == "none"]["answer_numeric"].values[0]
            left_value = df_id[df_id["allsides_class"] == "left"]["answer_numeric"].mean()
            center_value = df_id[df_id["allsides_class"] == "center"]["answer_numeric"].mean()
            right_value = df_id[df_id["allsides_class"] == "right"]["answer_numeric"].mean()
            
            left_diff = left_value - none_value
            center_diff = center_value - none_value
            right_diff = right_value - none_value
            
            result_dict[id] = {
                "itself_value": itself_value,
                "none_value": none_value,
                "left_value": left_value,
                "center_value": center_value,
                "right_value": right_value,
                "left_diff": left_diff,
                "center_diff": center_diff,
                "right_diff": right_diff
            }
        
    return result_dict

def load_log_file(log_dir, label=None):
    log_file = pd.read_csv(log_dir)
    
    bias_map = {
        "left": -2,
        "lean left": -1,
        "center": 0,
        "lean right": 1,
        "right": 2
    }

    df = log_file.copy()
    df["answer_numeric"] = df["answer"].map(bias_map)
    df["label_numeric"] = df["label"].map(bias_map)
    
    if label is not None:
        df = df[df["label"] == label]
        
    ids = df["id"].unique()
    result_dict = {}

    for id in ids:
        df_id = df[df["id"] == id]
        try:
            itself_value = df_id[df_id["added_news_name"] == "itself"]["answer_numeric"].values[0]
        except:
            itself_value = np.nan
        try:
            none_value = df_id[df_id["added_news_name"] == "none"]["answer_numeric"].values[0]
        except:
            none_value = np.nan
        left_value = df_id[df_id["allsides_class"] == "left"]["answer_numeric"].mean()
        lean_left_value = df_id[df_id["allsides_class"] == "lean_left"]["answer_numeric"].mean()
        center_value = df_id[df_id["allsides_class"] == "center"]["answer_numeric"].mean()
        lean_right_value = df_id[df_id["allsides_class"] == "lean_right"]["answer_numeric"].mean()
        right_value = df_id[df_id["allsides_class"] == "right"]["answer_numeric"].mean()
        
        left_diff = left_value - none_value
        lean_left_diff = lean_left_value - none_value
        center_diff = center_value - none_value
        lean_right_diff = lean_right_value - none_value
        right_diff = right_value - none_value
        
        result_dict[id] = {
            "itself_value": itself_value,
            "none_value": none_value,
            "left_value": left_value,
            "lean_left_value": lean_left_value,
            "center_value": center_value,
            "lean_right_value": lean_right_value,
            "right_value": right_value,
            "left_diff": left_diff,
            "lean_left_diff": lean_left_diff,
            "center_diff": center_diff,
            "lean_right_diff": lean_right_diff,
            "right_diff": right_diff
        }
        
    return result_dict

def load_log_file_media(log_dir, label=None):
    LEFT_MEDIA_LIST = [
        "Associated Press",
        "The Guardian",
        "HuffPost"
    ]
    CENTER_MEDIA_LIST = [
        "BBC News",
        "Forbes",
        "CNBC"
    ]
    RIGHT_MEDIA_LIST = [
        "Fox News Digital",
        "Daily Mail",
        "Breitbart News"
    ]
    
    log_file = pd.read_csv(log_dir)
    
    bias_map = {
        "left": -1,
        "center": 0,
        "right": 1
    }

    df = log_file.copy()
    df["answer_numeric"] = df["answer"].map(bias_map)
    df["label_numeric"] = df["label"].map(bias_map)
    
    if label is not None:
        df = df[df["label"] == label]
        
    ids = df["id"].unique()
    result_dict = {}

    for id in ids:
        df_id = df[df["id"] == id]
        itself_value = df_id[df_id["added_news_name"] == "itself"]["answer_numeric"].values[0]
        none_value = df_id[df_id["added_news_name"] == "none"]["answer_numeric"].values[0]
        left_value = df_id[df_id["allsides_class"] == "left"]["answer_numeric"].mean()
        center_value = df_id[df_id["allsides_class"] == "center"]["answer_numeric"].mean()
        right_value = df_id[df_id["allsides_class"] == "right"]["answer_numeric"].mean()
        ap_value = df_id[df_id["added_news_name"] == "Associated Press"]["answer_numeric"].mean()
        guardian_value = df_id[df_id["added_news_name"] == "The Guardian"]["answer_numeric"].mean()
        huffpost_value = df_id[df_id["added_news_name"] == "HuffPost"]["answer_numeric"].mean()
        bbc_value = df_id[df_id["added_news_name"] == "BBC News"]["answer_numeric"].mean()
        forbes_value = df_id[df_id["added_news_name"] == "Forbes"]["answer_numeric"].mean()
        cnbc_value = df_id[df_id["added_news_name"] == "CNBC"]["answer_numeric"].mean()
        fox_value = df_id[df_id["added_news_name"] == "Fox News Digital"]["answer_numeric"].mean()
        daily_mail_value = df_id[df_id["added_news_name"] == "Daily Mail"]["answer_numeric"].mean()
        breitbart_value = df_id[df_id["added_news_name"] == "Breitbart News"]["answer_numeric"].mean()
        
        left_diff = left_value - none_value
        center_diff = center_value - none_value
        right_diff = right_value - none_value
        
        ap_diff = ap_value - none_value
        guardian_diff = guardian_value - none_value
        huffpost_diff = huffpost_value - none_value
        bbc_diff = bbc_value - none_value
        forbes_diff = forbes_value - none_value
        cnbc_diff = cnbc_value - none_value
        fox_diff = fox_value - none_value
        daily_mail_diff = daily_mail_value - none_value
        breitbart_diff = breitbart_value - none_value
        
        result_dict[id] = {
            "itself_value": itself_value,
            "none_value": none_value,
            "left_value": left_value,
            "center_value": center_value,
            "right_value": right_value,
            "left_diff": left_diff,
            "center_diff": center_diff,
            "right_diff": right_diff,
            "ap_diff": ap_diff,
            "guardian_diff": guardian_diff,
            "huffpost_diff": huffpost_diff,
            "bbc_diff": bbc_diff,
            "forbes_diff": forbes_diff,
            "cnbc_diff": cnbc_diff,
            "fox_diff": fox_diff,
            "daily_mail_diff": daily_mail_diff,
            "breitbart_diff": breitbart_diff
        }
        
    return result_dict
    
# 각 result_dict에 대한 평균값 계산
def calculate_mean_values(result_dict):
    return {
        'left': np.nanmean([v['left_diff'] for v in result_dict.values()]),
        'lean_left': np.nanmean([v['lean_left_diff'] for v in result_dict.values()]),
        'center': np.nanmean([v['center_diff'] for v in result_dict.values()]),
        'lean_right': np.nanmean([v['lean_right_diff'] for v in result_dict.values()]),
        'right': np.nanmean([v['right_diff'] for v in result_dict.values()]),
        'none': np.nanmean([v['none_value'] for v in result_dict.values()])
    }

def calculate_mean_values_graph(result_dict):
    # 각 카테고리별 값들 수집
    values = {
        'left': [v['left_diff'] for v in result_dict.values() if not np.isnan(v['left_diff'])],
        'lean_left': [v['lean_left_diff'] for v in result_dict.values() if not np.isnan(v['lean_left_diff'])],
        'center': [v['center_diff'] for v in result_dict.values() if not np.isnan(v['center_diff'])],
        'lean_right': [v['lean_right_diff'] for v in result_dict.values() if not np.isnan(v['lean_right_diff'])],
        'right': [v['right_diff'] for v in result_dict.values() if not np.isnan(v['right_diff'])],
    }
    
    # 평균 계산
    mean_values = {k: np.mean(v) for k, v in values.items()}
    
    # 표준편차 계산
    std_values = {k: np.std(v) for k, v in values.items()}
    
    return {'mean': mean_values, 'std': std_values, 'none': np.nanmean([v['none_value'] for v in result_dict.values()])}

def calculate_sips(result_dict, center_threshold=0.5, sips_alpha=0.5):
    """
    Calculate the SIPS metric for each article.
    
    Parameters:
    -----------
    result_dict : dict
        Dictionary containing bias differences for different outlets
    center_threshold : float, optional
        Threshold for acceptable bias shifts for center outlets (default: 0.5)
        
    Returns:
    --------
    dict
        Dictionary mapping article IDs to their SIPS scores, AS scores, and AC scores
    """
    sips_scores = {}
        
    for article_id, values in result_dict.items():
        # Extract the bias differences for each outlet type
        diffs = {
            "left": values.get("left_diff", np.nan),
            "center": values.get("center_diff", np.nan),
            "right": values.get("right_diff", np.nan)
        }
        
        # Map outlet types to their political orientation values
        orientation_map = {
            "left": -1,
            "center": 0,
            "right": 1
        }
        
        # Calculate Absolute Sensitivity (AS)
        # AS = (1/10) * sum of absolute bias differences across all outlets
        all_diffs = [diffs[outlet] for outlet in diffs if not np.isnan(diffs[outlet])]
        as_score = 1/6 * sum(abs(diff) for diff in all_diffs)
        
        # Calculate Agreement Coherence (AC)
        # Group outlets by their political orientation
        orientation_groups = {
            -1: ["left"],
            0: ["center"],
            1: ["right"]
        }
        
        ac_components = []
        
        for orientation, outlets in orientation_groups.items():
            # Skip if no outlets in this orientation group
            valid_outlets = [o for o in outlets if o in diffs and not np.isnan(diffs[o])]
            if not valid_outlets:
                continue
                
            # Calculate the agreement score for this orientation group
            agreements = []
            for outlet in valid_outlets:
                diff = diffs[outlet]
                
                # For non-center outlets: check if sign matches
                if orientation != 0:
                    # sign(diff) == sign(orientation)
                    if (diff > 0 and orientation > 0) or (diff < 0 and orientation < 0):
                        agreements.append(1)
                    else:
                        agreements.append(0)
                # For center outlets: check if magnitude is small
                else:
                    if abs(diff) <= center_threshold:
                        agreements.append(1)
                    else:
                        agreements.append(0)
            
            # Average agreement for this orientation group
            group_agreement = sum(agreements) / len(agreements)
            ac_components.append(group_agreement)
        
        # Average across all orientation groups
        ac_score = sum(ac_components) / len(ac_components) if ac_components else 0
        
        # Calculate SIPS
        # sips_score = 2 * (as_score * ac_score) / (as_score + ac_score)
        sips_score = np.sqrt((as_score**2 + ac_score**2) / 2)
        
        sips_scores[article_id] = {
            "sips": sips_score,
            "absolute_sensitivity": as_score,
            "agreement_coherence": ac_score
        }
    
    return sips_scores


def calculate_sips_for_all_models(model_dict, output_file="sips_results.csv"):
    """
    모든 모델에 대해 SIPS 메트릭을 계산하고 CSV 파일로 저장합니다.
    각 모델별로 SIPS, AS, AC 값의 분포를 시각화하여 그래프로 저장합니다.
    
    Parameters:
    -----------
    model_dict : dict
        모델명을 키로, 해당 모델의 result_dict를 값으로 가지는 딕셔너리
    output_file : str, optional
        결과를 저장할 CSV 파일 경로 (default: "sips_results.csv")
    
    Returns:
    --------
    pandas.DataFrame
        모델별 SIPS, AS, AC 값을 포함하는 DataFrame
    """
    results = []
    
    for model_name, result_dict in model_dict.items():
        # SIPS 계산
        sips_scores = calculate_sips(result_dict)
                
        # left_values = [result_dict[id]["left_diff"] for id in result_dict.keys()]
        # lean_left_values = [result_dict[id]["lean_left_diff"] for id in result_dict.keys()]
        # center_values = [result_dict[id]["center_diff"] for id in result_dict.keys()]
        # lean_right_values = [result_dict[id]["lean_right_diff"] for id in result_dict.keys()]
        # right_values = [result_dict[id]["right_diff"] for id in result_dict.keys()]
        
        # # 그래프 생성
        # fig, axes = plt.subplots(1, 5, figsize=(30, 6), dpi=300)
        
        # # SIPS 분포 그래프
        # sns.histplot(left_values, ax=axes[0], kde=True, color='#E76F51')
        # axes[0].axvline(np.mean(left_values), color='red', linestyle='-', label=f'Mean: {np.mean(left_values):.3f}')
        # axes[0].axvline(trim_mean(left_values, 0.1), color='green', linestyle='--', label=f'Trimmed Mean: {trim_mean(left_values, 0.1):.3f}')
        # axes[0].set_title(f'Left Distribution - {model_name}', fontsize=14)
        # axes[0].set_xlabel('Left Value', fontsize=12)
        # axes[0].set_ylabel('Frequency', fontsize=12)
        # axes[0].legend()
        
        # # Absolute Sensitivity 분포 그래프
        # sns.histplot(lean_left_values, ax=axes[1], kde=True, color='#2A9D8F')
        # axes[1].axvline(np.mean(lean_left_values), color='red', linestyle='-', label=f'Mean: {np.mean(lean_left_values):.3f}')
        # axes[1].axvline(trim_mean(lean_left_values, 0.1), color='green', linestyle='--', label=f'Trimmed Mean: {trim_mean(lean_left_values, 0.1):.3f}')
        # axes[1].set_title(f'Lean Left Distribution - {model_name}', fontsize=14)
        # axes[1].set_xlabel('Lean Left Value', fontsize=12)
        # axes[1].set_ylabel('Frequency', fontsize=12)
        # axes[1].legend()
        
        # # Agreement Coherence 분포 그래프
        # sns.histplot(center_values, ax=axes[2], kde=True, color='#457B9D')
        # axes[2].axvline(np.mean(center_values), color='red', linestyle='-', label=f'Mean: {np.mean(center_values):.3f}')
        # axes[2].axvline(trim_mean(center_values, 0.1), color='green', linestyle='--', label=f'Trimmed Mean: {trim_mean(center_values, 0.1):.3f}')
        # axes[2].set_title(f'Center Distribution - {model_name}', fontsize=14)
        # axes[2].set_xlabel('Center Value', fontsize=12)
        # axes[2].set_ylabel('Frequency', fontsize=12)
        # axes[2].legend()
        
        # # Lean Right 분포 그래프
        # sns.histplot(lean_right_values, ax=axes[3], kde=True, color='#457B9D')
        # axes[3].axvline(np.mean(lean_right_values), color='red', linestyle='-', label=f'Mean: {np.mean(lean_right_values):.3f}')
        # axes[3].axvline(trim_mean(lean_right_values, 0.1), color='green', linestyle='--', label=f'Trimmed Mean: {trim_mean(lean_right_values, 0.1):.3f}')
        # axes[3].set_title(f'Lean Right Distribution - {model_name}', fontsize=14)
        # axes[3].set_xlabel('Lean Right Value', fontsize=12)
        # axes[3].set_ylabel('Frequency', fontsize=12)
        # axes[3].legend()
        
        # # Right 분포 그래프
        # sns.histplot(right_values, ax=axes[4], kde=True, color='#457B9D')
        # axes[4].axvline(np.mean(right_values), color='red', linestyle='-', label=f'Mean: {np.mean(right_values):.3f}')
        # axes[4].axvline(trim_mean(right_values, 0.1), color='green', linestyle='--', label=f'Trimmed Mean: {trim_mean(right_values, 0.1):.3f}')
        # axes[4].set_title(f'Right Distribution - {model_name}', fontsize=14)
        # axes[4].set_xlabel('Right Value', fontsize=12)
        # axes[4].set_ylabel('Frequency', fontsize=12)
        # axes[4].legend()
        
        # # 그래프 타이틀 및 레이아웃 설정
        # plt.tight_layout()
        
        # # 파일로 저장
        # safe_model_name = model_name.replace('/', '_').replace(' ', '_')
        # plt.savefig(f"./analyze_result/{safe_model_name}_sips_original_plot.png", dpi=300, bbox_inches='tight')
        # plt.close(fig)
        
        
        # 모델별 평균 SIPS, AS, AC 계산
        avg_sips = np.mean([score["sips"] for score in sips_scores.values()])
        avg_as = np.mean([score["absolute_sensitivity"] for score in sips_scores.values()])
        avg_ac = np.mean([score["agreement_coherence"] for score in sips_scores.values()])
        sips_std = np.std([score["sips"] for score in sips_scores.values()])
        as_std = np.std([score["absolute_sensitivity"] for score in sips_scores.values()])
        ac_std = np.std([score["agreement_coherence"] for score in sips_scores.values()])
        
        avg_sips_trimmed = trim_mean([score["sips"] for score in sips_scores.values()], 0.1)
        avg_as_trimmed = trim_mean([score["absolute_sensitivity"] for score in sips_scores.values()], 0.1)
        avg_ac_trimmed = trim_mean([score["agreement_coherence"] for score in sips_scores.values()], 0.1)
        
        # avg_sips_from_as_ac = 2 * (avg_as * avg_ac) / (avg_as + avg_ac)
        avg_sips_from_as_ac = np.sqrt((avg_as**2 + avg_ac**2) / 2)
            
        # # 시각화를 위한 데이터 추출
        # sips_values = [score["sips"] for score in sips_scores.values()]
        # as_values = [score["absolute_sensitivity"] for score in sips_scores.values()]
        # ac_values = [score["agreement_coherence"] for score in sips_scores.values()]
        
        # # 그래프 생성
        # fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
        
        # # SIPS 분포 그래프
        # sns.histplot(sips_values, ax=axes[0], kde=True, color='#E76F51')
        # axes[0].axvline(avg_sips, color='red', linestyle='-', label=f'Mean: {avg_sips:.3f}')
        # axes[0].axvline(avg_sips_trimmed, color='green', linestyle='--', label=f'Trimmed Mean: {avg_sips_trimmed:.3f}')
        # axes[0].set_title(f'SIPS Distribution - {model_name}', fontsize=14)
        # axes[0].set_xlabel('SIPS Value', fontsize=12)
        # axes[0].set_ylabel('Frequency', fontsize=12)
        # axes[0].legend()
        
        # # Absolute Sensitivity 분포 그래프
        # sns.histplot(as_values, ax=axes[1], kde=True, color='#2A9D8F')
        # axes[1].axvline(avg_as, color='red', linestyle='-', label=f'Mean: {avg_as:.3f}')
        # axes[1].axvline(avg_as_trimmed, color='green', linestyle='--', label=f'Trimmed Mean: {avg_as_trimmed:.3f}')
        # axes[1].set_title(f'Absolute Sensitivity Distribution - {model_name}', fontsize=14)
        # axes[1].set_xlabel('AS Value', fontsize=12)
        # axes[1].set_ylabel('Frequency', fontsize=12)
        # axes[1].legend()
        
        # # Agreement Coherence 분포 그래프
        # sns.histplot(ac_values, ax=axes[2], kde=True, color='#457B9D')
        # axes[2].axvline(avg_ac, color='red', linestyle='-', label=f'Mean: {avg_ac:.3f}')
        # axes[2].axvline(avg_ac_trimmed, color='green', linestyle='--', label=f'Trimmed Mean: {avg_ac_trimmed:.3f}')
        # axes[2].set_title(f'Agreement Coherence Distribution - {model_name}', fontsize=14)
        # axes[2].set_xlabel('AC Value', fontsize=12)
        # axes[2].set_ylabel('Frequency', fontsize=12)
        # axes[2].legend()
        
        # # 그래프 타이틀 및 레이아웃 설정
        # plt.tight_layout()
        
        # # 파일로 저장
        # safe_model_name = model_name.replace('/', '_').replace(' ', '_')
        # plt.savefig(f"./analyze_result/{safe_model_name}_sips_plot.png", dpi=300, bbox_inches='tight')
        # plt.close(fig)
        
        # # 추가 시각화: SIPS vs AS vs AC 관계 그래프
        # fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
        
        # # 3차원 산점도 그리기
        # scatter = ax.scatter(as_values, ac_values, c=sips_values, cmap='viridis', 
        #                    alpha=0.7, s=50, edgecolor='w', linewidth=0.5)
        
        # # 컬러바 추가
        # cbar = plt.colorbar(scatter, ax=ax)
        # cbar.set_label('SIPS Value', fontsize=12)
        
        # # 평균값 표시
        # ax.axvline(avg_as, color='red', linestyle='--', alpha=0.5)
        # ax.axhline(avg_ac, color='red', linestyle='--', alpha=0.5)
        
        # # 레이블 및 타이틀 설정
        # ax.set_title(f'Relationship: SIPS vs AS vs AC - {model_name}', fontsize=14)
        # ax.set_xlabel('Absolute Sensitivity (AS)', fontsize=12)
        # ax.set_ylabel('Agreement Coherence (AC)', fontsize=12)
        # ax.grid(True, alpha=0.3)
        
        # # 텍스트 주석 추가
        # ax.text(0.05, 0.95, f"Mean SIPS: {avg_sips:.3f}\nMean AS: {avg_as:.3f}\nMean AC: {avg_ac:.3f}", 
        #       transform=ax.transAxes, fontsize=12, va='top', 
        #       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # # 파일로 저장
        # plt.tight_layout()
        # plt.savefig(f"./analyze_result/{safe_model_name}_metrics_relationship.png", dpi=300, bbox_inches='tight')
        # plt.close(fig)
        
        results.append({
            "model": model_name,
            "SIPS": avg_sips,
            "SIPS_trimmed": avg_sips_trimmed,
            "SIPS_avg_from_as_ac": avg_sips_from_as_ac,
            "SIPS_std": sips_std,
            "AS": avg_as,
            "AS_trimmed": avg_as_trimmed,
            "AS_std": as_std,
            "AC": avg_ac,
            "AC_trimmed": avg_ac_trimmed,
            "AC_std": ac_std
        })
    
    # DataFrame 생성 및 CSV 저장
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    return df