import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random

from analyze_functions import load_log_file, calculate_mean_values, load_log_file_gen, calculate_mean_values_generated, calculate_sips, calculate_sips_for_all_models, calculate_mean_values_graph
import matplotlib.patheffects as path_effects
from adjustText import adjust_text
from sentence_transformers import SentenceTransformer


def analyze_main_result_allsides():
    qwen_log_dir = "../../logs/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias/20250219_145141/result_4400.csv"
    mistral_log_dir = "../../logs/allsides/mistralai/Mistral-Small-24B-Instruct-2501/random_medias/20250221_124715/result_4499.csv"
    phi_log_dir = "../../logs/allsides/microsoft/phi-4/random_medias/20250219_145326/result_4400.csv"
    llama_log_dir = "../../logs/allsides/meta-llama/Llama-3.3-70B-Instruct/random_medias/20250219_145234/result_4400.csv"
    gemma_log_dir = "../../logs/allsides/google/gemma-2-27b-it/random_medias/20250410_003055/result_4499.csv"
    gpt_4_1_log_dir = "../../logs/allsides/gpt-4.1/random_medias/allsides/20250512_160756/result_4499.csv"

    qwen_result_dict = load_log_file(qwen_log_dir)
    mistral_result_dict = load_log_file(mistral_log_dir)
    phi_result_dict = load_log_file(phi_log_dir)
    llama_result_dict = load_log_file(llama_log_dir)
    gemma_result_dict = load_log_file(gemma_log_dir)
    gpt_4_1_result_dict = load_log_file(gpt_4_1_log_dir)
    
    # 데이터 준비
    x_labels = ['left', 'lean_left', 'center', 'lean_right', 'right']
    x_positions = np.arange(len(x_labels))

    # 각 result_dict에 대한 평균값
    qwen_mean_values = calculate_mean_values(qwen_result_dict)
    mistral_mean_values = calculate_mean_values(mistral_result_dict)
    phi_mean_values = calculate_mean_values(phi_result_dict)
    llama_mean_values = calculate_mean_values(llama_result_dict)
    gemma_mean_values = calculate_mean_values(gemma_result_dict)
    gpt_4_1_mean_values = calculate_mean_values(gpt_4_1_result_dict)
    
    # 그래프 그리기
    plt.figure(figsize=(6, 6), dpi=300)  # 해상도 향상을 위해 크기와 DPI 증가

    # 폰트 설정
    # SF Pro 폰트를 찾을 수 없으므로 시스템 기본 폰트 사용
    plt.rcParams['font.family'] = 'sans-serif'  # 범용적인 sans-serif 폰트 사용
    plt.rcParams['font.size'] = 12  # 기본 글씨 크기 증가

    # 각 모델의 평균값 플롯
    plt.plot(x_positions, [qwen_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#E76F51', label='Qwen')
    plt.plot(x_positions, [mistral_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#E9C46A', label='Mistral')
    plt.plot(x_positions, [phi_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#2A9D8F', label='Phi')
    plt.plot(x_positions, [llama_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#457B9D', label='Llama')
    plt.plot(x_positions, [gemma_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#8D6AA9', label='Gemma')
    plt.plot(x_positions, [gpt_4_1_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#2A9D44', label='GPT-4.1')

    # Reference Line: Bias Class가 left -> right일 때 -2 -> 2
    biased_line = np.linspace(-2, 2, len(x_labels))
    plt.plot(x_positions, biased_line, linestyle='--', color='#2F2F2F')
    plt.text(x_positions[3] - 0.3, biased_line[3] + 0.7, "Reference Line - Biased", ha='center', va='bottom', fontsize=10, color='#2F2F2F')

    # Reference Line: 모든 때 0
    plt.axhline(0, color='#A8A8A8', linestyle='--')
    plt.text(x_positions[3] + 0.3, -0.3, "Reference Line - Unbiased", ha='center', va='bottom', fontsize=10, color='#A8A8A8')

    plt.xticks(x_positions, x_labels, fontsize=10)
    plt.xlabel('Bias Class', fontsize=11)
    plt.ylabel('Answer Numeric Value', fontsize=11)
    # plt.title('Bias Class별 Answer Numeric 값 분포')
    plt.ylim(-2, 2)  # y축 척도 고정
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()  # 여백 자동 조정
    plt.savefig("./analyze_result/main_figures/main_result_allsides.png", dpi=300, bbox_inches='tight')
    
    
def analyze_main_result_hyperpartisan():
    qwen_log_dir = "../../logs/hyperpartisan/Qwen/Qwen2.5-72B-Instruct/hyperpartisan/20250408_232523/result_1272.csv"
    mistral_log_dir = "../../logs/hyperpartisan/mistralai/Mistral-Small-24B-Instruct-2501/hyperpartisan/20250408_232519/result_1272.csv"
    phi_log_dir = "../../logs/hyperpartisan/microsoft/phi-4/hyperpartisan/20250408_232529/result_1272.csv"
    llama_log_dir = "../../logs/hyperpartisan/meta-llama/Llama-3.3-70B-Instruct/hyperpartisan/20250409_033417/result_1272.csv"
    gemma_log_dir = "../../logs/hyperpartisan/google/gemma-2-27b-it/hyperpartisan/20250411_200354/result_1272.csv"

    qwen_result_dict = load_log_file(qwen_log_dir)
    mistral_result_dict = load_log_file(mistral_log_dir)
    phi_result_dict = load_log_file(phi_log_dir)
    llama_result_dict = load_log_file(llama_log_dir)
    gemma_result_dict = load_log_file(gemma_log_dir)
    
    # 데이터 준비
    x_labels = ['left', 'lean_left', 'center', 'lean_right', 'right']
    x_positions = np.arange(len(x_labels))

    # 각 result_dict에 대한 평균값
    qwen_mean_values = calculate_mean_values(qwen_result_dict)
    mistral_mean_values = calculate_mean_values(mistral_result_dict)
    phi_mean_values = calculate_mean_values(phi_result_dict)
    llama_mean_values = calculate_mean_values(llama_result_dict)
    gemma_mean_values = calculate_mean_values(gemma_result_dict)
    
    # 그래프 그리기
    plt.figure(figsize=(6, 6), dpi=300)  # 해상도 향상을 위해 크기와 DPI 증가

    # 폰트 설정
    # SF Pro 폰트를 찾을 수 없으므로 시스템 기본 폰트 사용
    plt.rcParams['font.family'] = 'sans-serif'  # 범용적인 sans-serif 폰트 사용
    plt.rcParams['font.size'] = 12  # 기본 글씨 크기 증가

    # 각 모델의 평균값 플롯
    plt.plot(x_positions, [qwen_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#E76F51', label='Qwen')
    plt.plot(x_positions, [mistral_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#E9C46A', label='Mistral')
    plt.plot(x_positions, [phi_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#2A9D8F', label='Phi')
    plt.plot(x_positions, [llama_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#457B9D', label='Llama')
    plt.plot(x_positions, [gemma_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#8D6AA9', label='Gemma')

    # Reference Line: Bias Class가 left -> right일 때 -2 -> 2
    biased_line = np.linspace(-2, 2, len(x_labels))
    plt.plot(x_positions, biased_line, linestyle='--', color='#2F2F2F')
    plt.text(x_positions[3] - 0.3, biased_line[3] + 0.7, "Reference Line - Biased", ha='center', va='bottom', fontsize=10, color='#2F2F2F')

    # Reference Line: 모든 때 0
    plt.axhline(0, color='#A8A8A8', linestyle='--')
    plt.text(x_positions[3] + 0.3, -0.3, "Reference Line - Unbiased", ha='center', va='bottom', fontsize=10, color='#A8A8A8')

    plt.xticks(x_positions, x_labels, fontsize=10)
    plt.xlabel('Bias Class', fontsize=11)
    plt.ylabel('Answer Numeric Value', fontsize=11)
    # plt.title('Bias Class별 Answer Numeric 값 분포')
    plt.ylim(-2, 2)  # y축 척도 고정
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()  # 여백 자동 조정
    plt.savefig("./analyze_result/main_result_hyperpartisan.png", dpi=300, bbox_inches='tight')
    
    
def analyze_model_size_qwen():
    qwen_70_log_dir = "../../logs/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias/20250219_145141/result_4400.csv"
    qwen_32_log_dir = "../../logs/allsides/Qwen/Qwen2.5-32B-Instruct/random_medias/20250225_190144/result_4499.csv"
    qwen_14_log_dir = "../../logs/allsides/Qwen/Qwen2.5-14B-Instruct/random_medias/20250225_190224/result_4499.csv"
    qwen_7_log_dir = "../../logs/allsides/Qwen/Qwen2.5-7B-Instruct/random_medias/20250225_194956/result_4499.csv"
    qwen_3_log_dir = "../../logs/allsides/Qwen/Qwen2.5-3B-Instruct/random_medias/20250226_175204/result_4499.csv"
    qwen_15_log_dir = "../../logs/allsides/Qwen/Qwen2.5-1.5B-Instruct/random_medias/20250226_175241/result_4499.csv"
    qwen_05_log_dir = "../../logs/allsides/Qwen/Qwen2.5-0.5B-Instruct/random_medias/20250226_232257/result_4499.csv"
    
    qwen_70_result_dict = load_log_file(qwen_70_log_dir)
    qwen_32_result_dict = load_log_file(qwen_32_log_dir)
    qwen_14_result_dict = load_log_file(qwen_14_log_dir)
    qwen_7_result_dict = load_log_file(qwen_7_log_dir)
    qwen_3_result_dict = load_log_file(qwen_3_log_dir)
    qwen_15_result_dict = load_log_file(qwen_15_log_dir)
    qwen_05_result_dict = load_log_file(qwen_05_log_dir)
    
    
    # 데이터 준비
    x_labels = ['left', 'lean_left', 'center', 'lean_right', 'right']
    x_positions = np.arange(len(x_labels))

    # 각 result_dict에 대한 평균값
    qwen_70_mean_values = calculate_mean_values(qwen_70_result_dict)
    qwen_32_mean_values = calculate_mean_values(qwen_32_result_dict)
    qwen_14_mean_values = calculate_mean_values(qwen_14_result_dict)
    qwen_7_mean_values = calculate_mean_values(qwen_7_result_dict)
    qwen_3_mean_values = calculate_mean_values(qwen_3_result_dict)
    qwen_15_mean_values = calculate_mean_values(qwen_15_result_dict)
    qwen_05_mean_values = calculate_mean_values(qwen_05_result_dict)

    # 그래프 그리기
    plt.figure(figsize=(6, 6), dpi=300)  # 해상도 향상을 위해 크기와 DPI 증가

    # 폰트 설정
    # SF Pro 폰트를 찾을 수 없으므로 시스템 기본 폰트 사용
    plt.rcParams['font.family'] = 'sans-serif'  # 범용적인 sans-serif 폰트 사용
    plt.rcParams['font.size'] = 12  # 기본 글씨 크기 증가

    # 각 모델의 평균값 플롯
    plt.plot(x_positions, [qwen_70_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#E76F51', label='Qwen-72B')
    plt.plot(x_positions, [qwen_32_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#EA8D6E', label='Qwen-32B')
    plt.plot(x_positions, [qwen_14_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#EDAA8C', label='Qwen-14B')
    plt.plot(x_positions, [qwen_7_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#F0C7AA', label='Qwen-7B')
    plt.plot(x_positions, [qwen_3_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#F3E3C8', label='Qwen-3B')
    plt.plot(x_positions, [qwen_15_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#F6F0E0', label='Qwen-1.5B')
    plt.plot(x_positions, [qwen_05_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#FCFBF7', label='Qwen-0.5B')

    # Reference Line: Bias Class가 left -> right일 때 -2 -> 2
    biased_line = np.linspace(-2, 2, len(x_labels))
    plt.plot(x_positions, biased_line, linestyle='--', color='#2F2F2F')
    plt.text(x_positions[3] - 0.3, biased_line[3] + 0.7, "Reference Line - Biased", ha='center', va='bottom', fontsize=10, color='#2F2F2F')

    # Reference Line: 모든 때 0
    plt.axhline(0, color='#A8A8A8', linestyle='--')
    plt.text(x_positions[3] + 0.3, -0.3, "Reference Line - Unbiased", ha='center', va='bottom', fontsize=10, color='#A8A8A8')

    plt.xticks(x_positions, x_labels, fontsize=10)
    plt.xlabel('Bias Class', fontsize=11)
    plt.ylabel('Answer Numeric Value', fontsize=11)
    # plt.title('Bias Class별 Answer Numeric 값 분포')
    plt.ylim(-2, 2)  # y축 척도 고정
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()  # 여백 자동 조정
    plt.savefig("./analyze_result/size_difference_qwen.png", dpi=300, bbox_inches='tight')
    
    
def analyze_model_size_llama():
    llama_31_70_log_dir = "../../logs/allsides/meta-llama/Llama-3.1-70B-Instruct/random_medias/20250310_223517/result_4499.csv"
    llama_31_8_log_dir = "../../logs/allsides/meta-llama/Llama-3.1-8B-Instruct/random_medias/20250411_195339/result_4499.csv"
    llama_3_70_log_dir = "../../logs/allsides/meta-llama/Meta-Llama-3-70B-Instruct/random_medias/20250311_150159/result_4499.csv"
    llama_3_8_log_dir = "../../logs/allsides/meta-llama/Meta-Llama-3-8B-Instruct/random_medias/20250412_134400/result_4499.csv"
    
    llama_31_70_result_dict = load_log_file(llama_31_70_log_dir)
    llama_31_8_result_dict = load_log_file(llama_31_8_log_dir)
    llama_3_70_result_dict = load_log_file(llama_3_70_log_dir)
    llama_3_8_result_dict = load_log_file(llama_3_8_log_dir)
    
    # 데이터 준비
    x_labels = ['left', 'lean_left', 'center', 'lean_right', 'right']
    x_positions = np.arange(len(x_labels))

    # 각 result_dict에 대한 평균값
    llama_31_70_mean_values = calculate_mean_values(llama_31_70_result_dict)
    llama_31_8_mean_values = calculate_mean_values(llama_31_8_result_dict)
    llama_3_70_mean_values = calculate_mean_values(llama_3_70_result_dict)
    llama_3_8_mean_values = calculate_mean_values(llama_3_8_result_dict)

    # 그래프 그리기
    plt.figure(figsize=(6, 6), dpi=300)  # 해상도 향상을 위해 크기와 DPI 증가

    # 폰트 설정
    # SF Pro 폰트를 찾을 수 없으므로 시스템 기본 폰트 사용
    plt.rcParams['font.family'] = 'sans-serif'  # 범용적인 sans-serif 폰트 사용
    plt.rcParams['font.size'] = 12  # 기본 글씨 크기 증가

    # 각 모델의 평균값 플롯
    plt.plot(x_positions, [llama_31_70_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#1F4E79', label='Llama-3.1-70B')
    plt.plot(x_positions, [llama_31_8_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#6BAED6', label='Llama-3.1-8B')
    plt.plot(x_positions, [llama_3_70_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#5E3C99', label='Llama-3-70B')
    plt.plot(x_positions, [llama_3_8_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#B39DDB', label='Llama-3-8B')

    # Reference Line: Bias Class가 left -> right일 때 -2 -> 2
    biased_line = np.linspace(-2, 2, len(x_labels))
    plt.plot(x_positions, biased_line, linestyle='--', color='#2F2F2F')
    plt.text(x_positions[3] - 0.3, biased_line[3] + 0.7, "Reference Line - Biased", ha='center', va='bottom', fontsize=10, color='#2F2F2F')

    # Reference Line: 모든 때 0
    plt.axhline(0, color='#A8A8A8', linestyle='--')
    plt.text(x_positions[3] + 0.3, -0.3, "Reference Line - Unbiased", ha='center', va='bottom', fontsize=10, color='#A8A8A8')

    plt.xticks(x_positions, x_labels, fontsize=10)
    plt.xlabel('Bias Class', fontsize=11)
    plt.ylabel('Answer Numeric Value', fontsize=11)
    # plt.title('Bias Class별 Answer Numeric 값 분포')
    plt.ylim(-2, 2)  # y축 척도 고정
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()  # 여백 자동 조정
    plt.savefig("./analyze_result/size_difference_llama.png", dpi=300, bbox_inches='tight')
    
    
def analyze_model_series_qwen():
    qwen_25_log_dir = "../../logs/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias/20250219_145141/result_4400.csv"
    qwen_2_log_dir = "../../logs/allsides/Qwen/Qwen2-72B-Instruct/random_medias/20250309_025048/result_4499.csv"
    qwen_15_log_dir = "../../logs/allsides/Qwen/Qwen1.5-72B-Chat/random_medias/20250310_124421/result_4499.csv"
    qwen_1_log_dir = "../../logs/allsides/Qwen/Qwen-72B-Chat/random_medias/20250410_015336/result_4499.csv"
    
    qwen_25_result_dict = load_log_file(qwen_25_log_dir)
    qwen_2_result_dict = load_log_file(qwen_2_log_dir)
    qwen_15_result_dict = load_log_file(qwen_15_log_dir)
    qwen_1_result_dict = load_log_file(qwen_1_log_dir)
    
    # 데이터 준비
    x_labels = ['left', 'lean_left', 'center', 'lean_right', 'right']
    x_positions = np.arange(len(x_labels))
    
    # 각 result_dict에 대한 평균값
    qwen_25_mean_values = calculate_mean_values(qwen_25_result_dict)
    qwen_2_mean_values = calculate_mean_values(qwen_2_result_dict)
    qwen_15_mean_values = calculate_mean_values(qwen_15_result_dict)
    qwen_1_mean_values = calculate_mean_values(qwen_1_result_dict)
    
    # 그래프 그리기
    plt.figure(figsize=(6, 6), dpi=300)  # 해상도 향상을 위해 크기와 DPI 증가

    # 폰트 설정
    # SF Pro 폰트를 찾을 수 없으므로 시스템 기본 폰트 사용
    plt.rcParams['font.family'] = 'sans-serif'  # 범용적인 sans-serif 폰트 사용
    plt.rcParams['font.size'] = 12  # 기본 글씨 크기 증가

    # 각 모델의 평균값 플롯
    plt.plot(x_positions, [qwen_25_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#E76F51', label='Qwen-2.5-72B')
    plt.plot(x_positions, [qwen_2_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#CF6149', label='Qwen-2-72B')
    plt.plot(x_positions, [qwen_15_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#B55540', label='Qwen-1.5-72B')
    plt.plot(x_positions, [qwen_1_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#994637', label='Qwen-1-72B')

    # Reference Line: Bias Class가 left -> right일 때 -2 -> 2
    biased_line = np.linspace(-2, 2, len(x_labels))
    plt.plot(x_positions, biased_line, linestyle='--', color='#2F2F2F')
    plt.text(x_positions[3] - 0.3, biased_line[3] + 0.7, "Reference Line - Biased", ha='center', va='bottom', fontsize=10, color='#2F2F2F')

    # Reference Line: 모든 때 0
    plt.axhline(0, color='#A8A8A8', linestyle='--')
    plt.text(x_positions[3] + 0.3, -0.3, "Reference Line - Unbiased", ha='center', va='bottom', fontsize=10, color='#A8A8A8')

    plt.xticks(x_positions, x_labels, fontsize=10)
    plt.xlabel('Bias Class', fontsize=11)
    plt.ylabel('Answer Numeric Value', fontsize=11)
    # plt.title('Bias Class별 Answer Numeric 값 분포')
    plt.ylim(-2, 2)  # y축 척도 고정
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()  # 여백 자동 조정
    plt.savefig("./analyze_result/series_difference_qwen.png", dpi=300, bbox_inches='tight')
    
    
def analyze_model_series_llama():
    llama_33_log_dir = "../../logs/allsides/meta-llama/Llama-3.3-70B-Instruct/random_medias/20250219_145234/result_4400.csv"
    llama_31_log_dir = "../../logs/allsides/meta-llama/Llama-3.1-70B-Instruct/random_medias/20250310_223517/result_4499.csv"
    llama_3_log_dir = "../../logs/allsides/meta-llama/Meta-Llama-3-70B-Instruct/random_medias/20250311_150159/result_4499.csv"
    
    llama_33_result_dict = load_log_file(llama_33_log_dir)
    llama_31_result_dict = load_log_file(llama_31_log_dir)
    llama_3_result_dict = load_log_file(llama_3_log_dir)
    
    # 데이터 준비
    x_labels = ['left', 'lean_left', 'center', 'lean_right', 'right']
    x_positions = np.arange(len(x_labels))
    
    # 각 result_dict에 대한 평균값
    llama_33_mean_values = calculate_mean_values(llama_33_result_dict)
    llama_31_mean_values = calculate_mean_values(llama_31_result_dict)
    llama_3_mean_values = calculate_mean_values(llama_3_result_dict)
    
    # 그래프 그리기
    plt.figure(figsize=(6, 6), dpi=300)  # 해상도 향상을 위해 크기와 DPI 증가

    # SF Pro 폰트를 찾을 수 없으므로 시스템 기본 폰트 사용
    plt.rcParams['font.family'] = 'sans-serif'  # 범용적인 sans-serif 폰트 사용
    plt.rcParams['font.size'] = 12  # 기본 글씨 크기 증가

    # 각 모델의 평균값 플롯
    plt.plot(x_positions, [llama_33_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#457B9D', label='Llama-3.3-70B')
    plt.plot(x_positions, [llama_31_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#1F4E79', label='Llama-3.1-70B')
    plt.plot(x_positions, [llama_3_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#142F43', label='Llama-3-70B')

    # Reference Line: Bias Class가 left -> right일 때 -2 -> 2
    biased_line = np.linspace(-2, 2, len(x_labels))
    plt.plot(x_positions, biased_line, linestyle='--', color='#2F2F2F')
    plt.text(x_positions[3] - 0.3, biased_line[3] + 0.7, "Reference Line - Biased", ha='center', va='bottom', fontsize=10, color='#2F2F2F')

    # Reference Line: 모든 때 0
    plt.axhline(0, color='#A8A8A8', linestyle='--')
    plt.text(x_positions[3] + 0.3, -0.3, "Reference Line - Unbiased", ha='center', va='bottom', fontsize=10, color='#A8A8A8')

    plt.xticks(x_positions, x_labels, fontsize=10)
    plt.xlabel('Bias Class', fontsize=11)
    plt.ylabel('Answer Numeric Value', fontsize=11)
    # plt.title('Bias Class별 Answer Numeric 값 분포')
    plt.ylim(-2, 2)  # y축 척도 고정
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()  # 여백 자동 조정
    plt.savefig("./analyze_result/series_difference_llama.png", dpi=300, bbox_inches='tight')
    

def analyze_impact_of_it_qwen_large():
    qwen_70_log_dir = "../../logs/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias/20250219_145141/result_4400.csv"
    qwen_70_non_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-72B/random_medias/20250227_115730/result_4499.csv"
    qwen_32_log_dir = "../../logs/allsides/Qwen/Qwen2.5-32B-Instruct/random_medias/20250225_190144/result_4499.csv"
    qwen_32_non_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-32B/random_medias/20250228_000829/result_4499.csv"
    qwen_14_log_dir = "../../logs/allsides/Qwen/Qwen2.5-14B-Instruct/random_medias/20250225_190224/result_4499.csv"
    qwen_14_non_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-14B/random_medias/20250228_000843/result_4499.csv"
    
    qwen_70_result_dict = load_log_file(qwen_70_log_dir)
    qwen_70_non_it_result_dict = load_log_file(qwen_70_non_it_log_dir)
    qwen_32_result_dict = load_log_file(qwen_32_log_dir)
    qwen_32_non_it_result_dict = load_log_file(qwen_32_non_it_log_dir)
    qwen_14_result_dict = load_log_file(qwen_14_log_dir)
    qwen_14_non_it_result_dict = load_log_file(qwen_14_non_it_log_dir)

    # 데이터 준비
    x_labels = ['left', 'lean_left', 'center', 'lean_right', 'right']
    x_positions = np.arange(len(x_labels))

    # 각 result_dict에 대한 평균값
    qwen_70_mean_values = calculate_mean_values(qwen_70_result_dict)
    qwen_70_non_it_mean_values = calculate_mean_values(qwen_70_non_it_result_dict)
    qwen_32_mean_values = calculate_mean_values(qwen_32_result_dict)
    qwen_32_non_it_mean_values = calculate_mean_values(qwen_32_non_it_result_dict)
    qwen_14_mean_values = calculate_mean_values(qwen_14_result_dict)
    qwen_14_non_it_mean_values = calculate_mean_values(qwen_14_non_it_result_dict)

    # 그래프 그리기
    plt.figure(figsize=(6, 6), dpi=300)  # 해상도 향상을 위해 크기와 DPI 증가

    # 폰트 설정
    # SF Pro 폰트를 찾을 수 없으므로 시스템 기본 폰트 사용
    plt.rcParams['font.family'] = 'sans-serif'  # 범용적인 sans-serif 폰트 사용
    plt.rcParams['font.size'] = 12  # 기본 글씨 크기 증가

    # 각 모델의 평균값 플롯
    plt.plot(x_positions, [qwen_70_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#E76F51', label='Qwen-72B')
    plt.plot(x_positions, [qwen_70_non_it_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, linestyle='--', color='#E76F51', label='Qwen-72B-Non-IT')
    plt.plot(x_positions, [qwen_32_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#EA8D6E', label='Qwen-32B')
    plt.plot(x_positions, [qwen_32_non_it_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, linestyle='--', color='#EA8D6E', label='Qwen-32B-Non-IT')
    plt.plot(x_positions, [qwen_14_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#EDAA8C', label='Qwen-14B')
    plt.plot(x_positions, [qwen_14_non_it_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, linestyle='--', color='#EDAA8C', label='Qwen-14B-Non-IT')

    # Reference Line: Bias Class가 left -> right일 때 -2 -> 2
    biased_line = np.linspace(-2, 2, len(x_labels))
    plt.plot(x_positions, biased_line, linestyle='--', color='#2F2F2F')
    plt.text(x_positions[3] - 0.3, biased_line[3] + 0.7, "Reference Line - Biased", ha='center', va='bottom', fontsize=10, color='#2F2F2F')

    # Reference Line: 모든 때 0
    plt.axhline(0, color='#A8A8A8', linestyle='--')
    plt.text(x_positions[3] + 0.3, -0.3, "Reference Line - Unbiased", ha='center', va='bottom', fontsize=10, color='#A8A8A8')

    plt.xticks(x_positions, x_labels, fontsize=10)
    plt.xlabel('Bias Class', fontsize=11)
    plt.ylabel('Answer Numeric Value', fontsize=11)
    # plt.title('Bias Class별 Answer Numeric 값 분포')
    plt.ylim(-2, 2)  # y축 척도 고정
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()  # 여백 자동 조정
    plt.savefig("./analyze_result/impact_of_it_qwen_large.png", dpi=300, bbox_inches='tight')

    
def analyze_impact_of_it_qwen_small():
    qwen_7_log_dir = "../../logs/allsides/Qwen/Qwen2.5-7B-Instruct/random_medias/20250225_194956/result_4499.csv"
    qwen_7_non_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-7B/random_medias/20250304_163144/result_4499.csv"
    qwen_3_log_dir = "../../logs/allsides/Qwen/Qwen2.5-3B-Instruct/random_medias/20250226_175204/result_4499.csv"
    qwen_3_non_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-3B/random_medias/20250304_163155/result_4499.csv"
    qwen_15_log_dir = "../../logs/allsides/Qwen/Qwen2.5-1.5B-Instruct/random_medias/20250226_175241/result_4499.csv"
    qwen_15_non_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-1.5B/random_medias/20250304_163208/result_4499.csv"
    qwen_05_log_dir = "../../logs/allsides/Qwen/Qwen2.5-0.5B-Instruct/random_medias/20250226_232257/result_4499.csv"
    qwen_05_non_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-0.5B/random_medias/20250309_024522/result_4499.csv"
  
    qwen_7_result_dict = load_log_file(qwen_7_log_dir)
    qwen_7_non_it_result_dict = load_log_file(qwen_7_non_it_log_dir)
    qwen_3_result_dict = load_log_file(qwen_3_log_dir)
    qwen_3_non_it_result_dict = load_log_file(qwen_3_non_it_log_dir)
    qwen_15_result_dict = load_log_file(qwen_15_log_dir)
    qwen_15_non_it_result_dict = load_log_file(qwen_15_non_it_log_dir)
    qwen_05_result_dict = load_log_file(qwen_05_log_dir)
    qwen_05_non_it_result_dict = load_log_file(qwen_05_non_it_log_dir)
    
    # 데이터 준비
    x_labels = ['left', 'lean_left', 'center', 'lean_right', 'right']
    x_positions = np.arange(len(x_labels))
    
    # 각 result_dict에 대한 평균값
    qwen_7_mean_values = calculate_mean_values(qwen_7_result_dict)
    qwen_7_non_it_mean_values = calculate_mean_values(qwen_7_non_it_result_dict)
    qwen_3_mean_values = calculate_mean_values(qwen_3_result_dict)
    qwen_3_non_it_mean_values = calculate_mean_values(qwen_3_non_it_result_dict)
    qwen_15_mean_values = calculate_mean_values(qwen_15_result_dict)
    qwen_15_non_it_mean_values = calculate_mean_values(qwen_15_non_it_result_dict)
    qwen_05_mean_values = calculate_mean_values(qwen_05_result_dict)
    qwen_05_non_it_mean_values = calculate_mean_values(qwen_05_non_it_result_dict)
    
    # 그래프 그리기
    plt.figure(figsize=(6, 6), dpi=300)  # 해상도 향상을 위해 크기와 DPI 증가
    
    plt.plot(x_positions, [qwen_7_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#F0C7AA', label='Qwen-7B')
    plt.plot(x_positions, [qwen_7_non_it_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, linestyle='--', color='#F0C7AA', label='Qwen-7B-Non-IT')
    plt.plot(x_positions, [qwen_3_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#F3E3C8', label='Qwen-3B')
    plt.plot(x_positions, [qwen_3_non_it_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, linestyle='--', color='#F3E3C8', label='Qwen-3B-Non-IT')
    plt.plot(x_positions, [qwen_15_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#F6F0E0', label='Qwen-1.5B')
    plt.plot(x_positions, [qwen_15_non_it_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, linestyle='--', color='#F6F0E0', label='Qwen-1.5B-Non-IT')
    plt.plot(x_positions, [qwen_05_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#FCFBF7', label='Qwen-0.5B')
    plt.plot(x_positions, [qwen_05_non_it_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, linestyle='--', color='#FCFBF7', label='Qwen-0.5B-Non-IT')

    # Reference Line: Bias Class가 left -> right일 때 -2 -> 2
    biased_line = np.linspace(-2, 2, len(x_labels))
    plt.plot(x_positions, biased_line, linestyle='--', color='#2F2F2F')
    plt.text(x_positions[3] - 0.3, biased_line[3] + 0.7, "Reference Line - Biased", ha='center', va='bottom', fontsize=10, color='#2F2F2F')

    # Reference Line: 모든 때 0
    plt.axhline(0, color='#A8A8A8', linestyle='--')
    plt.text(x_positions[3] + 0.3, -0.3, "Reference Line - Unbiased", ha='center', va='bottom', fontsize=10, color='#A8A8A8')

    plt.xticks(x_positions, x_labels, fontsize=10)
    plt.xlabel('Bias Class', fontsize=11)
    plt.ylabel('Answer Numeric Value', fontsize=11)
    # plt.title('Bias Class별 Answer Numeric 값 분포')
    plt.ylim(-2, 2)  # y축 척도 고정
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()  # 여백 자동 조정
    plt.savefig("./analyze_result/impact_of_it_qwen_small.png", dpi=300, bbox_inches='tight')
    

def analyze_impact_of_it_llama_and_mistral():
    llama_31_log_dir = "../../logs/allsides/meta-llama/Llama-3.1-70B-Instruct/random_medias/20250310_223517/result_4499.csv"
    llama_31_non_it_log_dir = "../../logs/allsides/meta-llama/Llama-3.1-70B/random_medias/allsides/20250504_142806/result_2550.csv"
    mistral_log_dir = "../../logs/allsides/mistralai/Mistral-Small-24B-Instruct-2501/random_medias/20250221_124715/result_4499.csv"
    mistral_non_it_log_dir = "../../logs/allsides/mistralai/Mistral-Small-24B-Base-2501/random_medias/20250309_024825/result_4499.csv"

    llama_31_result_dict = load_log_file(llama_31_log_dir)
    llama_31_non_it_result_dict = load_log_file(llama_31_non_it_log_dir)
    mistral_result_dict = load_log_file(mistral_log_dir)
    mistral_non_it_result_dict = load_log_file(mistral_non_it_log_dir)
    
    # 데이터 준비
    x_labels = ['left', 'lean_left', 'center', 'lean_right', 'right']
    x_positions = np.arange(len(x_labels))
    
    # 각 result_dict에 대한 평균값
    llama_31_mean_values = calculate_mean_values(llama_31_result_dict)
    llama_31_non_it_mean_values = calculate_mean_values(llama_31_non_it_result_dict)
    mistral_mean_values = calculate_mean_values(mistral_result_dict)
    mistral_non_it_mean_values = calculate_mean_values(mistral_non_it_result_dict)
    
    # 그래프 그리기
    plt.figure(figsize=(6, 6), dpi=300)  # 해상도 향상을 위해 크기와 DPI 증가
    
    plt.plot(x_positions, [llama_31_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#1F4E79', label='Llama-3.1-70B')
    plt.plot(x_positions, [llama_31_non_it_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, linestyle='--', color='#1F4E79', label='Llama-3.1-70B-Non-IT')
    plt.plot(x_positions, [mistral_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#E9C46A', label='Mistral-Small-24B')
    plt.plot(x_positions, [mistral_non_it_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, linestyle='--', color='#E9C46A', label='Mistral-Small-24B-Non-IT')
    
    # Reference Line: Bias Class가 left -> right일 때 -2 -> 2
    biased_line = np.linspace(-2, 2, len(x_labels))
    plt.plot(x_positions, biased_line, linestyle='--', color='#2F2F2F')
    plt.text(x_positions[3] - 0.3, biased_line[3] + 0.7, "Reference Line - Biased", ha='center', va='bottom', fontsize=10, color='#2F2F2F')

    # Reference Line: 모든 때 0
    plt.axhline(0, color='#A8A8A8', linestyle='--')
    plt.text(x_positions[3] + 0.3, -0.3, "Reference Line - Unbiased", ha='center', va='bottom', fontsize=10, color='#A8A8A8')

    plt.xticks(x_positions, x_labels, fontsize=10)
    plt.xlabel('Bias Class', fontsize=11)
    plt.ylabel('Answer Numeric Value', fontsize=11)
    plt.ylim(-2, 2)  # y축 척도 고정
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()  # 여백 자동 조정
    plt.savefig("./analyze_result/impact_of_it_llama_and_mistral.png", dpi=300, bbox_inches='tight')
    

def analyze_impact_of_reasoning():
    qwen_32_log_dir = "../../logs/allsides/Qwen/Qwen2.5-32B-Instruct/random_medias/20250225_190144/result_4499.csv"
    qwq_log_dir = "../../logs/allsides/Qwen/QwQ-32B/random_medias_for_summarization/20250421_003803/result_449.csv"
    
    qwen_32_result_dict = load_log_file(qwen_32_log_dir)
    qwq_result_dict = load_log_file(qwq_log_dir)
    
    # 데이터 준비
    x_labels = ['left', 'lean_left', 'center', 'lean_right', 'right']
    x_positions = np.arange(len(x_labels))
    
    # 각 result_dict에 대한 평균값
    qwen_32_mean_values = calculate_mean_values(qwen_32_result_dict)
    qwq_mean_values = calculate_mean_values(qwq_result_dict)
    
    # 그래프 그리기
    plt.figure(figsize=(6, 6), dpi=300)  # 해상도 향상을 위해 크기와 DPI 증가
    
    plt.plot(x_positions, [qwen_32_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, color='#EA8D6E', label='Qwen-32B')
    plt.plot(x_positions, [qwq_mean_values[label] for label in x_labels], 
            marker='o', linewidth=2, linestyle='-.', color='#EA8D6E', label='QwQ-32B')
    
    # Reference Line: Bias Class가 left -> right일 때 -2 -> 2
    biased_line = np.linspace(-2, 2, len(x_labels))
    plt.plot(x_positions, biased_line, linestyle='--', color='#2F2F2F')
    plt.text(x_positions[3] - 0.3, biased_line[3] + 0.7, "Reference Line - Biased", ha='center', va='bottom', fontsize=10, color='#2F2F2F')

    # Reference Line: 모든 때 0
    plt.axhline(0, color='#A8A8A8', linestyle='--')
    plt.text(x_positions[3] + 0.3, -0.3, "Reference Line - Unbiased", ha='center', va='bottom', fontsize=10, color='#A8A8A8')

    plt.xticks(x_positions, x_labels, fontsize=10)
    plt.xlabel('Bias Class', fontsize=11)
    plt.ylabel('Answer Numeric Value', fontsize=11)
    plt.ylim(-2, 2)  # y축 척도 고정
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()  # 여백 자동 조정
    plt.savefig("./analyze_result/impact_of_reasoning.png", dpi=300, bbox_inches='tight')
    
    
def analyze_synthetic_medias():
    qwen_log_dir = "../../logs/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias/20250219_145141/result_4400.csv"
    mistral_log_dir = "../../logs/allsides/mistralai/Mistral-Small-24B-Instruct-2501/random_medias/20250221_124715/result_4499.csv"
    phi_log_dir = "../../logs/allsides/microsoft/phi-4/random_medias/20250219_145326/result_4400.csv"
    llama_log_dir = "../../logs/allsides/meta-llama/Llama-3.3-70B-Instruct/random_medias/20250219_145234/result_4400.csv"
    gemma_log_dir = "../../logs/allsides/google/gemma-2-27b-it/random_medias/20250410_003055/result_4499.csv"

        
    qwen_syn_log_dir = "../../logs/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias/generated/20250429_093118/result_4499.csv"
    mistral_syn_log_dir = "../../logs/allsides/mistralai/Mistral-Small-24B-Instruct-2501/random_medias/generated/20250429_093221/result_4499.csv"
    phi_syn_log_dir = "../../logs/allsides/microsoft/phi-4/random_medias/generated/20250502_151251/result_4499.csv"
    llama_syn_log_dir = "../../logs/allsides/meta-llama/Llama-3.3-70B-Instruct/random_medias/generated/20250430_155555/result_4499.csv"
    gemma_syn_log_dir = "../../logs/allsides/google/gemma-2-27b-it/random_medias/generated/20250429_093511/result_4499.csv"

    qwen_syn_result_dict = load_log_file_gen(qwen_log_dir, qwen_syn_log_dir)
    mistral_syn_result_dict = load_log_file_gen(mistral_log_dir, mistral_syn_log_dir)
    phi_syn_result_dict = load_log_file_gen(phi_log_dir, phi_syn_log_dir)
    llama_syn_result_dict = load_log_file_gen(llama_log_dir, llama_syn_log_dir)
    gemma_syn_result_dict = load_log_file_gen(gemma_log_dir, gemma_syn_log_dir)
    
    x_labels = ['left', 'lean_left', 'center', 'lean_right', 'right']
    x_labels_generated = ['generated_left', 'generated_right', 'formulated_left', 'formulated_right']
    x_positions = np.arange(len(x_labels))
    x_positions_generated = np.arange(len(x_labels_generated))

    # 각 result_dict에 대한 평균값
    qwen_syn_mean_values = calculate_mean_values_generated(qwen_syn_result_dict)
    mistral_syn_mean_values = calculate_mean_values_generated(mistral_syn_result_dict)
    phi_syn_mean_values = calculate_mean_values_generated(phi_syn_result_dict)
    llama_syn_mean_values = calculate_mean_values_generated(llama_syn_result_dict)
    gemma_syn_mean_values = calculate_mean_values_generated(gemma_syn_result_dict)
    
    # 그래프 그리기
    plt.figure(figsize=(6, 6), dpi=300)  # 해상도 향상을 위해 크기와 DPI 증가

    # 폰트 설정
    # SF Pro 폰트를 찾을 수 없으므로 시스템 기본 폰트 사용
    plt.rcParams['font.family'] = 'sans-serif'  # 범용적인 sans-serif 폰트 사용
    plt.rcParams['font.size'] = 12  # 기본 글씨 크기 증가

    # 각 모델의 평균값 플롯
    plt.plot(x_positions_generated, [qwen_syn_mean_values[label] for label in x_labels_generated], 
            marker='o', linewidth=2, color='#E76F51', label='Qwen')
    plt.plot(x_positions_generated, [mistral_syn_mean_values[label] for label in x_labels_generated], 
            marker='o', linewidth=2, color='#E9C46A', label='Mistral')
    plt.plot(x_positions_generated, [phi_syn_mean_values[label] for label in x_labels_generated], 
            marker='o', linewidth=2, color='#2A9D8F', label='Phi')
    plt.plot(x_positions_generated, [llama_syn_mean_values[label] for label in x_labels_generated], 
            marker='o', linewidth=2, color='#457B9D', label='Llama')
    plt.plot(x_positions_generated, [gemma_syn_mean_values[label] for label in x_labels_generated], 
            marker='o', linewidth=2, color='#8D6AA9', label='Gemma')

    # Reference Line: Bias Class가 left -> right일 때 -2 -> 2
    biased_line = np.linspace(-2, 2, len(x_labels_generated))
    plt.plot(x_positions_generated, biased_line, linestyle='--', color='#2F2F2F')
    plt.text(x_positions_generated[1] - 0.3, biased_line[1] + 0.7, "Reference Line - Biased", ha='center', va='bottom', fontsize=10, color='#2F2F2F')

    # Reference Line: 모든 때 0
    plt.axhline(0, color='#A8A8A8', linestyle='--')
    plt.text(x_positions_generated[1] + 0.3, -0.3, "Reference Line - Unbiased", ha='center', va='bottom', fontsize=10, color='#A8A8A8')

    plt.xticks(x_positions_generated, x_labels_generated, fontsize=10)
    plt.xlabel('Bias Class', fontsize=11)
    plt.ylabel('Answer Numeric Value', fontsize=11)
    # plt.title('Bias Class별 Answer Numeric 값 분포')
    plt.ylim(-2, 2)  # y축 척도 고정
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()  # 여백 자동 조정
    plt.savefig("./analyze_result/syn_result.png", dpi=300, bbox_inches='tight')
    
    
def analyze_sips():
    # allsides
    qwen2_5_72b_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias/20250219_145141/result_4400.csv"
    mistral_small_24b_it_log_dir = "../../logs/allsides/mistralai/Mistral-Small-24B-Instruct-2501/random_medias/20250221_124715/result_4499.csv"
    phi_4_it_log_dir = "../../logs/allsides/microsoft/phi-4/random_medias/20250219_145326/result_4400.csv"
    llama_3_3_70b_it_log_dir = "../../logs/allsides/meta-llama/Llama-3.3-70B-Instruct/random_medias/20250219_145234/result_4400.csv"
    gemma_2_27b_it_log_dir = "../../logs/allsides/google/gemma-2-27b-it/random_medias/20250410_003055/result_4499.csv"
    
    # hyperpartisan 
    qwen2_5_72b_it_hp_log_dir = "../../logs/hyperpartisan/Qwen/Qwen2.5-72B-Instruct/hyperpartisan/20250408_232523/result_1272.csv"
    mistral_small_24b_it_hp_log_dir = "../../logs/hyperpartisan/mistralai/Mistral-Small-24B-Instruct-2501/hyperpartisan/20250408_232519/result_1272.csv"
    phi_4_it_hp_log_dir = "../../logs/hyperpartisan/microsoft/phi-4/hyperpartisan/20250408_232529/result_1272.csv"
    llama_3_3_70b_it_hp_log_dir = "../../logs/hyperpartisan/meta-llama/Llama-3.3-70B-Instruct/hyperpartisan/20250409_033417/result_1272.csv"
    gemma_2_27b_it_hp_log_dir = "../../logs/hyperpartisan/google/gemma-2-27b-it/hyperpartisan/20250411_200354/result_1272.csv"
    
    gpt_4_1_it_log_dir = "../../logs/allsides/gpt-4.1/random_medias/allsides/20250512_160756/result_4499.csv"
    gpt_4_1_mini_it_log_dir = "../../logs/allsides/gpt-4.1-mini/random_medias/allsides/20250512_160909/result_4499.csv"
    gpt_4_1_nano_it_log_dir = "../../logs/allsides/gpt-4.1-nano/random_medias/allsides/20250512_160924/result_4499.csv"

    gpt_4_1_it_hp_log_dir = "../../logs/hyperpartisan/gpt-4.1/hyperpartisan/allsides/20250513_090710/result_1272.csv"
    gpt_4_1_mini_it_hp_log_dir = "../../logs/hyperpartisan/gpt-4.1-mini/hyperpartisan/allsides/20250513_090747/result_1272.csv"
    gpt_4_1_nano_it_hp_log_dir = "../../logs/hyperpartisan/gpt-4.1-nano/hyperpartisan/allsides/20250513_090803/result_1272.csv"


    # qwen_size    
    qwen2_5_72b_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias/20250219_145141/result_4400.csv"
    qwen2_5_32b_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-32B-Instruct/random_medias/20250225_190144/result_4499.csv"
    qwen2_5_14b_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-14B-Instruct/random_medias/20250225_190224/result_4499.csv"
    qwen2_5_7b_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-7B-Instruct/random_medias/20250225_194956/result_4499.csv"
    qwen2_5_3b_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-3B-Instruct/random_medias/20250226_175204/result_4499.csv"
    qwen2_5_15b_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-1.5B-Instruct/random_medias/20250226_175241/result_4499.csv"
    qwen2_5_05b_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-0.5B-Instruct/random_medias/20250226_232257/result_4499.csv"
    
    # llama_size
    llama_3_1_70b_it_log_dir = "../../logs/allsides/meta-llama/Llama-3.1-70B-Instruct/random_medias/20250310_223517/result_4499.csv"
    llama_3_1_8b_it_log_dir = "../../logs/allsides/meta-llama/Llama-3.1-8B-Instruct/random_medias/20250411_195339/result_4499.csv"
    llama_3_70b_it_log_dir = "../../logs/allsides/meta-llama/Meta-Llama-3-70B-Instruct/random_medias/20250311_150159/result_4499.csv"
    llama_3_8b_it_log_dir = "../../logs/allsides/meta-llama/Meta-Llama-3-8B-Instruct/random_medias/20250412_134400/result_4499.csv"
    
    # qwen series
    qwen2_5_72b_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias/20250219_145141/result_4400.csv"
    qwen2_72b_it_log_dir = "../../logs/allsides/Qwen/Qwen2-72B-Instruct/random_medias/20250309_025048/result_4499.csv"
    qwen1_5_72b_it_log_dir = "../../logs/allsides/Qwen/Qwen1.5-72B-Chat/random_medias/20250310_124421/result_4499.csv"
    qwen_72b_it_log_dir = "../../logs/allsides/Qwen/Qwen-72B-Chat/random_medias/20250410_015336/result_4499.csv"
    
    # llama series
    llama_3_3_70b_it_log_dir = "../../logs/allsides/meta-llama/Llama-3.3-70B-Instruct/random_medias/20250219_145234/result_4400.csv"
    llama_3_1_70b_it_log_dir = "../../logs/allsides/meta-llama/Llama-3.1-70B-Instruct/random_medias/20250310_223517/result_4499.csv"
    llama_3_70b_it_log_dir = "../../logs/allsides/meta-llama/Meta-Llama-3-70B-Instruct/random_medias/20250311_150159/result_4499.csv"
    
    # it vs non it
    qwen2_5_72b_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias/20250219_145141/result_4400.csv"
    qwen2_5_72b_non_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-72B/random_medias/20250227_115730/result_4499.csv"
    qwen2_5_32b_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-32B-Instruct/random_medias/20250225_190144/result_4499.csv"
    qwen2_5_32b_non_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-32B/random_medias/20250228_000829/result_4499.csv"
    qwen2_5_14b_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-14B-Instruct/random_medias/20250225_190224/result_4499.csv"
    qwen2_5_14b_non_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-14B/random_medias/20250228_000843/result_4499.csv"
    qwen2_5_7b_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-7B-Instruct/random_medias/20250225_194956/result_4499.csv"
    qwen2_5_7b_non_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-7B/random_medias/20250304_163144/result_4499.csv"
    qwen2_5_3b_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-3B-Instruct/random_medias/20250226_175204/result_4499.csv"
    qwen2_5_3b_non_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-3B/random_medias/20250304_163155/result_4499.csv"
    qwen2_5_15b_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-1.5B-Instruct/random_medias/20250226_175241/result_4499.csv"
    qwen2_5_15b_non_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-1.5B/random_medias/20250304_163208/result_4499.csv"
    qwen2_5_05b_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-0.5B-Instruct/random_medias/20250226_232257/result_4499.csv"
    qwen2_5_05b_non_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-0.5B/random_medias/20250309_024522/result_4499.csv"
    
    llama_3_1_70b_it_log_dir = "../../logs/allsides/meta-llama/Llama-3.1-70B-Instruct/random_medias/20250310_223517/result_4499.csv"
    llama_3_1_70b_non_it_log_dir = "../../logs/allsides/meta-llama/Llama-3.1-70B/random_medias/allsides/20250504_142806/result_2550.csv"
    llama_3_1_8b_it_log_dir = "../../logs/allsides/meta-llama/Llama-3.1-8B-Instruct/random_medias/20250411_195339/result_4499.csv"
    llama_3_1_8b_non_it_log_dir = "../../logs/allsides/meta-llama/Llama-3.1-8B/random_medias/allsides/20250511_151624/result_4499.csv"
    mistral_small_24b_it_log_dir = "../../logs/allsides/mistralai/Mistral-Small-24B-Instruct-2501/random_medias/20250221_124715/result_4499.csv"
    mistral_small_24b_non_it_log_dir = "../../logs/allsides/mistralai/Mistral-Small-24B-Base-2501/random_medias/20250309_024825/result_4499.csv"
    
    # reasoning
    qwen2_5_32b_it_log_dir = "../../logs/allsides/Qwen/Qwen2.5-32B-Instruct/random_medias/20250225_190144/result_4499.csv"
    qwq_32b_it_log_dir = "../../logs/allsides/Qwen/QwQ-32B/random_medias_for_summarization/20250421_003803/result_449.csv"
    
    # synthetic
    qwen2_5_72b_it_syn_log_dir = "../../logs/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias/generated/20250429_093118/result_4499.csv"
    mistral_small_24b_it_syn_log_dir = "../../logs/allsides/mistralai/Mistral-Small-24B-Instruct-2501/random_medias/generated/20250429_093221/result_4499.csv"
    phi_4_it_syn_log_dir = "../../logs/allsides/microsoft/phi-4/random_medias/generated/20250502_151251/result_4499.csv"
    llama_3_3_70b_it_syn_log_dir = "../../logs/allsides/meta-llama/Llama-3.3-70B-Instruct/random_medias/generated/20250430_155555/result_4499.csv"
    gemma_2_27b_it_syn_log_dir = "../../logs/allsides/google/gemma-2-27b-it/random_medias/generated/20250429_093511/result_4499.csv"
    
    # ------------------------------------------------------------
    
    qwen_2_5_72b_it_result_dict = load_log_file(qwen2_5_72b_it_log_dir)
    mistral_small_24b_it_result_dict = load_log_file(mistral_small_24b_it_log_dir)
    phi_4_it_result_dict = load_log_file(phi_4_it_log_dir)
    llama_3_3_70b_it_result_dict = load_log_file(llama_3_3_70b_it_log_dir)
    gemma_2_27b_it_result_dict = load_log_file(gemma_2_27b_it_log_dir)
    
    qwen_2_5_72b_it_hp_result_dict = load_log_file(qwen2_5_72b_it_hp_log_dir)
    mistral_small_24b_it_hp_result_dict = load_log_file(mistral_small_24b_it_hp_log_dir)
    phi_4_it_hp_result_dict = load_log_file(phi_4_it_hp_log_dir)
    llama_3_3_70b_it_hp_result_dict = load_log_file(llama_3_3_70b_it_hp_log_dir)
    gemma_2_27b_it_hp_result_dict = load_log_file(gemma_2_27b_it_hp_log_dir)
    
    gpt_4_1_it_result_dict = load_log_file(gpt_4_1_it_log_dir)
    gpt_4_1_mini_it_result_dict = load_log_file(gpt_4_1_mini_it_log_dir)
    gpt_4_1_nano_it_result_dict = load_log_file(gpt_4_1_nano_it_log_dir)
    
    gpt_4_1_it_hp_result_dict = load_log_file(gpt_4_1_it_hp_log_dir)
    gpt_4_1_mini_it_hp_result_dict = load_log_file(gpt_4_1_mini_it_hp_log_dir)
    gpt_4_1_nano_it_hp_result_dict = load_log_file(gpt_4_1_nano_it_hp_log_dir)
    
    qwen_2_5_32b_it_result_dict = load_log_file(qwen2_5_32b_it_log_dir)
    qwen_2_5_14b_it_result_dict = load_log_file(qwen2_5_14b_it_log_dir)
    qwen_2_5_7b_it_result_dict = load_log_file(qwen2_5_7b_it_log_dir)
    qwen_2_5_3b_it_result_dict = load_log_file(qwen2_5_3b_it_log_dir)
    qwen_2_5_15b_it_result_dict = load_log_file(qwen2_5_15b_it_log_dir)
    qwen_2_5_05b_it_result_dict = load_log_file(qwen2_5_05b_it_log_dir)
    
    llama_3_1_70b_it_result_dict = load_log_file(llama_3_1_70b_it_log_dir)
    llama_3_1_8b_it_result_dict = load_log_file(llama_3_1_8b_it_log_dir)
    llama_3_70b_it_result_dict = load_log_file(llama_3_70b_it_log_dir)
    llama_3_8b_it_result_dict = load_log_file(llama_3_8b_it_log_dir)
    
    qwen_2_72b_it_result_dict = load_log_file(qwen2_72b_it_log_dir)
    qwen_1_5_72b_it_result_dict = load_log_file(qwen1_5_72b_it_log_dir)
    qwen_72b_it_result_dict = load_log_file(qwen_72b_it_log_dir)
    
    qwen_2_5_72b_non_it_result_dict = load_log_file(qwen2_5_72b_non_it_log_dir)
    qwen_2_5_32b_non_it_result_dict = load_log_file(qwen2_5_32b_non_it_log_dir)
    qwen_2_5_14b_non_it_result_dict = load_log_file(qwen2_5_14b_non_it_log_dir)
    qwen_2_5_7b_non_it_result_dict = load_log_file(qwen2_5_7b_non_it_log_dir)
    qwen_2_5_3b_non_it_result_dict = load_log_file(qwen2_5_3b_non_it_log_dir)
    qwen_2_5_15b_non_it_result_dict = load_log_file(qwen2_5_15b_non_it_log_dir)
    qwen_2_5_05b_non_it_result_dict = load_log_file(qwen2_5_05b_non_it_log_dir)
    
    llama_3_1_70b_non_it_result_dict = load_log_file(llama_3_1_70b_non_it_log_dir)
    llama_3_1_8b_non_it_result_dict = load_log_file(llama_3_1_8b_non_it_log_dir)
    mistral_small_24b_non_it_result_dict = load_log_file(mistral_small_24b_non_it_log_dir)
    
    qwq_32b_it_result_dict = load_log_file(qwq_32b_it_log_dir)
    
#     qwen_syn_result_dict = load_log_file(qwen_syn_log_dir)
#     mistral_syn_result_dict = load_log_file(mistral_syn_log_dir)
#     phi_syn_result_dict = load_log_file(phi_syn_log_dir)
#     llama_syn_result_dict = load_log_file(llama_syn_log_dir)
#     gemma_syn_result_dict = load_log_file(gemma_syn_log_dir)
    
    all_models = {
        "Qwen2.5-72B-IT": qwen_2_5_72b_it_result_dict,
        "Mistral-Small-24B-IT": mistral_small_24b_it_result_dict,
        "Phi-4-IT": phi_4_it_result_dict,
        "Llama-3.3-70B-IT": llama_3_3_70b_it_result_dict,
        "Gemma-2-27B-IT": gemma_2_27b_it_result_dict,
        
        "Qwen2.5-72B-IT-HP": qwen_2_5_72b_it_hp_result_dict,
        "Mistral-Small-24B-IT-HP": mistral_small_24b_it_hp_result_dict,
        "Phi-4-IT-HP": phi_4_it_hp_result_dict,
        "Llama-3.3-70B-IT-HP": llama_3_3_70b_it_hp_result_dict,
        "Gemma-2-27B-IT-HP": gemma_2_27b_it_hp_result_dict,
        
        "GPT-4.1-IT": gpt_4_1_it_result_dict,
        "GPT-4.1-Mini-IT": gpt_4_1_mini_it_result_dict,
        "GPT-4.1-Nano-IT": gpt_4_1_nano_it_result_dict,
        
        "GPT-4.1-IT-HP": gpt_4_1_it_hp_result_dict,
        "GPT-4.1-Mini-IT-HP": gpt_4_1_mini_it_hp_result_dict,
        "GPT-4.1-Nano-IT-HP": gpt_4_1_nano_it_hp_result_dict,
        
        "Qwen2.5-72B-IT": qwen_2_5_72b_it_result_dict,
        "Qwen2.5-32B-IT": qwen_2_5_32b_it_result_dict,
        "Qwen2.5-14B-IT": qwen_2_5_14b_it_result_dict,
        "Qwen2.5-7B-IT": qwen_2_5_7b_it_result_dict,
        "Qwen2.5-3B-IT": qwen_2_5_3b_it_result_dict,
        "Qwen2.5-1.5B-IT": qwen_2_5_15b_it_result_dict,
        "Qwen2.5-0.5B-IT": qwen_2_5_05b_it_result_dict,
        
        "Llama-3.1-70B-IT": llama_3_1_70b_it_result_dict,
        "Llama-3.1-8B-IT": llama_3_1_8b_it_result_dict,
        "Llama-3-70B-IT": llama_3_70b_it_result_dict,
        "Llama-3-8B-IT": llama_3_8b_it_result_dict,
        
        "Qwen2.5-72B-IT": qwen_2_5_72b_it_result_dict,
        "Qwen2-72B-IT": qwen_2_72b_it_result_dict,
        "Qwen1.5-72B-IT": qwen_1_5_72b_it_result_dict,
        "Qwen1-72B-IT": qwen_72b_it_result_dict,
        
        "Llama-3.3-70B-IT": llama_3_3_70b_it_result_dict,
        "Llama-3.1-70B-IT": llama_3_1_70b_it_result_dict,
        "Llama-3-70B-IT": llama_3_70b_it_result_dict,
        
        "Qwen2.5-72B-Base": qwen_2_5_72b_non_it_result_dict,
        "Qwen2.5-32B-Base": qwen_2_5_32b_non_it_result_dict,
        "Qwen2.5-14B-Base": qwen_2_5_14b_non_it_result_dict,
        "Qwen2.5-7B-Base": qwen_2_5_7b_non_it_result_dict,
        "Qwen2.5-3B-Base": qwen_2_5_3b_non_it_result_dict,
        "Qwen2.5-1.5B-Base": qwen_2_5_15b_non_it_result_dict,
        "Qwen2.5-0.5B-Base": qwen_2_5_05b_non_it_result_dict,
        
        "Llama-3.1-70B-Base": llama_3_1_70b_non_it_result_dict,
        "Llama-3.1-8B-Base": llama_3_1_8b_non_it_result_dict,
        "Mistral-Small-24B-Base": mistral_small_24b_non_it_result_dict,
        
        "QwQ-32B-IT": qwq_32b_it_result_dict,
    }
    
    # Hyperpartisan 데이터셋에 대한 SIPS 계산 및 저장
    all_sips_df = calculate_sips_for_all_models(all_models, "./analyze_result/sips_all_v3.csv")
    print("All SIPS results saved to './analyze_result/sips_all_v3.csv'") 

    return all_sips_df

def analyze_sips_prompt():
    qwen_2_5_72b_it_prompt_log_dir = "../../logs/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias_for_summarization/allsides/20250519_055459/result_449.csv"
    gemma_2_27b_it_prompt_log_dir = "../../logs/allsides/google/gemma-2-27b-it/random_medias_for_summarization/allsides/20250519_205156/result_449.csv"
    qwen_2_5_72b_it_prompt_result_dict = load_log_file(qwen_2_5_72b_it_prompt_log_dir)
    gemma_2_27b_it_prompt_result_dict = load_log_file(gemma_2_27b_it_prompt_log_dir)
    
    all_models = {
        "Qwen2.5-72B-IT": qwen_2_5_72b_it_prompt_result_dict,
        "Gemma-2-27B-IT": gemma_2_27b_it_prompt_result_dict
    }
    
    all_sips_df = calculate_sips_for_all_models(all_models, "./analyze_result/sips_all_prompt_v2.csv")
    print("All SIPS results saved to './analyze_result/sips_all_prompt_v2.csv'") 

    return all_sips_df

def analyze_sips_prompt_rvt():
    mistral_small_24b_it_log_dir = "../../logs/allsides/mistralai/Mistral-Small-24B-Instruct-2501/random_medias_for_summarization/allsides/20250628_181211/result_449.csv"
    phi_4_it_log_dir = "../../logs/allsides/microsoft/phi-4/random_medias_for_summarization/allsides/20250628_181721/result_449.csv"
    llama_3_3_70b_it_log_dir = "../../logs/allsides/meta-llama/Llama-3.3-70B-Instruct/random_medias_for_summarization/allsides/20250628_181900/result_449.csv"
    gpt_4_1_log_dir = "../../logs/allsides/gpt-4.1/random_medias_for_summarization/allsides/20250628_181110/result_449.csv"
    
    mistral_small_24b_it_log_dir_ori = "../../logs/allsides/mistralai/Mistral-Small-24B-Instruct-2501/random_medias_for_summarization/allsides/20250629_161155/result_449.csv"
    phi_4_it_log_dir_ori = "../../logs/allsides/microsoft/phi-4/random_medias_for_summarization/allsides/20250629_161016/result_449.csv"
    llama_3_3_70b_it_log_dir_ori = "../../logs/allsides/meta-llama/Llama-3.3-70B-Instruct/random_medias_for_summarization/allsides/20250629_161022/result_449.csv"
    gpt_4_1_log_dir_ori = "../../logs/allsides/gpt-4.1/random_medias_for_summarization/allsides/20250629_161037/result_449.csv"
    
    mistral_small_24b_it_prompt_result_dict = load_log_file(mistral_small_24b_it_log_dir)
    phi_4_it_prompt_result_dict = load_log_file(phi_4_it_log_dir)
    llama_3_3_70b_it_prompt_result_dict = load_log_file(llama_3_3_70b_it_log_dir)
    gpt_4_1_prompt_result_dict = load_log_file(gpt_4_1_log_dir)
    
    mistral_small_24b_it_prompt_result_dict_ori = load_log_file(mistral_small_24b_it_log_dir_ori)
    phi_4_it_prompt_result_dict_ori = load_log_file(phi_4_it_log_dir_ori)
    llama_3_3_70b_it_prompt_result_dict_ori = load_log_file(llama_3_3_70b_it_log_dir_ori)
    gpt_4_1_prompt_result_dict_ori = load_log_file(gpt_4_1_log_dir_ori)
    
    all_models = {
        "Mistral-Small-24B-IT": mistral_small_24b_it_prompt_result_dict,
        "Phi-4-IT": phi_4_it_prompt_result_dict,
        "Llama-3.3-70B-IT": llama_3_3_70b_it_prompt_result_dict,
        "GPT-4.1-IT": gpt_4_1_prompt_result_dict,
        "Mistral-Small-24B-IT-ori": mistral_small_24b_it_prompt_result_dict_ori,
        "Phi-4-IT-ori": phi_4_it_prompt_result_dict_ori,
        "Llama-3.3-70B-IT-ori": llama_3_3_70b_it_prompt_result_dict_ori,
        "GPT-4.1-IT-ori": gpt_4_1_prompt_result_dict_ori
    }
    
    all_sips_df = calculate_sips_for_all_models(all_models, "./analyze_result/sips_all_prompt_rvt_v2.csv")
    print("All SIPS results saved to './analyze_result/sips_all_prompt_rvt_v2.csv'") 

    return all_sips_df

def analyze_summarization_political_bias_only_classification():    
    gemma_summary_dir = "../../logs/summarization/allsides/google/gemma-2-27b-it/random_medias_for_summarization/5/20250503_124817/result_449.csv"
    llama_summary_dir = "../../logs/summarization/allsides/meta-llama/Llama-3.3-70B-Instruct/random_medias_for_summarization/5/20250505_015658/result_449.csv"
    phi_summary_dir = "../../logs/summarization/allsides/microsoft/phi-4/random_medias_for_summarization/5/20250501_160948/result_449.csv"
    mistral_summary_dir = "../../logs/summarization/allsides/mistralai/Mistral-Small-24B-Instruct-2501/random_medias_for_summarization/5/20250423_170608/result_449.csv"
    qwen_summary_dir = "../../logs/summarization/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias_for_summarization/5/20250421_003934/result_449.csv"
    
    gemma_summary_df = pd.read_csv(gemma_summary_dir)
    llama_summary_df = pd.read_csv(llama_summary_dir)
    phi_summary_df = pd.read_csv(phi_summary_dir)
    mistral_summary_df = pd.read_csv(mistral_summary_dir)
    qwen_summary_df = pd.read_csv(qwen_summary_dir)
    
    gemma_id_list = gemma_summary_df['id'].unique().tolist()
    llama_id_list = llama_summary_df['id'].unique().tolist()
    phi_id_list = phi_summary_df['id'].unique().tolist()
    mistral_id_list = mistral_summary_df['id'].unique().tolist()
    qwen_id_list = qwen_summary_df['id'].unique().tolist()
    
    print(len(gemma_id_list))
    print(len(llama_id_list))
    print(len(phi_id_list))
    print(len(mistral_id_list))
    print(len(qwen_id_list))
    
    for id in gemma_id_list:
        if id not in llama_id_list:
            print(id)
        if id not in phi_id_list:
            print(id)
        if id not in mistral_id_list:
            print(id)
        if id not in qwen_id_list:
            print(id)
    
    # 1. 정치적 편향 분류 모델 로드
    from torch import argmax
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.nn.functional import softmax, cosine_similarity
    import torch
    
    # CUDA 2번 디바이스 설정
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("launch/POLITICS")
    political_model = AutoModelForSequenceClassification.from_pretrained("matous-volf/political-leaning-politics").to(device)
    
    # 3. 기존 SBERT 모델도 유지 (비교용)
    sbert = SentenceTransformer("all-mpnet-base-v2", device=device)
    
    def get_political_bias_score(text):
        """텍스트의 정치적 편향 점수를 계산하는 함수"""
        try:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            tokens = {k: v.to(device) for k, v in tokens.items()}  # 토큰을 디바이스로 이동
            with torch.no_grad():
                output = political_model(**tokens)
                logits = output.logits
                probabilities = softmax(logits, dim=1)
                political_leaning = argmax(logits, dim=1).item()
                score = probabilities[0, political_leaning].item()
                return political_leaning, score
        except Exception as e:
            print(f"Error processing text: {e}")
            return 0, 0.5  # 기본값 반환
    
    model_names = ["Qwen", "Mistral", "Gemma", "Llama", "Phi"]
    model_dfs = [qwen_summary_df, mistral_summary_df, gemma_summary_df, llama_summary_df, phi_summary_df]
    
    # 결과를 저장할 리스트
    results = []
    
    for id in gemma_id_list:
        # 각 모델에 대해 분석
        for model_name, model_df in zip(model_names, model_dfs):
            # 각 bias class별 summary 가져오기
            for allsides_class in ["left", "center", "right"]:
                model_summaries = model_df.loc[
                    (model_df['id'] == id) & 
                    (model_df['allsides_class'] == allsides_class)
                ]['answer'].values.tolist()
                
                if len(model_summaries) > 0:
                    # 각 summary 분석
                    political_leanings = []
                    scores = []
                    sbert_similarities = []
                    
                    for summary in model_summaries:
                        # 1. 정치적 편향 점수
                        leaning, score = get_political_bias_score(summary)
                        political_leanings.append(leaning)
                        scores.append(score)

                    # 평균 계산
                    avg_political_leaning = sum(political_leanings) / len(political_leanings)
                    avg_score = sum(scores) / len(scores)
                    
                    # 결과 저장
                    results.append({
                        'content_id': id,
                        'model': model_name,
                        'model_bias_class': allsides_class,
                        'avg_political_leaning': avg_political_leaning,
                        'avg_score': avg_score,
                        'num_summaries': len(model_summaries)
                    })
    
    # DataFrame으로 변환
    results_df = pd.DataFrame(results)
    
    # 5. 시각화
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # 결과 디렉토리 생성
    os.makedirs("./analyze_result/summarization_rvt", exist_ok=True)
    # 결과 저장
    results_df.to_csv("./analyze_result/summarization_rvt/comprehensive_analysis_original.csv", index=False)
    
    # 통계 분석 및 시각화
    print("=== 종합 분석 결과 ===\n")
    
    print("\n[Model Summary 정치적 편향 점수]")
    print("\n모델별 & Bias Class별 평균 및 표준편차:")
    
    # 평균과 표준편차 둘 다 계산
    model_score_mean = results_df.groupby(['model', 'model_bias_class'])[['avg_political_leaning', 'avg_score']].mean()
    model_score_std = results_df.groupby(['model', 'model_bias_class'])[['avg_political_leaning', 'avg_score']].std()
    
    for model in model_names:
        print(f"\n  {model}:")
        for bias_class in ["left", "center", "right"]:
            try:
                avg_leaning = model_score_mean.loc[(model, bias_class), 'avg_political_leaning']
                std_leaning = model_score_std.loc[(model, bias_class), 'avg_political_leaning']
                avg_score = model_score_mean.loc[(model, bias_class), 'avg_score']
                std_score = model_score_std.loc[(model, bias_class), 'avg_score']
                print(f"    {bias_class.upper()}: 정치적 편향 {avg_leaning:.4f} (±{std_leaning:.4f}), 신뢰도 {avg_score:.4f} (±{std_score:.4f})")
            except KeyError:
                print(f"    {bias_class.upper()}: 데이터 없음")
    
    return results_df


def analyze_summarization_political_bias():
    reference_summary_dir = "../../logs/summarization_reference/allsides/gpt-4.1/random_medias_for_summarization_rvt_v2/5/20250630_131941/result_29.csv"
    
    qwen_summary_dir = "../../logs/summarization/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias_for_summarization_rvt_v2/5/20250630_044703/result_29.csv"
    mistral_summary_dir = "../../logs/summarization/allsides/mistralai/Mistral-Small-24B-Instruct-2501/random_medias_for_summarization_rvt_v2/5/20250630_024141/result_29.csv"
    gemma_summary_dir = "../../logs/summarization/allsides/google/gemma-2-27b-it/random_medias_for_summarization_rvt_v2/5/20250630_033205/result_29.csv"
    llama_summary_dir = "../../logs/summarization/allsides/meta-llama/Llama-3.3-70B-Instruct/random_medias_for_summarization_rvt_v2/5/20250630_142519/result_29.csv"
    phi_summary_dir = "../../logs/summarization/allsides/microsoft/phi-4/random_medias_for_summarization_rvt_v2/5/20250630_142558/result_29.csv"
    
    reference_summary_df = pd.read_csv(reference_summary_dir)
    qwen_summary_df = pd.read_csv(qwen_summary_dir)
    mistral_summary_df = pd.read_csv(mistral_summary_dir)
    gemma_summary_df = pd.read_csv(gemma_summary_dir)
    llama_summary_df = pd.read_csv(llama_summary_dir)
    phi_summary_df = pd.read_csv(phi_summary_dir)
    
    content_id_list = reference_summary_df['id'].unique().tolist()
    
    # 1. 정치적 편향 분류 모델 로드
    from torch import argmax
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.nn.functional import softmax, cosine_similarity
    import torch
    
    # CUDA 2번 디바이스 설정
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("launch/POLITICS")
    political_model = AutoModelForSequenceClassification.from_pretrained("matous-volf/political-leaning-politics").to(device)
    
    # 3. 기존 SBERT 모델도 유지 (비교용)
    sbert = SentenceTransformer("all-mpnet-base-v2", device=device)
    
    def get_political_bias_score(text):
        """텍스트의 정치적 편향 점수를 계산하는 함수"""
        try:
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            tokens = {k: v.to(device) for k, v in tokens.items()}  # 토큰을 디바이스로 이동
            with torch.no_grad():
                output = political_model(**tokens)
                logits = output.logits
                probabilities = softmax(logits, dim=1)
                political_leaning = argmax(logits, dim=1).item()
                score = probabilities[0, political_leaning].item()
                return political_leaning, score
        except Exception as e:
            print(f"Error processing text: {e}")
            return 0, 0.5  # 기본값 반환
    
    reference_class_list = ["left", "center", "right"]
    model_names = ["Qwen", "Mistral", "Gemma", "Llama", "Phi"]
    model_dfs = [qwen_summary_df, mistral_summary_df, gemma_summary_df, llama_summary_df, phi_summary_df]
    
    # 결과를 저장할 리스트
    results = []
    results_single = []
    
    for id in content_id_list:
        for reference_class in reference_class_list:
            # Reference summary 가져오기
            reference_summary = reference_summary_df.loc[
                (reference_summary_df['id'] == id) & 
                (reference_summary_df['reference_class'] == reference_class)
            ]['answer'].values[0]
            
            # Reference summary 분석
            ref_political_leaning, ref_score = get_political_bias_score(reference_summary)
            ref_sbert_embedding = sbert.encode(reference_summary)
            
            # 각 모델에 대해 분석
            for model_name, model_df in zip(model_names, model_dfs):
                # 각 bias class별 summary 가져오기
                for allsides_class in ["left", "center", "right"]:
                    model_summaries = model_df.loc[
                        (model_df['id'] == id) & 
                        (model_df['allsides_class'] == allsides_class)
                    ]['answer'].values.tolist()
                    added_news_names = model_df.loc[
                        (model_df['id'] == id) & 
                        (model_df['allsides_class'] == allsides_class)
                    ]['added_news_name'].values.tolist()
                    
                    if len(model_summaries) > 0:
                        # 각 summary 분석
                        political_leanings = []
                        scores = []
                        sbert_similarities = []
                        
                        for summary in model_summaries:
                            # 1. 정치적 편향 점수
                            leaning, score = get_political_bias_score(summary)
                            political_leanings.append(leaning)
                            scores.append(score)
                            
                            # 3. SBERT 임베딩 유사도
                            model_sbert_embedding = sbert.encode(summary)
                            sbert_similarity = cosine_similarity(
                                torch.tensor(ref_sbert_embedding).unsqueeze(0),
                                torch.tensor(model_sbert_embedding).unsqueeze(0)
                            ).item()
                            sbert_similarities.append(sbert_similarity)
                        
                        # 평균 계산
                        avg_political_leaning = sum(political_leanings) / len(political_leanings)
                        avg_score = sum(scores) / len(scores)
                        avg_sbert_similarity = sum(sbert_similarities) / len(sbert_similarities)
                        
                        # Reference와 model summary 간의 편향 차이 계산
                        bias_difference = abs(ref_political_leaning - avg_political_leaning)
                        
                        # 동일 bias인지 여부 확인
                        is_same_bias = (reference_class == allsides_class)
                        
                        # 결과 저장
                        results.append({
                            'content_id': id,
                            'reference_class': reference_class,
                            'model': model_name,
                            'model_bias_class': allsides_class,
                            'is_same_bias': is_same_bias,
                            'ref_political_leaning': ref_political_leaning,
                            'ref_score': ref_score,
                            'avg_political_leaning': avg_political_leaning,
                            'avg_score': avg_score,
                            'bias_difference': bias_difference,
                            'avg_sbert_similarity': avg_sbert_similarity,
                            'num_summaries': len(model_summaries)
                        })
                        for i, added_news_name in enumerate(added_news_names):
                            results_single.append({
                                'content_id': id,
                                'model': model_name,
                                'model_bias_class': allsides_class,
                                'added_news_name': added_news_name,
                                'added_news_political_leaning': political_leanings[i],
                                'added_news_score': scores[i],
                                'added_news_sbert_similarity': sbert_similarities[i],
                            })
    
    # DataFrame으로 변환
    results_df = pd.DataFrame(results)
    results_single_df = pd.DataFrame(results_single)
    # 5. 시각화
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # 결과 디렉토리 생성
    os.makedirs("./analyze_result/summarization_rvt", exist_ok=True)
    # 결과 저장
    results_df.to_csv("./analyze_result/summarization_rvt/comprehensive_analysis.csv", index=False)
    results_single_df.to_csv("./analyze_result/summarization_rvt/comprehensive_analysis_single.csv", index=False)
    
    # 통계 분석 및 시각화
    print("=== 종합 분석 결과 ===\n")
    
    # 1. 전체 통계 (편향 차이 기준)
    same_bias_differences = results_df[results_df['is_same_bias'] == True]['bias_difference']
    diff_bias_differences = results_df[results_df['is_same_bias'] == False]['bias_difference']
    
    print(f"동일 bias 편향 차이 평균: {same_bias_differences.mean():.4f} (±{same_bias_differences.std():.4f})")
    print(f"다른 bias 편향 차이 평균: {diff_bias_differences.mean():.4f} (±{diff_bias_differences.std():.4f})")
    print(f"차이: {diff_bias_differences.mean() - same_bias_differences.mean():.4f}\n")
    
    # 2. 유사도 분석
    print("=== 유사도 분석 결과 ===")
    
    # SBERT 유사도
    same_bias_sbert_sim = results_df[results_df['is_same_bias'] == True]['avg_sbert_similarity']
    diff_bias_sbert_sim = results_df[results_df['is_same_bias'] == False]['avg_sbert_similarity']
    
    print(f"\n[SBERT 임베딩 유사도]")
    print(f"동일 bias 유사도 평균: {same_bias_sbert_sim.mean():.4f} (±{same_bias_sbert_sim.std():.4f})")
    print(f"다른 bias 유사도 평균: {diff_bias_sbert_sim.mean():.4f} (±{diff_bias_sbert_sim.std():.4f})")
    print(f"차이: {same_bias_sbert_sim.mean() - diff_bias_sbert_sim.mean():.4f}\n")
    
    # 3. Reference class별 분석
    print("=== Reference Class별 분석 ===")
    for ref_class in reference_class_list:
        ref_data = results_df[results_df['reference_class'] == ref_class]
        same_bias_diff = ref_data[ref_data['is_same_bias'] == True]['bias_difference']
        diff_bias_diff = ref_data[ref_data['is_same_bias'] == False]['bias_difference']
        same_bias_sbert = ref_data[ref_data['is_same_bias'] == True]['avg_sbert_similarity']
        diff_bias_sbert = ref_data[ref_data['is_same_bias'] == False]['avg_sbert_similarity']
        
        print(f"\n{ref_class.upper()} Reference:")
        print(f"  편향 차이 - 동일 bias: {same_bias_diff.mean():.4f} | 다른 bias: {diff_bias_diff.mean():.4f}")
        print(f"  SBERT 유사도 - 동일 bias: {same_bias_sbert.mean():.4f} | 다른 bias: {diff_bias_sbert.mean():.4f}")
    
    # 4. 모델별 분석
    print("\n=== 모델별 분석 ===")
    for model in model_names:
        model_data = results_df[results_df['model'] == model]
        same_bias_diff = model_data[model_data['is_same_bias'] == True]['bias_difference']
        diff_bias_diff = model_data[model_data['is_same_bias'] == False]['bias_difference']
        same_bias_sbert = model_data[model_data['is_same_bias'] == True]['avg_sbert_similarity']
        diff_bias_sbert = model_data[model_data['is_same_bias'] == False]['avg_sbert_similarity']
        
        print(f"\n{model}:")
        print(f"  편향 차이 - 동일 bias: {same_bias_diff.mean():.4f} | 다른 bias: {diff_bias_diff.mean():.4f}")
        print(f"  SBERT 유사도 - 동일 bias: {same_bias_sbert.mean():.4f} | 다른 bias: {diff_bias_sbert.mean():.4f}")
    
    # 5. 정치적 편향 점수별 상세 분석
    print("\n=== 정치적 편향 점수 상세 분석 ===")
    
    # Reference 편향 점수 분석
    print("\n[Reference Summary 정치적 편향 점수]")
    ref_score_by_class = results_df.groupby('reference_class')[['ref_political_leaning', 'ref_score']].mean()
    print("Reference Class별 평균:")
    for ref_class in reference_class_list:
        avg_leaning = ref_score_by_class.loc[ref_class, 'ref_political_leaning']
        avg_score = ref_score_by_class.loc[ref_class, 'ref_score']
        print(f"  {ref_class.upper()}: 정치적 편향 {avg_leaning:.4f}, 신뢰도 {avg_score:.4f}")
    
    print(f"\n전체 Reference 평균: 정치적 편향 {results_df['ref_political_leaning'].mean():.4f}, 신뢰도 {results_df['ref_score'].mean():.4f}")
    
    # Model Summary 편향 점수 분석
    print("\n[Model Summary 정치적 편향 점수]")
    print("\n모델별 & Bias Class별 평균 및 표준편차:")
    
    # 평균과 표준편차 둘 다 계산
    model_score_mean = results_df.groupby(['model', 'model_bias_class'])[['avg_political_leaning', 'avg_score']].mean()
    model_score_std = results_df.groupby(['model', 'model_bias_class'])[['avg_political_leaning', 'avg_score']].std()
    
    for model in model_names:
        print(f"\n  {model}:")
        for bias_class in ["left", "center", "right"]:
            try:
                avg_leaning = model_score_mean.loc[(model, bias_class), 'avg_political_leaning']
                std_leaning = model_score_std.loc[(model, bias_class), 'avg_political_leaning']
                avg_score = model_score_mean.loc[(model, bias_class), 'avg_score']
                std_score = model_score_std.loc[(model, bias_class), 'avg_score']
                print(f"    {bias_class.upper()}: 정치적 편향 {avg_leaning:.4f} (±{std_leaning:.4f}), 신뢰도 {avg_score:.4f} (±{std_score:.4f})")
            except KeyError:
                print(f"    {bias_class.upper()}: 데이터 없음")
    
    print(f"\n전체 Model Summary 평균: 정치적 편향 {results_df['avg_political_leaning'].mean():.4f}, 신뢰도 {results_df['avg_score'].mean():.4f}")
    
    # Reference vs Model 비교 (동일 bias vs 다른 bias)
    print("\n[Reference vs Model 정치적 편향 비교]")
    same_bias_leaning_diff = results_df[results_df['is_same_bias'] == True]['avg_political_leaning'] - results_df[results_df['is_same_bias'] == True]['ref_political_leaning']
    diff_bias_leaning_diff = results_df[results_df['is_same_bias'] == False]['avg_political_leaning'] - results_df[results_df['is_same_bias'] == False]['ref_political_leaning']
    
    print(f"동일 bias - Reference와 Model 편향 차이 평균: {same_bias_leaning_diff.mean():.4f} (±{same_bias_leaning_diff.std():.4f})")
    print(f"다른 bias - Reference와 Model 편향 차이 평균: {diff_bias_leaning_diff.mean():.4f} (±{diff_bias_leaning_diff.std():.4f})")
    
    # 정치적 편향 점수 분포 분석
    print("\n[정치적 편향 점수 분포]")
    print("Reference:")
    ref_leaning_counts = results_df['ref_political_leaning'].round().value_counts().sort_index()
    for leaning, count in ref_leaning_counts.items():
        leaning_label = "Left" if leaning == 0 else "Right"
        percentage = (count / len(results_df)) * 100
        print(f"  {leaning_label} ({int(leaning)}): {count}개 ({percentage:.1f}%)")
    
    print("\nModel Summary:")
    model_leaning_counts = results_df['avg_political_leaning'].round().value_counts().sort_index()
    for leaning, count in model_leaning_counts.items():
        leaning_label = "Left" if leaning == 0 else "Right"
        percentage = (count / len(results_df)) * 100
        print(f"  {leaning_label} ({int(leaning)}): {count}개 ({percentage:.1f}%)")
    
    # 6. 상세 분석표 생성
    print("\n=== 상세 분석표 ===")
    print("\n[편향 차이]")
    pivot_table_bias = results_df.groupby(['reference_class', 'model', 'is_same_bias'])['bias_difference'].agg(['mean', 'std', 'count']).round(4)
    print(pivot_table_bias)
    
    print("\n[SBERT 유사도]")
    pivot_table_sbert = results_df.groupby(['reference_class', 'model', 'is_same_bias'])['avg_sbert_similarity'].agg(['mean', 'std', 'count']).round(4)
    print(pivot_table_sbert)
    
    return results_df

def analyze_sips_by_model_size():
    # SIPS 데이터 읽기
    sips_df = pd.read_csv("./analyze_result/sips_all_for_figure.csv")
    
    # 모델명에서 필요한 정보 추출: 모델 계열(family), 크기(size), instruction tuning 여부
    model_info = []
    
    for idx, row in sips_df.iterrows():
        model_name = row['model']
        sips = row['SIPS']
        
        # 기본값 설정
        family = "Unknown"
        series = "Unknown"
        size = 0
        is_it = True
        
        # non-IT 여부 확인
        if "non-IT" in model_name:
            is_it = False
            model_name = model_name.replace("-non-IT", "")
            
        # HP(Hyperpartisan) 모델 제외
        if "-HP" in model_name:
            continue
            
        # 모델명 파싱
        parts = model_name.split("-")
        
        # 모델 계열 추출
        if parts[0] == "Qwen":
            family = "Qwen"
            if len(parts) > 1 and parts[1].replace(".", "").isdigit():
                # Qwen-2.5-72B와 같은 형식
                series = f"Qwen-{parts[1]}"
            else:
                series = "Qwen"
        elif parts[0] == "Llama" or (parts[0] == "Meta" and parts[1] == "Llama"):
            family = "Llama"
            if parts[0] == "Llama":
                if len(parts) > 1 and (parts[1] == "3" or parts[1].startswith("3.")):
                    series = f"Llama-{parts[1]}"
                else:
                    series = "Llama"
            else:
                series = f"Llama-{parts[2]}"
        elif parts[0] == "Mistral":
            family = "Mistral"
            if len(parts) > 1:
                series = f"Mistral-{parts[1]}"
            else:
                series = "Mistral"
        elif parts[0] == "Phi":
            family = "Phi"
            if len(parts) > 1:
                series = f"Phi-{parts[1]}"
            else:
                series = "Phi"
        elif parts[0] == "Gemma":
            family = "Gemma"
            if len(parts) > 1:
                series = f"Gemma-{parts[1]}"
            else:
                series = "Gemma"
        elif parts[0] == "QwQ":
            family = "QwQ"
            series = "QwQ"
            
        # 모델 크기 추출
        for part in parts:
            if part.endswith("B"):
                try:
                    size = float(part.replace("B", ""))
                    break
                except ValueError:
                    pass
        
        # 특수 케이스: Phi-4-14B 등 새로운 형식 처리
        if size == 0 and family == "Phi" and len(parts) >= 3:
            try:
                if parts[2].endswith("B"):
                    size = float(parts[2].replace("B", ""))
            except (ValueError, IndexError):
                pass
        
        model_info.append({
            'model': model_name,
            'family': family,
            'series': series,
            'size': size,
            'is_it': is_it,
            'sips': sips
        })
    
    # 디버깅: 파싱 결과 확인
    model_df = pd.DataFrame(model_info)
    print("Model parsing results:")
    print(model_df[['model', 'family', 'series', 'size', 'is_it', 'sips']])
    
    # 유효한 크기 정보가 있는 모델만 선택
    model_df = model_df[model_df['size'] > 0]
    
    # --- 두 개의 subplot 생성 (동일한 크기) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300, gridspec_kw={'width_ratios': [1, 1]})
    ax = axes[0]  # 왼쪽: SIPS by Model Size
    ax2 = axes[1] # 오른쪽: Allsides 그래프
    
    # --- SIPS 그래프 (왼쪽) ---
    family_markers = {
        'Qwen': 'o',       # 원
        'Llama': 's',      # 사각형
        'Mistral': '^',    # 삼각형
        'Phi': '*',        # 별
        'Gemma': 'D',      # 다이아몬드
        'QwQ': 'X'         # X
    }
    
    series_colors = {
        'Qwen-2.5': '#E76F51',
        'Llama-3.3': '#1F4E79',
        'Llama-3.1': '#457B9D',
        'Llama-3': '#6BAED6',
        'Mistral': '#E9C46A',
        'Phi-4': '#2A9D8F',
        'Gemma-2': '#8D6AA9',
        'QwQ': '#C2185B'
    }
    
    # 추가된 데이터 그룹화 및 정렬
    grouped = model_df.groupby(['series', 'is_it'])
    
    # 범례를 위한 항목 추적
    legend_items = {}
    
    # 범례 항목을 모으기 위한 리스트
    combined_handles = []
    combined_labels = []
    
    # 각 시리즈 및 IT 여부별로 그래프 그리기
    for (series, is_it), group in grouped:
        # 색상 및 마커 결정
        family = group['family'].iloc[0]
        color = next((v for k, v in series_colors.items() if series.startswith(k)), '#333333')
        marker = family_markers.get(family, 'o')
        
        # 라인 스타일 결정 (IT 여부에 따라)
        linestyle = '-' if is_it else '--'
        
        # Non-IT 모델을 위한 빈 마커 설정
        if is_it:
            markerfacecolor = color  # IT 모델은 채워진 마커
        else:
            markerfacecolor = 'white'  # Non-IT 모델은 빈 마커
        
        # 크기 순으로 정렬
        group = group.sort_values('size')
        
        # 그래프 그리기
        line, = ax.plot(group['size'], group['sips'], 
                 marker=marker, linestyle=linestyle, color=color, 
                 markerfacecolor=markerfacecolor, markeredgecolor=color,
                 markersize=10, linewidth=2)
        
        # 각 포인트에 모델명 추가하는 부분 삭제 (라벨 없애기)
        
        # 범례 아이템 저장
        label = f"{series} {'(IT)' if is_it else '(non-IT)'}"
        legend_items[label] = line
    
    ax.set_xscale('log')
    ax.set_xlabel('Model Size (Billion Parameters)', fontsize=12)
    ax.set_ylabel('SIPS', fontsize=12)
    ax.set_title('SIPS by Model Size and Family', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # --- 오른쪽 그래프: Allsides 결과 ---
    # 모델 데이터 로드
    qwen_log_dir = "../../logs/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias/20250219_145141/result_4400.csv"
    mistral_log_dir = "../../logs/allsides/mistralai/Mistral-Small-24B-Instruct-2501/random_medias/20250221_124715/result_4499.csv"
    phi_log_dir = "../../logs/allsides/microsoft/phi-4/random_medias/20250219_145326/result_4400.csv"
    llama_log_dir = "../../logs/allsides/meta-llama/Llama-3.3-70B-Instruct/random_medias/20250219_145234/result_4400.csv"
    gemma_log_dir = "../../logs/allsides/google/gemma-2-27b-it/random_medias/20250410_003055/result_4499.csv"

    qwen_result_dict = load_log_file(qwen_log_dir)
    mistral_result_dict = load_log_file(mistral_log_dir)
    phi_result_dict = load_log_file(phi_log_dir)
    llama_result_dict = load_log_file(llama_log_dir)
    gemma_result_dict = load_log_file(gemma_log_dir)
    
    # 데이터 준비
    x_labels = ['left', 'lean_left', 'center', 'lean_right', 'right']
    x_positions = np.arange(len(x_labels))

    # 각 result_dict에 대한 평균, 최소, 최대값
    qwen_values = calculate_mean_values_graph(qwen_result_dict)
    mistral_values = calculate_mean_values_graph(mistral_result_dict)
    phi_values = calculate_mean_values_graph(phi_result_dict)
    llama_values = calculate_mean_values_graph(llama_result_dict)
    gemma_values = calculate_mean_values_graph(gemma_result_dict)
    
    # 모델별 색상과 마커 (왼쪽 그래프와 일치)
    model_info = [
        {'name': 'Qwen', 'values': qwen_values, 'family': 'Qwen', 'series': 'Qwen-2.5'},
        {'name': 'Mistral', 'values': mistral_values, 'family': 'Mistral', 'series': 'Mistral'},
        {'name': 'Phi', 'values': phi_values, 'family': 'Phi', 'series': 'Phi-4'},
        {'name': 'Llama', 'values': llama_values, 'family': 'Llama', 'series': 'Llama-3.3'},
        {'name': 'Gemma', 'values': gemma_values, 'family': 'Gemma', 'series': 'Gemma-2'}
    ]
    
    # 각 모델의 none 값 출력
    print("\n=== None values and SIPS for each model ===")
    
    # 각 모델 플롯
    for model in model_info:
        family = model['family']
        series = model['series']
        color = next((v for k, v in series_colors.items() if series.startswith(k)), '#333333')
        marker = family_markers.get(family, 'o')
        
        # 평균값 그리기
        line, = ax2.plot(x_positions, [model['values']['mean'][label] for label in x_labels], 
                marker=marker, linewidth=2, color=color, label=model['name'])
        
        # 모델 이름 결정
        model_label = ""
        if model['name'] == 'Qwen':
            model_label = "Qwen-2.5-72B"
        elif model['name'] == 'Llama':
            model_label = "Llama-3.3-70B"
        elif model['name'] == 'Phi':
            model_label = "Phi-4-14B"
        elif model['name'] == 'Gemma':
            model_label = "Gemma-2-27B"
        elif model['name'] == 'Mistral':
            model_label = "Mistral-small-24B"
        
        # none 값 터미널에 출력
        none_val = model['values']['none']
        print(f"{model_label} - none: {none_val:.2f}")
        
        # SIPS 값 터미널에 출력
        model_sips_row = model_df[model_df['model']==model_label]
        if not model_sips_row.empty:
            model_sips = model_sips_row['sips'].iloc[0]
            print(f"{model_label} - SIPS: {model_sips:.2f}")
    
    # Reference Line: Bias Class가 left -> right일 때 -2 -> 2
    biased_line = np.linspace(-2, 2, len(x_labels))
    ax2.plot(x_positions, biased_line, linestyle='--', color='#2F2F2F')
    
    # Reference Line: 모든 값이 0
    ax2.axhline(0, color='#A8A8A8', linestyle='--')
    
    # Reference Line 텍스트 제거 (더 이상 필요 없음)

    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(x_labels, fontsize=12)
    ax2.set_xlabel('Media Source Political Bias', fontsize=13)
    ax2.set_ylabel('Model Prediction Shift', fontsize=13)
    ax2.set_title('Political Bias by Media Source', fontsize=16)
    ax2.set_ylim(-2, 2)
    ax2.grid(True, alpha=0.3)
    
    # 우측 그림의 범례 제거
    
    # 왼쪽 그래프의 범례 항목 추가
    for label, line in legend_items.items():
        combined_handles.append({'handle': line, 'label': f"{label}"})
    
    # 오른쪽 그래프의 모델 라인에 대한 범례도 추가
    for model in model_info:
        model_name = model['name']
        color = next((v for k, v in series_colors.items() if model['series'].startswith(k)), '#333333')
        marker = family_markers.get(model['family'], 'o')
        
        combined_handles.append({'handle': plt.Line2D([0], [0], color=color, marker=marker, 
                                              linestyle='-', markersize=10),
                                'label': f"{model_name} (Right Graph)"})
    
    # 기준선에 대한 범례도 추가
    combined_handles.append({'handle': plt.Line2D([0], [0], color='#2F2F2F', linestyle='--'), 
                           'label': 'Biased Reference'})
    combined_handles.append({'handle': plt.Line2D([0], [0], color='#A8A8A8', linestyle='--'), 
                           'label': 'Unbiased Reference'})
    
    # 범례를 figure 하단에 배치 (공유)
    handles = [item['handle'] for item in combined_handles]
    labels = [item['label'] for item in combined_handles]
    fig.legend(handles, labels, fontsize=11, loc='lower center',  
              bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=True)
    
    # 더 큰 하단 여백 확보
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, right=0.95)  # 하단 여백 증가
    plt.savefig("./analyze_result/sips_by_model_size_double.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Combined figure saved to './analyze_result/sips_by_model_size_double.png'")
    

def analyze_summarization_human_evaluation():
    human_evaluation_dir = "./a-study-on-the-degree-of-political-bias-perceived-in-ai-generated-news-summaries.csv"
    human_evaluation_df = pd.read_csv(human_evaluation_dir)
    
    # Krippendorff's alpha 계산을 위한 라이브러리 import
    try:
        import krippendorff
    except ImportError:
        print("krippendorff 라이브러리가 설치되지 않았습니다. 다음 명령어로 설치하세요:")
        print("pip install krippendorff")
        return
    
    import numpy as np
    
    alpha_results = []
    
    print("=== Krippendorff's Alpha 분석 ===\n")
    print("편향 매핑: Left=0, Center=1, Right=2\n")
    
    bias_columns = [f"bias{i}" for i in range(1, 31)]
    only_bias_df = human_evaluation_df[bias_columns]

    # 문자열을 숫자로 매핑
    bias_mapping = {
        'Left': 0, 'left': 0, 'LEFT': 0,
        'Center': 1, 'center': 1, 'CENTER': 1,
        'Right': 2, 'right': 2, 'RIGHT': 2
    }
    
    def map_bias_values(row):
        """각 행의 bias 값들을 숫자로 매핑"""
        mapped_row = []
        for value in row:
            if pd.isna(value):
                mapped_row.append(None)  # 결측값
            elif isinstance(value, str):
                if value in bias_mapping:
                    mapped_row.append(bias_mapping[value])
                else:
                    print(f"알 수 없는 값 '{value}' 발견")
                    mapped_row.append(None)
            else:
                # 이미 숫자인 경우
                mapped_row.append(float(value))
        return mapped_row

    coder1 = map_bias_values(only_bias_df.iloc[0, :].values.tolist())
    coder2 = map_bias_values(only_bias_df.iloc[1, :].values.tolist())
    coder3 = map_bias_values(only_bias_df.iloc[2, :].values.tolist())
    coder4 = map_bias_values(only_bias_df.iloc[3, :].values.tolist())
    coder5 = map_bias_values(only_bias_df.iloc[4, :].values.tolist())
    
    coder1_good = 0
    coder2_good = 0
    coder3_good = 0
    coder4_good = 0
    coder5_good = 0
    
    coder1_bad = 0
    coder2_bad = 0
    coder3_bad = 0
    coder4_bad = 0
    coder5_bad = 0
    
    for i in range(1, 31, 3):
        this_coder1 = coder1[i:i+3]
        this_coder2 = coder2[i:i+3]
        this_coder3 = coder3[i:i+3]
        this_coder4 = coder4[i:i+3]
        this_coder5 = coder5[i:i+3]
        
        if len(set(this_coder1)) > 1:
            coder1_good += 1
        else:
            coder1_bad += 1
            
        if len(set(this_coder2)) > 1:
            coder2_good += 1
        else:
            coder2_bad += 1
            
        if len(set(this_coder3)) > 1:
            coder3_good += 1
        else:
            coder3_bad += 1
            
        if len(set(this_coder4)) > 1:
            coder4_good += 1
        else:
            coder4_bad += 1
            
        if len(set(this_coder5)) > 1:
            coder5_good += 1
        else:
            coder5_bad += 1
            
    print(f"coder1_good: {coder1_good}, coder1_bad: {coder1_bad}")
    print(f"coder2_good: {coder2_good}, coder2_bad: {coder2_bad}")
    print(f"coder3_good: {coder3_good}, coder3_bad: {coder3_bad}")
    print(f"coder4_good: {coder4_good}, coder4_bad: {coder4_bad}")
    print(f"coder5_good: {coder5_good}, coder5_bad: {coder5_bad}")
    

if __name__ == "__main__":
#     print("analyze_main_result_allsides")
#     analyze_main_result_allsides()
#     print("analyze_main_result_hyperpartisan")
#     analyze_main_result_hyperpartisan()
#     print("analyze_model_size_qwen")
#     analyze_model_size_qwen()
#     print("analyze_model_size_llama")
#     analyze_model_size_llama()
#     print("analyze_model_series_qwen")
#     analyze_model_series_qwen()
#     print("analyze_model_series_llama")
#     analyze_model_series_llama()
#     print("analyze_impact_of_it_qwen_large")
#     analyze_impact_of_it_qwen_large()
#     print("analyze_impact_of_it_qwen_small")
#     analyze_impact_of_it_qwen_small()
#     print("analyze_impact_of_it_llama_and_mistral")
#     analyze_impact_of_it_llama_and_mistral()
#     print("analyze_impact_of_reasoning")
#     analyze_impact_of_reasoning()
#     print("analyze_synthetic_medias")
#     analyze_synthetic_medias()
#     print("analyze_sips")
#     analyze_sips()
#     print("analyze_sips_prompt")
#     analyze_sips_prompt()
#     print("analyze_sips_prompt_rvt")
#     analyze_sips_prompt_rvt()
#     print("analyze_sips_order_rvt")
#     analyze_sips_order_rvt()
    # print("analyze_difference_between_order")
    # analyze_difference_between_order()
#     print("analyze_sips_by_model_size")
#     analyze_sips_by_model_size()
    # print("analyze_summarization_political_bias")
    # analyze_summarization_political_bias()
    # print("analyze_summarization_political_bias_only_classification")
    # analyze_summarization_political_bias_only_classification()
    print("analyze_summarization_human_evaluation")
    analyze_summarization_human_evaluation()
    