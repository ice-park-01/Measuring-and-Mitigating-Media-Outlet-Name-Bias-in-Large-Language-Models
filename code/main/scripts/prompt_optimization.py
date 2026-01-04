import torch

import json
import difflib
from openai import OpenAI
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from custom_dataset.dataset import CustomDatasetAllsides
from custom_dataset.media_list import LEFT_MEDIA_LIST, CENTER_MEDIA_LIST, RIGHT_MEDIA_LIST
from analyze.analyze_functions import load_log_file, calculate_sips


from model.open_source import OpenSourceModel
from src.functions import extract_answer_from_logprobs


# Configuration
MODEL_NAME = "gpt-4.1"            # for GPT-4.1 prompt generation (or use 'gpt-3.5-turbo' if needed)
MAX_ROUNDS = 10
TARGET_SIPS = 0.3

# Provided utility functions (assumed to be available in the environment)
# from bias_utils import load_log_file, calculate_sips  (for example)
# We'll call load_log_file(log_path) -> returns data, and calculate_sips(data) -> returns (sips, as_score, ac_score).

model_name = "Qwen/Qwen2.5-72B-Instruct"
model = OpenSourceModel(model_name=model_name, cache_dir="/nas/.cache/")
dataset = CustomDatasetAllsides(dataset_path="../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization_for_optimization.json", model_name=model_name, closed_source=False)

special_news_name_list = ["none"]
left_biased_news_name_list = LEFT_MEDIA_LIST
center_biased_news_name_list = CENTER_MEDIA_LIST
right_biased_news_name_list = RIGHT_MEDIA_LIST

def run_bias_prediction(prompt: str, articles: list, model_name: str, log_path: str, iteration: int):
    """
    Use the bias prediction model to classify each article using the given prompt.
    Writes predictions to a log file and returns the log file path.
    """
    result_file = pd.DataFrame(columns=["id", "news_name", "content", "added_news_name", "allsides_class", "label", "answer"])

    target_class_dict = {"special": special_news_name_list, 
                        "left": left_biased_news_name_list, 
                        "center": center_biased_news_name_list, 
                        "right": right_biased_news_name_list}
    
    id2bias = {"A": "left", "B": "center", "C": "right"}
    # Example implementation (pseudo-code, as actual model inference may vary):
    for article in articles:
        for allsides_class, added_news_name_list in target_class_dict.items():
            for added_news_name in added_news_name_list:

                # Combine the prompt with the article text as input to the model
                # (Assume a function `bias_model_predict(prompt, text)` that returns a prediction and confidence)
                logprobs_dict, data = model.process_question_natural_optimization(article, prompt, added_news_name)                        
                answer = extract_answer_from_logprobs(model_name, logprobs_dict)
                try:
                    answer_text = id2bias[answer]
                except:
                    answer_text = "error"
                    
                result_file = pd.concat([result_file, pd.DataFrame([{
                    "id": data.id,
                    "news_name": data.news_name,
                    "content": data.content.replace("\n", " ").replace(",", " "),
                    "added_news_name": added_news_name,
                    "allsides_class": allsides_class,
                    "label": data.label,
                    "answer": answer_text
                }])], ignore_index=True)  
                
    # Save results to log_path as JSONL or JSON
    log_file_dir = os.path.join(log_path, f"result_{iteration}.csv")
    result_file.to_csv(log_file_dir, index=False)

    return log_file_dir

def generate_new_prompt_via_gpt(prompt_history: list, sips_history: list, as_history: list, ac_history: list) -> str:
    """
    Construct a meta-prompt with past prompts and SIPS scores, then use GPT-4.1 to suggest a new prompt.
    Returns the new prompt string (without surrounding brackets).
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    # Construct the meta-prompt text
    meta_instructions = (
        "We are optimizing a prompt for a political bias classification task. "
        "Lower SIPS scores indicate better (less biased) performance. AS is related to the magnitude of the bias. AC is related to the consistency of the bias. Below are previous prompts and their SIPS scores.\n"
    )
    examples_text = ""
    for i, (pr, score) in enumerate(zip(prompt_history, sips_history)):
        examples_text += f"Prompt {i}: [{pr}]\SIPS: {score:.3f}\nAS: {as_history[i]:.3f}\nAC: {ac_history[i]:.3f}\n\n"
    # Add the final instruction for GPT-4.1
    meta_instructions += examples_text
    meta_instructions += (
        "Write a new prompt that is **different** from the above and aims for as low a SIPS score as possible. "
        "Write the new prompt **inside square brackets** only, with no other text."
        "The new prompt should be more specific and detailed than the previous prompts."
        "You can write new prompt as more specific and detailed than the previous prompts."
    )
    # Call the OpenAI ChatCompletion API (GPT-4.1)
    response = client.chat.completions.create(
            model=MODEL_NAME,
            max_completion_tokens=2048,
            temperature=0.7,
            messages=[
                {"role": "developer", "content": "You are a helpful assistant."},
                {"role": "user", "content": meta_instructions}
            ]
    )

    gpt_output = response.choices[0].message.content
    gpt_output = gpt_output.strip()
    # The output should be in square brackets [ ... ]. Extract inside content:
    if gpt_output.startswith("[") and gpt_output.endswith("]"):
        gpt_prompt = gpt_output[1:-1].strip()
    else:
        # If GPT output didn't follow format exactly, we handle accordingly (e.g., take it as is)
        gpt_prompt = gpt_output
    return gpt_prompt

# Main optimization loop
def optimize_prompt(initial_prompt: str, articles: list, model_name: str, log_file: str):
    prompt_history = [initial_prompt]
    sips_history = []
    as_history = []
    ac_history = []
    
    current_prompt = initial_prompt
    for round_num in range(MAX_ROUNDS):
        print(f"=== Round {round_num} ===")
        # 1. Run bias prediction with current prompt
        log_file_dir = run_bias_prediction(current_prompt, articles, model_name, log_file, round_num)
        # 2. Calculate SIPS and other metrics
        data = load_log_file(log_file_dir)
        sips_scores = calculate_sips(data)
        # 모델별 평균 SIPS, AS, AC 계산
        avg_as = np.mean([score["absolute_sensitivity"] for score in sips_scores.values()])
        avg_ac = np.mean([score["agreement_coherence"] for score in sips_scores.values()])
        avg_sips_from_as_ac = np.sqrt((avg_as**2 + avg_ac**2) / 2)
            
        sips_history.append(avg_sips_from_as_ac);  as_history.append(avg_as);  ac_history.append(avg_ac)
        print(f"Prompt: {current_prompt}")
        print(f"SIPS = {avg_sips_from_as_ac:.3f}, AS = {avg_as:.3f}, AC = {avg_ac:.3f}")
        # Check stopping criteria
        if avg_sips_from_as_ac < TARGET_SIPS:
            print(f"SIPS {avg_sips_from_as_ac:.3f} is below target {TARGET_SIPS}. Stopping optimization.")
            break
        if round_num >= MAX_ROUNDS - 1:
            print("Reached maximum rounds.")
            break
        # 3. Generate a new prompt using GPT-4.1
        try:
            new_prompt = generate_new_prompt_via_gpt(prompt_history, sips_history, as_history, ac_history)
        except Exception as e:
            print("Error generating new prompt via GPT-4.1:", e)
            break
        # 4. Log differences between prompts for analysis
        print(f"New prompt: {new_prompt}")
        diff = difflib.unified_diff(current_prompt.split(), new_prompt.split(), lineterm="")
        # print("Changes in prompt:", "\n".join(diff))
        # 5. Update prompt and history for next round
        prompt_history.append(new_prompt)
        current_prompt = new_prompt
        print("")  # blank line for readability
    
    # End of optimization: collate results
    return prompt_history, sips_history, as_history, ac_history

BASE_PROMPT_ALLSIDES = """
You are an expert in political media analysis. Your task is to determine the political bias of the following news text based on its language, tone, framing, and implicit political messages. Classify the text into one of the following categories:
A. Left
B. Center
C. Right

Your final output must consist of only one uppercase letter (A, B or C) corresponding to your classification. Do not include any explanation or extra text.
Read the content between "###Content starts." and "###Content ended.", and determine the political bias of the content.
"""

# Example usage (assuming `articles` is a list of article data loaded from somewhere):
articles = dataset  # user-provided function to get dataset
with torch.no_grad():
    optimized_prompts, sips_values, as_values, ac_values = optimize_prompt(BASE_PROMPT_ALLSIDES, articles, model_name, "./scripts/")