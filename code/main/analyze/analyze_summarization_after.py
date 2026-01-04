import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import pandas as pd
import os
from tqdm import tqdm
    
# import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from NewsSentiment import TargetSentimentClassifier
import argparse
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def inspect_result_df(df, summary_length):
    ans_longer_than_summary_length = 0
    total_ans_length = 0
    avg_ans_length = 0
    for index, row in df.iterrows():
        model_summary = row['answer']
        if len(model_summary.split(".")) > summary_length:
            ans_longer_than_summary_length += 1
        total_ans_length += len(model_summary.split("."))
    avg_ans_length = total_ans_length / len(df)
    print(f"total answer length: {total_ans_length}")
    print(f"answer length longer than summary length: {ans_longer_than_summary_length}")
    print(f"average answer length: {avg_ans_length}")
    

def predict_bias_mtsc(df_dir):
    print(df_dir)
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    tsc = TargetSentimentClassifier()
    
    df = pd.read_csv(df_dir)
    
    
    result_df = pd.DataFrame(columns=['id', 'news_name', 'content', 'added_news_name', 'allsides_class', 'label', 'answer','Entity', 'Word', 'Sentiment', 'Probability'])
    progress_bar = tqdm(range(len(df)))
    for index, row in df.iterrows():
        try:
            model_summary = row['answer']
            
            ner_spans = nlp(model_summary)
            ents = [span["word"] for span in ner_spans]
            # print(f"Entities: {ents}")
            for span in ner_spans:
                l = model_summary[:span['start']]
                m = model_summary[span['start']:span['end']]
                r = model_summary[span['end']:]
                sentiment = tsc.infer(l, m, r, disable_tqdm=True)
                # print(sentiment)
                # print(f"{span['entity']}\t{sentiment[0]['class_label']}\t{sentiment[0]['class_prob']:.2f}\t{m}")

                result_row = row.copy()
                result_row['Entity'] = span['entity']
                result_row['Word'] = span['word']
                result_row['Sentiment'] = sentiment[0]['class_label']
                result_row['Probability'] = sentiment[0]['class_prob']
                result_df = pd.concat([result_df, pd.DataFrame([result_row])], ignore_index=True)
                
        except Exception as e:
            print(f"Error: {e}")
            model_summary = row['answer'][:int(0.1*len(row['answer']))]
            
            ner_spans = nlp(model_summary)
            ents = [span["word"] for span in ner_spans]
            # print(f"Entities: {ents}")
            for span in ner_spans:
                l = model_summary[:span['start']]
                m = model_summary[span['start']:span['end']]
                r = model_summary[span['end']:]
                sentiment = tsc.infer(l, m, r, disable_tqdm=True)
                # print(sentiment)
                # print(f"{span['entity']}\t{sentiment[0]['class_label']}\t{sentiment[0]['class_prob']:.2f}\t{m}")

                result_row = row.copy()
                result_row['Entity'] = span['entity']
                result_row['Word'] = span['word']
                result_row['Sentiment'] = sentiment[0]['class_label']
                result_row['Probability'] = sentiment[0]['class_prob']
                result_df = pd.concat([result_df, pd.DataFrame([result_row])], ignore_index=True)

        progress_bar.update(1)
        
        if index % 50 == 0:
            result_df.to_csv(df_dir.replace(".csv", f"_bias_{index}.csv"), index=False)
    result_df.to_csv(df_dir.replace(".csv", "_bias.csv"), index=False)
        

def predict_bias_of_article(df_dir, model_name):
    df = pd.read_csv(df_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    map_dict = {0: "left", 1: "center", 2: "right"}
    
    result_df = df.copy()
    progress_bar = tqdm(range(len(df)))
    for index, row in df.iterrows():
        original_article = row['content']
        original_article_tokens = tokenizer.encode(original_article, return_tensors='pt', max_length=512, truncation=True)
        original_article_tokens = original_article_tokens.to(device)
        
        with torch.no_grad():
            outputs = model(original_article_tokens)
            logits = outputs.logits
            probs = str(logits.softmax(dim=-1)[0].tolist())
            predictions = torch.argmax(logits, dim=1).cpu().detach().numpy()[0]
            result_df.loc[index, 'answer'] = map_dict[predictions]
            result_df.loc[index, 'probs'] = probs
        progress_bar.update(1)
    result_df.to_csv(df_dir.replace(".csv", "_original_article_bias.csv"), index=False)      
    

def predict_bias_mtsc_batch_optimized(df_dir, batch_size=64):
    print(df_dir)
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-large-NER", max_length=500, truncation=True)
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-large-NER")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # NER 파이프라인 설정 (매개변수 직접 전달)
    nlp = pipeline(
        "ner", 
        model=model, 
        tokenizer=tokenizer, 
        device=0 if torch.cuda.is_available() else -1,
        aggregation_strategy="simple",
    )
    tsc = TargetSentimentClassifier()
    
    # 데이터 로드
    df = pd.read_csv(df_dir)
        
    # 1. df.iterrows()를 사용하여 데이터셋 생성 및 배치로 분할
    rows = list(df.iterrows())
    total_samples = len(rows)
    num_batches = (total_samples + batch_size - 1) // batch_size
    
    # NER 캐시 저장용 딕셔너리
    ner_cache = {}
    
    print("1단계: 배치 단위로 NER 수행 및 캐시")
    progress_bar = tqdm(total=num_batches)
    
    # 2. 배치 단위로 ner_spans 도출 및 캐시 저장
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_samples)
        batch_rows = rows[start_idx:end_idx]
        
        # 배치의 텍스트 추출
        batch_indices = [idx for idx, _ in batch_rows]
        batch_texts = []
        
        for _, row in batch_rows:
            text = row['answer']
            # 너무 긴 텍스트는 미리 자르기
            if len(text) > 1000:  # 문자 단위 임계값 설정
                text = text[:1000]
            batch_texts.append(text)
        
        try:
            # 배치 단위로 NER 처리
            batch_ner_results = nlp(batch_texts)
            
            # 결과 구조 확인 (배치 결과가 리스트의 리스트인지 단일 리스트인지)
            if not isinstance(batch_ner_results[0], list) and len(batch_texts) > 1:
                # 단일 리스트인 경우 (첫 번째 텍스트의 결과만 포함)
                ner_cache[batch_indices[0]] = batch_ner_results
                # 나머지 텍스트는 개별 처리
                for i in range(1, len(batch_texts)):
                    idx = batch_indices[i]
                    ner_cache[idx] = nlp(batch_texts[i])
            else:
                # 리스트의 리스트인 경우 (모든 텍스트의 결과 포함)
                for i, (idx, ner_spans) in enumerate(zip(batch_indices, batch_ner_results)):
                    ner_cache[idx] = ner_spans
            
        except Exception as e:
            print(f"배치 처리 오류 (배치 {batch_idx}): {e}")
            # 오류 발생 시 개별 처리
            for i, (idx, row) in enumerate(batch_rows):
                try:
                    text = batch_texts[i]
                    single_result = nlp(text)
                    ner_cache[idx] = single_result
                except Exception as e2:
                    print(f"개별 처리 오류 (샘플 {start_idx + i}): {e2}")
                    ner_cache[idx] = []
        
        progress_bar.update(1)
    
    progress_bar.close()
    print("2단계: 캐시된 NER 결과에 대한 감정 분석 대상 준비")
    
    # 3. 캐시된 NER 결과에 대한 감정 분석 대상 준비
    all_targets = []
    all_span_info = []
    row_indices = []  # 각 target이 어떤 row에서 왔는지 추적

    print("2단계: 감정 분석을 위한 target 준비")
    progress_bar = tqdm(total=total_samples)

    for idx, (index, row) in enumerate(rows):
        try:
            model_summary = row['answer']
            ner_spans = ner_cache[index]
            
            if not ner_spans:  # 빈 리스트인 경우 건너뛰기
                progress_bar.update(1)
                continue
            
            for span in ner_spans:
                l = model_summary[:span['start']]
                m = model_summary[span['start']:span['end']]
                r = model_summary[span['end']:]
                
                # 전체 targets 리스트에 추가
                all_targets.append((l.strip(), m.strip(), r.strip()))
                # 결과 매핑을 위한 정보 저장
                all_span_info.append({
                    'entity': span['entity_group'],
                    'word': span['word'],
                    'row': row  # 원본 행 저장
                })
                row_indices.append(idx)
        except Exception as e:
            print(f"대상 준비 오류 (샘플 {idx}): {e}")
        
        progress_bar.update(1)

    progress_bar.close()


    print(f"3단계: 감정 분석 수행 (총 {len(all_targets)}개 타겟)")
    # 결과 데이터프레임 초기화
    result_df = pd.DataFrame(columns=['id', 'news_name', 'content', 'added_news_name', 'allsides_class', 'label', 'answer', 'Entity', 'Word', 'Sentiment', 'Probability'])
        
    # 배치로 감정 분석 수행
    chunk_size = 128
    num_chunks = ((len(all_targets)) // chunk_size) + 1
    progress_bar = tqdm(total=num_chunks)
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(all_targets))
        chunk_targets = all_targets[start_idx:end_idx]
        chunk_info = all_span_info[start_idx:end_idx]
        
        try:
            sentiments = tsc.infer(targets=chunk_targets, batch_size=128, disable_tqdm=True)
            
            for i, sentiment in enumerate(sentiments):
                info = chunk_info[i]
                sentiment = sentiment[0]
                
                result_row = info['row'].copy()
                result_row['Entity'] = info['entity']
                result_row['Word'] = info['word']
                result_row['Sentiment'] = sentiment['class_label']
                result_row['Probability'] = sentiment['class_prob']
                result_df = pd.concat([result_df, pd.DataFrame([result_row])], ignore_index=True)
                
        except Exception as e:
            print(f"처리 오류: {e}, idx: {i}, 단일 추론")
            for i, target in enumerate(chunk_targets):
                try:
                    single_sentiment = tsc.infer(text_left=target[0], target_mention=target[1], text_right=target[2], disable_tqdm=True)
                    info = chunk_info[i]
                    
                    single_sentiment_most = single_sentiment[0]
                    
                    result_row = info['row'].copy()
                    result_row['Entity'] = info['entity']
                    result_row['Word'] = info['word']
                    result_row['Sentiment'] = single_sentiment_most['class_label']
                    result_row['Probability'] = single_sentiment_most['class_prob']
                    result_df = pd.concat([result_df, pd.DataFrame([result_row])], ignore_index=True)
                except Exception as e2:
                    print(f"처리 오류: {e2}, idx: {i}")
                    continue
        
        progress_bar.update(1)
    # try:
    #     sentiments = tsc.infer(targets=all_targets, batch_size=128, disable_tqdm=False)
        
    #     # 결과 저장
    #     for i, sentiment in enumerate(sentiments):
    #         info = all_span_info[i]
    #         sentiment = sentiment[0]
            
    #         result_row = info['row'].copy()
    #         result_row['Entity'] = info['entity']
    #         result_row['Word'] = info['word']
    #         result_row['Sentiment'] = sentiment['class_label']
    #         result_row['Probability'] = sentiment['class_prob']
    #         result_df = pd.concat([result_df, pd.DataFrame([result_row])], ignore_index=True)
            
    # except Exception as e:
    #     print(f"처리 오류: idx: {i},{e}")
    
    # 4. 최종 결과 저장
    result_df.to_csv(df_dir.replace(".csv", "_bias_batch_optimized.csv"), index=False)
    print(f"분석 완료. 결과 저장됨: {df_dir.replace('.csv', '_bias_batch_optimized.csv')}")
    
    return result_df


if __name__ == "__main__":   
    # df_dir = "../../logs/summarization/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias_for_summarization/3/20250418_204514/result_449.csv"
    # df = pd.read_csv(df_dir)
    # predict_bias_mtsc_batch_optimized(df_dir)
    
    # df_dir = "../../logs/summarization/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias_for_summarization/5/20250421_003934/result_449.csv"
    # df = pd.read_csv(df_dir)
    # predict_bias_mtsc_batch_optimized(df_dir)
    
    # df_dir = "../../logs/summarization/allsides/Qwen/Qwen2.5-72B-Instruct/random_medias_for_summarization/10/20250422_100037/result_449.csv"
    # df = pd.read_csv(df_dir)
    # predict_bias_mtsc_batch_optimized(df_dir)
    
    # df_dir = "../../logs/summarization/allsides/mistralai/Mistral-Small-24B-Instruct-2501/random_medias_for_summarization/3/20250418_204510/result_449.csv"
    # df = pd.read_csv(df_dir)
    # predict_bias_mtsc_batch_optimized(df_dir)
    
    # df_dir = "../../logs/summarization/allsides/mistralai/Mistral-Small-24B-Instruct-2501/random_medias_for_summarization/5/20250423_170608/result_449.csv"
    # df = pd.read_csv(df_dir)
    # predict_bias_mtsc_batch_optimized(df_dir)
    
    # ------------------------------------------------------------------------------------------------

    # df_dir = "../../logs/summarization/allsides/mistralai/Mistral-Small-24B-Instruct-2501/random_medias_for_summarization/10/20250423_170726/result_449.csv"
    # df = pd.read_csv(df_dir)
    # predict_bias_mtsc_batch_optimized(df_dir)
    
    # df_dir = "../../logs/summarization/allsides/microsoft/phi-4/random_medias_for_summarization/3/20250501_160924/result_449.csv"
    # df = pd.read_csv(df_dir)
    # predict_bias_mtsc_batch_optimized(df_dir)
    
    # df_dir = "../../logs/summarization/allsides/microsoft/phi-4/random_medias_for_summarization/5/20250501_160948/result_449.csv"
    # df = pd.read_csv(df_dir)
    # predict_bias_mtsc_batch_optimized(df_dir)
    
    # df_dir = "../../logs/summarization/allsides/microsoft/phi-4/random_medias_for_summarization/10/20250502_154939/result_449.csv"
    # df = pd.read_csv(df_dir)
    # predict_bias_mtsc_batch_optimized(df_dir)
    
    # df_dir = "../../logs/summarization/allsides/meta-llama/Llama-3.3-70B-Instruct/random_medias_for_summarization/3/20250504_145626/result_449.csv"
    # df = pd.read_csv(df_dir)
    # predict_bias_mtsc_batch_optimized(df_dir)
    
    # ------------------------------------------------------------------------------------------------
    
    # df_dir = "../../logs/summarization/allsides/meta-llama/Llama-3.3-70B-Instruct/random_medias_for_summarization/5/20250505_015658/result_449.csv"
    # df = pd.read_csv(df_dir)
    # predict_bias_mtsc_batch_optimized(df_dir)
    
    # df_dir = "../../logs/summarization/allsides/meta-llama/Llama-3.3-70B-Instruct/random_medias_for_summarization/10/20250505_140706/result_449.csv"
    # df = pd.read_csv(df_dir)
    # predict_bias_mtsc_batch_optimized(df_dir)
    
    # df_dir = "../../logs/summarization/allsides/google/gemma-2-27b-it/random_medias_for_summarization/3/20250502_124918/result_449.csv"
    # df = pd.read_csv(df_dir)
    # predict_bias_mtsc_batch_optimized(df_dir)
    
    # df_dir = "../../logs/summarization/allsides/google/gemma-2-27b-it/random_medias_for_summarization/5/20250503_124817/result_449.csv"
    # df = pd.read_csv(df_dir)
    # predict_bias_mtsc_batch_optimized(df_dir)
    
    # df_dir = "../../logs/summarization/allsides/google/gemma-2-27b-it/random_medias_for_summarization/10/20250503_125850/result_449.csv"
    # df = pd.read_csv(df_dir)
    # predict_bias_mtsc_batch_optimized(df_dir)
    
    # ------------------------------------------------------------------------------------------------
    
    df_dir = "../../logs/summarization/allsides/google/gemma-2-27b-it/random_medias_for_summarization_center/5/20250519_022946/result_50.csv"
    df = pd.read_csv(df_dir)
    predict_bias_mtsc_batch_optimized(df_dir)
    
    