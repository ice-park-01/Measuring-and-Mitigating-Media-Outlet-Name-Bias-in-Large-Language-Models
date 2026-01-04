python scripts/check_attention_mask.py \
--model_name "Qwen/Qwen2.5-7B-Instruct" \
--media_outlet_type "allsides" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization.json" \
--output_path "../../logs_attention_heatmap/" \
--cuda "1"