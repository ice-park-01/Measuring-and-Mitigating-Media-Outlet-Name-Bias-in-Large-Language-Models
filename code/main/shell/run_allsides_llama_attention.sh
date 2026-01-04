python scripts/check_attention_mask.py \
--model_name "meta-llama/Llama-3.1-8B" \
--media_outlet_type "allsides" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization.json" \
--output_path "../../logs_attention_heatmap/" \
--cuda "2"