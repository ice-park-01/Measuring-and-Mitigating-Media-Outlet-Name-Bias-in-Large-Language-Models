python scripts/inference.py \
--task "summarization" \
--summary_length 5 \
--model_name "meta-llama/Llama-3.3-70B-Instruct" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization_rvt_v2.json" \
--output_path "../../logs/" \
--cuda "0"