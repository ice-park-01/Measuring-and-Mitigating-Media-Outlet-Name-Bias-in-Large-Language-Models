python scripts/inference.py \
--model_name "meta-llama/Llama-3.3-70B-Instruct" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization.json" \
--output_path "../../logs/" \
--cuda "2"