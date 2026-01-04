python scripts/inference.py \
--model_name "meta-llama/Llama-3.1-8B" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization.json" \
--output_path "../../logs/" \
--cuda "0"
