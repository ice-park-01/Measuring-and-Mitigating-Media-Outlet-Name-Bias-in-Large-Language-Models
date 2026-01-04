python scripts/inference.py \
--model_name "meta-llama/Llama-3.1-8B-Instruct" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias.json" \
--output_path "../../logs/" \
--cuda "0"

python scripts/inference.py \
--model_name "meta-llama/Meta-Llama-3-8B-Instruct" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias.json" \
--output_path "../../logs/" \
--cuda "0"