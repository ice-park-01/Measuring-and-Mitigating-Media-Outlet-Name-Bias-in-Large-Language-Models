python scripts/inference.py \
--model_name "meta-llama/Llama-2-7b-chat-hf" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias.json" \
--output_path "../../logs/" \
--cuda "2"

python scripts/inference.py \
--model_name "meta-llama/Llama-2-13b-chat-hf" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias.json" \
--output_path "../../logs/" \
--cuda "2"

python scripts/inference.py \
--model_name "meta-llama/Llama-2-70b-chat-hf" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias.json" \
--output_path "../../logs/" \
--cuda "2"