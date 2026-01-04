python scripts/inference.py \
--model_name "meta-llama/Llama-3.3-70B-Instruct" \
--media_outlet_type "generated" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias.json" \
--output_path "../../logs/" \
--cuda "2"

python scripts/inference.py \
--model_name "microsoft/phi-4" \
--media_outlet_type "generated" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias.json" \
--output_path "../../logs/" \
--cuda "2"