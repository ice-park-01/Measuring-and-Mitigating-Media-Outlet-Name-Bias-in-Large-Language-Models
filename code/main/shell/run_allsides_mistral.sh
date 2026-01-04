python scripts/inference.py \
--model_name "mistralai/Mistral-Small-24B-Instruct-2501" \
--media_outlet_type "generated" \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias.json" \
--output_path "../../logs/" \
--cuda "2"