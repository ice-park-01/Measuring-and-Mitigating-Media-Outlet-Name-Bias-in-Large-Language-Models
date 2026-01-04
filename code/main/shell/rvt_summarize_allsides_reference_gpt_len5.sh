python scripts/inference.py \
--task "summarization_reference" \
--summary_length 5 \
--model_name "gpt-4.1" \
--closed_source True \
--dataset_name "allsides" \
--dataset_path "../../data/allsides/Article-Bias-Prediction/data/custom-split/random_medias_for_summarization_rvt_v2.json" \
--output_path "../../logs/" \
--cuda "2"