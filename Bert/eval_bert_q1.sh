

CUDA_VISIBLE_DEVICES=$1 python run_bert_single.py --data_dir $2 --do_eval --bert_model $3 --task_name cola --cache_dir checkpoints/sent_classification --output_dir checkpoints/sent_classification
