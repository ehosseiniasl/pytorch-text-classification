

CUDA_VISIBLE_DEVICES=$1 python examples/run_classifier.py --data_dir $2 --do_eval --bert_model $3 --task_name cola --output_dir checkpoints/sent_classification
