#CUDA_VISIBLE_DEVICES=0 python run_classifier.py --data_dir /home/enouri/projects/pytorch-text-classification/data/question_1  --do_train --do_eval --bert_model bert-base-uncased --num_train_epochs 20 --task_name cola --output_dir sent_classification

# CUDA_VISIBLE_DEVICES=1 python run_classifier.py --data_dir ./home/enouri/projects/pytorch-text-classification/data --do_train --do_eval --bert_model bert-base-uncased --num_train_epochs 20 --task_name cola --output_dir sent_classification


CUDA_VISIBLE_DEVICES=$1 python run_classifier.py --data_dir $2 --do_train --do_eval --bert_model bert-base-uncased --num_train_epochs $3 --task_name cola --output_dir checkpoints/sent_classification



#CUDA_VISIBLE_DEVICES=$1 python examples/run_classifier.py --data_dir $2 --do_train --do_eval --bert_model $3 --num_train_epochs $4 --task_name cola --output_dir checkpoints/sent_classification
