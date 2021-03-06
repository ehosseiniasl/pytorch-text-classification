
GPU=$1
EPOCHS=$2
EMB=400
HDD=128
LAYER=2
#DATA=data/elno_cleaned_data_current
DATA=data/Q1-format-elno_cleaned_data_after
RNN=LSTM
CLASSES=5
BATCH=128
MODEL=elno_cleaned_data_current_lstm_${EMB}_${HDD}_${LAYER}

CUDA_VISIBLE_DEVICES=$GPU python main.py --embedding-size $EMB --hidden-size $HDD --layer $LAYER --classes $CLASSES --cuda --data $DATA --epochs $EPOCHS --rnn $RNN --model $MODEL --multi_label --batch-size $BATCH --use_glove --glove $3 
#--mean_seq 

