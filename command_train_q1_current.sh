
GPU=$1
EPOCHS=$2
EMB=400
HDD=128
LAYER=2
#DATA=data/elno_cleaned_data_current
DATA=data/Q1-yesno-elno_cleaned_data_current
RNN=LSTM
CLASSES=2
BATCH=16
MODEL=elno_cleaned_data_current_lstm_${EMB}_${HDD}_${LAYER}

CUDA_VISIBLE_DEVICES=$GPU python main.py --embedding-size $EMB --hidden-size $HDD --layer $LAYER --classes $CLASSES --cuda --data $DATA --epochs $EPOCHS --rnn $RNN --model $MODEL --batch-size $BATCH --use_glove --glove $3
#--mean_seq 
