
GPU=$1
EPOCHS=$2
EMB=400
HDD=128
LAYER=2
DATA=data/elno_cleaned_data_pre
RNN=LSTM
CLASSES=2

MODEL=question1_lstm_${EMB}_${HDD}_${LAYER}

CUDA_VISIBLE_DEVICES=$GPU python main.py --embedding-size $EMB --hidden-size $HDD --layer $LAYER --classes $CLASSES --cuda --data $DATA --epochs $EPOCHS --rnn $RNN --model $MODEL --mean_seq 

