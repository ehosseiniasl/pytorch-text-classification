
GPU=$1
EPOCHS=$2
EMB=400
HDD=128
LAYER=2
DATA=data/question_2
RNN=LSTM
CLASSES=6

MODEL=question2_lstm_${EMB}_${HDD}_${LAYER}
#MODEL=question2_lstm_${2}_${3}_${4}

CUDA_VISIBLE_DEVICES=$GPU python main.py --embedding-size $EMB --hidden-size $HDD --layer $LAYER --classes $CLASSES --cuda --data $DATA --rnn $RNN --model $MODEL  --epochs $EPOCHS --multi_label --mean_seq 

