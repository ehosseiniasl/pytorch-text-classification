
GPU=$1
EPOCHS=$2
EMB=400
HDD=128
LAYER=2
#DATA=data/question_3
DATA=data/Q3-action-elno_cleaned_data_after
RNN=LSTM
CLASSES=7

MODEL=question2_lstm_${EMB}_${HDD}_${LAYER}
#MODEL=question3_lstm_${2}_${3}_${4}

CUDA_VISIBLE_DEVICES=$GPU python main.py --embedding-size $EMB --hidden-size $HDD --layer $LAYER --classes $CLASSES --cuda --data $DATA --rnn $RNN --model $MODEL --multi_label --mean_seq 

