
MODEL=LSTM_${2}_${3}_${4}

CUDA_VISIBLE_DEVICES=$1 python main.py --embedding-size $2 --hidden-size $3 --layer $4 --classes $5 --cuda --data $6 --rnn LSTM --model $MODEL --multi_label 
