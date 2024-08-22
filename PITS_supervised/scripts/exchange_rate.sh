if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

export CUDA_VISIBLE_DEVICES=0  # 첫 번째 GPU만 사용하도록 설정
seq_len=96
model_name=PITS
TORCH_USE_CUDA_DSA=0
root_path_name=./dataset/exchange_rate/
data_path_name=exchange_rate.csv
model_id_name=pits_exchange_0822
data_name=custom

random_seed=2023
for pred_len in 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --c_in 8 \
      --d_model 128 \
      --patch_len 16\
      --stride 8\
      --train_epochs 25\
      --patience 5\
      --itr 1 --batch_size 128 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done