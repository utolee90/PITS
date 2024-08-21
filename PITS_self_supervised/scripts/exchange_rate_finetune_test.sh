log_dir="logs/FineTune_test"

# 디렉터리가 없으면 생성
if [ ! -d "$log_dir" ]; then
  mkdir -p "$log_dir"
fi

ds_pretrain='exchange'
ds_finetune='exchange'

# (1) Model Size
d_model=128

# (2) Input Size
context_points=96 # 입력 사이즈
patch_len=16 # 조각 길이
stride=8 # 분할 길이
num_patches=11 # 패치 수

# (3) Finetune Epoch
ep_ft_head=5
ep_ft_entire=10

for pred_len in 96 192
  do 
    python PITS_finetune.py --dset_pretrain $ds_pretrain --dset_finetune $ds_finetune \
      --n_epochs_finetune_head $ep_ft_head --n_epochs_finetune_entire $ep_ft_entire \
      --target_points $pred_len --num_patches $num_patches --context_points $context_points \
      --d_model $d_model --patch_len $patch_len --stride $stride \
        --is_finetune 1 >> $log_dir/PITS_$ds_pretrain'_'$context_points'_'$pred_len'_0821.log'
  done