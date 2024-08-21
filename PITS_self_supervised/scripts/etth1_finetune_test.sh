log_dir="logs/FineTune_test"

# 디렉터리가 없으면 생성
if [ ! -d "$log_dir" ]; then
  mkdir -p "$log_dir"
fi

ds_pretrain='etth1'
ds_finetune='etth1'

# (1) Model Size
d_model=128

# (2) Input Size
context_points=512
patch_len=12 
stride=12
num_patches=42

# (3) Finetune Epoch
ep_ft_head=5
ep_ft_entire=10
seq_len=96

for pred_len in 96 192
  do 
    python PITS_finetune.py --dset_pretrain $ds_pretrain --dset_finetune $ds_finetune \
      --n_epochs_finetune_head $ep_ft_head --n_epochs_finetune_entire $ep_ft_entire --context_points $seq_len \
      --target_points $pred_len --num_patches $num_patches --context_points $context_points \
      --d_model $d_model --patch_len $patch_len --stride $stride \
        --is_finetune 1 >> $log_dir/PITS_$ds_pretrain'_'$seq_len'_'$pred_len'_0821.log'
  done