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

python -u PITS_pretrain.py \
  --dset_pretrain $ds_pretrain \
  --context_points $context_points \
  --d_model $d_model \
  --patch_len $patch_len \
  --stride $stride \
  --device_id 0
