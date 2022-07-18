phase='train'
exp_name='reproduce_cmr_pg'
backbone='ResNet18'
dataset='dex_ycb'
dex_ycb_which_crop='size_2'
dex_ycb_which_split='s1'
dex_ycb_use_world_or_cam='cam'
model='cmr_pg_pp'
device_idx='7,6,5,4'
epochs=15
decay_step_1=10
decay_step_2=12
lr_decay=0.1
export CUDA_VISIBLE_DEVICES=$device_idx
python main_dex_ycb.py \
    --phase $phase \
    --exp_name $exp_name \
    --dataset $dataset \
    --model $model \
    --backbone $backbone \
    --device_idx $device_idx \
    --epochs $epochs\
    --decay_step $decay_step_1 $decay_step_2\
    --dex_ycb_which_crop $dex_ycb_which_crop\
    --dex_ycb_use_world_or_cam $dex_ycb_use_world_or_cam\
    --dex_ycb_which_split $dex_ycb_which_split\
    --lr_decay $lr_decay\
