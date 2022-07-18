phase='train'
exp_name='reproduce_2d_model'
backbone='ResNet18'
dataset='dex_ycb'
dex_ycb_which_crop='size_2'
dex_ycb_which_split='s1'
dex_ycb_use_world_or_cam='cam'
batch_size=32
model='cmr_2d_pose'
device_idx='0,1,2,3'
epochs=15
decay_step_1=10
decay_step_2=15
lr_decay=0.1
export CUDA_VISIBLE_DEVICES=$device_idx
python main_dex_ycb_2d.py \
    --phase $phase \
    --exp_name $exp_name \
    --dataset $dataset \
    --batch_size $batch_size\
    --model $model \
    --backbone $backbone \
    --device_idx $device_idx \
    --dex_ycb_which_crop $dex_ycb_which_crop\
    --dex_ycb_which_split $dex_ycb_which_split\
    --dex_ycb_use_world_or_cam $dex_ycb_use_world_or_cam\
    --epochs $epochs\
    --decay_step $decay_step_1 $decay_step_2\
    --lr_decay $lr_decay\