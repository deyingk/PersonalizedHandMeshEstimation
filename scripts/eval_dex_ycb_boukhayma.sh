phase='eval'
exp_name='reproduce_boukhayma_model'
dataset='dex_ycb'
dex_ycb_which_crop='size_2'
dex_ycb_which_split='s1'
dex_ycb_use_world_or_cam='cam'
batch_size=32
model='boukhayma_model'
shape_feature_out_c=256
mano_pose_comps=48
device_idx='0'
size=256
export CUDA_VISIBLE_DEVICES=$device_idx
python main_dex_ycb.py \
    --size $size \
    --phase $phase \
    --exp_name $exp_name \
    --dataset $dataset \
    --batch_size $batch_size\
    --model $model \
    --shape_feature_out_c $shape_feature_out_c\
    --mano_pose_comps $mano_pose_comps\
    --device_idx $device_idx \
    --dex_ycb_which_crop $dex_ycb_which_crop\
    --dex_ycb_which_split $dex_ycb_which_split\
    --dex_ycb_use_world_or_cam $dex_ycb_use_world_or_cam\
    --resume 'checkpoint_last_old.pt'\