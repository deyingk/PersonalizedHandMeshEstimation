phase='eval'
exp_name='reproduce_mano_our_model_with_gt_shape'
backbone='ResNet50'
dataset='dex_ycb'
dex_ycb_which_crop='size_2'
dex_ycb_which_split='s1'
dex_ycb_use_world_or_cam='cam'
batch_size=32
model='mano_based_model_iterative_pose'
mano_pose_comps=48
pose_agg_mode='multiply'
device_idx='0,1,2,3'
template_mode='groundtruth'
export CUDA_VISIBLE_DEVICES=$device_idx
python main_dex_ycb.py \
    --phase $phase \
    --exp_name $exp_name \
    --dataset $dataset \
    --batch_size $batch_size\
    --model $model \
    --backbone $backbone \
    --pose_agg_mode $pose_agg_mode\
    --mano_pose_comps $mano_pose_comps\
    --device_idx $device_idx \
    --template_mode $template_mode\
    --dex_ycb_which_crop $dex_ycb_which_crop\
    --dex_ycb_which_split $dex_ycb_which_split\
    --dex_ycb_use_world_or_cam $dex_ycb_use_world_or_cam\
    --resume 'checkpoint_last.pt'\
