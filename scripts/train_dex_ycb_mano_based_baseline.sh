phase='train'
exp_name='reproduce_mano_baseline'
backbone='ResNet50'
dataset='dex_ycb'
dex_ycb_which_crop='size_2'
dex_ycb_which_split='s1'
dex_ycb_use_world_or_cam='cam'
batch_size=32
model='mano_based_model_iterative_pose_without_gt_shape'
shape_feature_out_c=256
mano_pose_comps=48
pose_agg_mode='multiply'
device_idx='0,1,2,3'
epochs=15
decay_step_1=5
decay_step_2=10
lr_decay=0.1
export CUDA_VISIBLE_DEVICES=$device_idx
python main_dex_ycb.py \
    --phase $phase \
    --exp_name $exp_name \
    --dataset $dataset \
    --batch_size $batch_size\
    --model $model \
    --shape_feature_out_c $shape_feature_out_c\
    --pose_agg_mode $pose_agg_mode\
    --mano_pose_comps $mano_pose_comps\
    --backbone $backbone \
    --device_idx $device_idx \
    --dex_ycb_which_crop $dex_ycb_which_crop\
    --dex_ycb_which_split $dex_ycb_which_split\
    --dex_ycb_use_world_or_cam $dex_ycb_use_world_or_cam\
    --epochs $epochs\
    --decay_step $decay_step_1 $decay_step_2\
    --lr_decay $lr_decay\