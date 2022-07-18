# phase='train'
# exp_name='mano_based_model_baseline_without_gt_shape_crop_2_s1_cam_epochs_5_10_15_both_mesh_and_pose_loss_resnet50_multiply'
# # exp_name='play_around'
# backbone='ResNet50'
# dataset='dex_ycb'
# dex_ycb_which_crop='size_2'
# dex_ycb_which_split='s1'
# dex_ycb_use_world_or_cam='cam'
# batch_size=32
# model='mano_based_model_baseline_without_gt_shape'
# shape_feature_out_c=256
# mano_pose_comps=48
# pose_agg_mode='multiply'
# with_mesh_loss='yes'
# with_keypoints_loss='no'
# only_mesh_loss='no'
# device_idx='0,1,2,3'
# use_rotate='no'
# epochs=15
# decay_step_1=5
# decay_step_2=10
# lr_decay=0.1
# export CUDA_VISIBLE_DEVICES=$device_idx
# python main_dex_ycb.py \
#     --phase $phase \
#     --exp_name $exp_name \
#     --dataset $dataset \
#     --batch_size $batch_size\
#     --model $model \
#     --shape_feature_out_c $shape_feature_out_c\
#     --pose_agg_mode $pose_agg_mode\
#     --with_keypoints_loss $with_keypoints_loss\
#     --with_mesh_loss $with_mesh_loss\
#     --mano_pose_comps $mano_pose_comps\
#     --backbone $backbone \
#     --device_idx $device_idx \
#     --dex_ycb_which_crop $dex_ycb_which_crop\
#     --dex_ycb_which_split $dex_ycb_which_split\
#     --dex_ycb_use_world_or_cam $dex_ycb_use_world_or_cam\
#     --epochs $epochs\
#     --decay_step $decay_step_1 $decay_step_2\
#     --lr_decay $lr_decay\


# phase='train'
# exp_name='mano_based_model_iterative_pose_3x_crop_2_s1_cam_epochs_5_10_15_both_provided_keypoints_and_pose_loss_resnet50_multiply'
# # exp_name='play_around'
# backbone='ResNet50'
# dataset='dex_ycb'
# dex_ycb_which_crop='size_2'
# dex_ycb_which_split='s1'
# dex_ycb_use_world_or_cam='cam'
# batch_size=32
# model='mano_based_model_iterative_pose'
# shape_feature_out_c=256
# mano_pose_comps=48
# pose_agg_mode='multiply'
# with_mesh_loss='no'
# with_keypoints_loss='yes'
# only_mesh_loss='no'
# device_idx='3,2,1,0'
# use_rotate='no'
# epochs=15
# decay_step_1=5
# decay_step_2=10
# lr_decay=0.1
# export CUDA_VISIBLE_DEVICES=$device_idx
# python main_dex_ycb.py \
#     --phase $phase \
#     --exp_name $exp_name \
#     --dataset $dataset \
#     --batch_size $batch_size\
#     --model $model \
#     --shape_feature_out_c $shape_feature_out_c\
#     --pose_agg_mode $pose_agg_mode\
#     --with_keypoints_loss $with_keypoints_loss\
#     --with_mesh_loss $with_mesh_loss\
#     --mano_pose_comps $mano_pose_comps\
#     --backbone $backbone \
#     --device_idx $device_idx \
#     --dex_ycb_which_crop $dex_ycb_which_crop\
#     --dex_ycb_which_split $dex_ycb_which_split\
#     --dex_ycb_use_world_or_cam $dex_ycb_use_world_or_cam\
#     --epochs $epochs\
#     --decay_step $decay_step_1 $decay_step_2\
#     --lr_decay $lr_decay\



phase='train'
# exp_name='mano_based_model_iterative_pose_without_gt_shape_crop_2_s11_cam_epochs_5_10_15_both_mesh_and_pose_loss_resnet50_multiply'
exp_name='reproduce_mano_baseline'
# exp_name='play_around'
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
use_rotate='no'
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