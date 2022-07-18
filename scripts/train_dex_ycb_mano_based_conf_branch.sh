phase='train'
# exp_name='mano_based_model_iterative_pose_conf_2nd_stage_crop_2_s1_cam_epochs_5_10_15_both_keypoints_and_pose_loss_resnet50_multiply_ranking_loss_weight_1_margine_1_layers_1_bs_128'
exp_name='reproduce_train_conf_branch'
backbone='ResNet50'
dataset='dex_ycb'
dex_ycb_which_crop='size_2'
dex_ycb_which_split='s1'
dex_ycb_use_world_or_cam='cam'
batch_size=128
model='mano_based_model_iterative_pose_without_gt_shape_with_conf'
pretrained_mano_based_model_iterative_pose_without_gt_shape_with_conf_dir="out/dex_ycb/reproduce_mano_baseline/checkpoints"
mano_based_conf_second_stage_train_conf_branch='yes'
conf_hidden_layers=1
conf_weight_scalar=1.0
ranking_loss_margin=1.0
conf_hidden_layers_drop_out='no'
conf_use_sigmoid='no'
shape_feature_out_c=256
mano_pose_comps=48
pose_agg_mode='multiply'
device_idx='6'
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
    --pretrained_mano_based_model_iterative_pose_without_gt_shape_with_conf_dir $pretrained_mano_based_model_iterative_pose_without_gt_shape_with_conf_dir\
    --shape_feature_out_c $shape_feature_out_c\
    --pose_agg_mode $pose_agg_mode\
    --mano_pose_comps $mano_pose_comps\
    --backbone $backbone \
    --device_idx $device_idx \
    --dex_ycb_which_crop $dex_ycb_which_crop\
    --dex_ycb_which_split $dex_ycb_which_split\
    --dex_ycb_use_world_or_cam $dex_ycb_use_world_or_cam\
    --conf_weight_scalar $conf_weight_scalar\
    --epochs $epochs\
    --decay_step $decay_step_1 $decay_step_2\
    --lr_decay $lr_decay\
    --ranking_loss_margin $ranking_loss_margin\
    --mano_based_conf_second_stage_train_conf_branch $mano_based_conf_second_stage_train_conf_branch\
    --conf_hidden_layers $conf_hidden_layers\
    --conf_hidden_layers_drop_out $conf_hidden_layers_drop_out\
    --conf_use_sigmoid $conf_use_sigmoid\
