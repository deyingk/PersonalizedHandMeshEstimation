# phase='train'
# exp_name='cmr_pg_no_mask_crop_2_s0_world_fine_tune_5_8_10'
# # exp_name='play_around'
# backbone='ResNet18'
# dataset='dex_ycb'
# dex_ycb_which_crop='size_2'
# dex_ycb_which_split='s0'
# dex_ycb_use_world_or_cam='world'
# model='cmr_pg_pp'
# device_idx='0,1,2,3'
# epochs=10
# decay_step_1=5
# decay_step_2=8
# lr_decay=0.1
# fine_tune='yes'
# pretrained_dir='/home/deyingk/projects/HandMesh/out/humbi/cmr_pg_no_mask_full_dataset_epochs_25_30/checkpoints/checkpoint_best.pt'
# export CUDA_VISIBLE_DEVICES=$device_idx
# python main_dex_ycb.py \
#     --phase $phase \
#     --exp_name $exp_name \
#     --dataset $dataset \
#     --model $model \
#     --backbone $backbone \
#     --device_idx $device_idx \
#     --epochs $epochs\
#     --decay_step $decay_step_1 $decay_step_2\
#     --dex_ycb_which_crop $dex_ycb_which_crop\
#     --dex_ycb_use_world_or_cam $dex_ycb_use_world_or_cam\
#     --dex_ycb_which_split $dex_ycb_which_split\
#     --lr_decay $lr_decay\
#     --fine_tune $fine_tune\
#     --pretrained_dir $pretrained_dir\


phase='train'
# exp_name='cmr_pg_no_mask_crop_2_s10_cam_epochs_10_15_20'
exp_name='train_confidence_branch'
backbone='ResNet18'
dataset='dex_ycb'
dex_ycb_which_crop='size_2'
dex_ycb_which_split='s1'
dex_ycb_use_world_or_cam='cam'
model='cmr_pg_pp_conf'
device_idx='0,1,2,3'
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