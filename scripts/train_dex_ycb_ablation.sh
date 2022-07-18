
phase='train'
exp_name='cmr_pg_pp_ablation_crop_2_s1_cam_epochs_5_10_15'
# exp_name='play_around'
backbone='ResNet18'
dataset='dex_ycb'
dex_ycb_which_crop='size_2'
dex_ycb_which_split='s1'
dex_ycb_use_world_or_cam='cam'
batch_size=32
model='cmr_pg_pp_ablation'
device_idx='0,1,2,3'
template_mode='groundtruth'
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
    --backbone $backbone \
    --device_idx $device_idx \
    --template_mode $template_mode\
    --dex_ycb_which_crop $dex_ycb_which_crop\
    --dex_ycb_which_split $dex_ycb_which_split\
    --dex_ycb_use_world_or_cam $dex_ycb_use_world_or_cam\
    --epochs $epochs\
    --decay_step $decay_step_1 $decay_step_2\
    --lr_decay $lr_decay\


# phase='train'
# exp_name='cmr_pg_pp_ia_split_1_full_dataset'
# # exp_name='play_around'
# backbone='ResNet18'
# dataset='dex_ycb'
# batch_size=32
# model='cmr_pg_pp_ia'
# device_idx='7,6,5,4'
# template_mode='groundtruth'
# export CUDA_VISIBLE_DEVICES=$device_idx
# python main_dex_ycb.py \
#     --phase $phase \
#     --exp_name $exp_name \
#     --dataset $dataset \
#     --batch_size $batch_size\
#     --model $model \
#     --backbone $backbone \
#     --device_idx $device_idx \
#     --template_mode $template_mode\
