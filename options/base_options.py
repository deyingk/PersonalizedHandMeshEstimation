import argparse

class BaseOptions():
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--exp_name', type=str, default='test')
        parser.add_argument('--n_threads', type=int, default=4)
        parser.add_argument('--device_idx', type=str, default='cpu')

        # dataset hyperparameters
        parser.add_argument('--dataset', type=str, default='dex_ycb')
        parser.add_argument('--use_rotate', type=self.str2bool, default='no')
        parser.add_argument('--color_aug', type=self.str2bool, default='yes')
        parser.add_argument('--flip_aug', type=self.str2bool, default='no')
        parser.add_argument('--size', type=int, default=224)
        parser.add_argument('--img_mean', type=float, default=0.5)
        parser.add_argument('--img_std', type=float, default=0.5)
        parser.add_argument('--ms_mesh', type=self.str2bool, default='yes')
        parser.add_argument('--group_size', type=int, default=8)
        parser.add_argument('--humbi_which_split', type=str, default='split_1')
        parser.add_argument('--dex_ycb_which_split', type=str, default='s1')
        parser.add_argument('--dex_ycb_which_crop', type=str, default='size_2')
        parser.add_argument('--dex_ycb_use_world_or_cam', type=str, default='cam')
        parser.add_argument('--template_mode', type=str, default='groundtruth')



        # network hyperparameters
        parser.add_argument('--out_channels', nargs='+', default=[64, 128, 256, 512], type=int)
        parser.add_argument('--ds_factors', nargs='+', default=[2, 2, 2, 2], type=float)
        parser.add_argument('--in_channels', type=int, default=3)
        parser.add_argument('--seq_length', type=int, default=[27, 27, 27, 27], nargs='+')
        parser.add_argument('--dilation', type=int, default=[1, 1, 1, 1], nargs='+')
        parser.add_argument('--model', type=str, default='cmr_sg')
        parser.add_argument('--backbone', type=str, default='ResNet18')
        parser.add_argument('--mano_pose_comps', type=int, default=20)
        parser.add_argument('--bn', type=self.str2bool, default='no')
        parser.add_argument('--att', type=self.str2bool, default='no')
        parser.add_argument('--use_pretrained_baseline', type=self.str2bool, default='no')
        parser.add_argument('--baseline_model_dir', type=str, default='')

        # for iterative pose estimation
        parser.add_argument('--iteration', type=int, default=3)
        parser.add_argument('--shape_feature_out_c', type=int, default=256)
        parser.add_argument('--img_feature_out_c', type=int, default=256)
        parser.add_argument('--only_last_iteration_loss', type=self.str2bool, default='no')
        parser.add_argument('--pose_agg_mode', type=str, default='add')
        
        # For confidence models
        parser.add_argument('--conf_weight_scalar', type=float, default=10.0)
        parser.add_argument('--use_ranking_loss', type=self.str2bool, default='yes')  
        parser.add_argument('--user_wise_rank_loss', type=self.str2bool, default='no')  
        parser.add_argument('--conf_use_sigmoid', type=self.str2bool, default='no')  
        parser.add_argument('--ranking_loss_margin', type=float, default='1.0')
        parser.add_argument('--conf_hidden_layers', type=int, default=3)
        parser.add_argument('--conf_hidden_layers_drop_out', type=self.str2bool, default='no')

        # For mano-based confidence models
        parser.add_argument('--pretrained_mano_based_model_iterative_pose_without_gt_shape_with_conf_dir', type=str, default='')
        parser.add_argument('--mano_based_conf_second_stage_train_conf_branch', type=self.str2bool, default='no')

        # optimizer hyperparmeters
        parser.add_argument('--optimizer', type=str, default='Adam')
        parser.add_argument('--lr_scheduled', type=str, default='MultiStep')
        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--lr_decay', type=float, default=0.1)
        parser.add_argument('--decay_step', type=int, nargs='+', default=[30, ])
        parser.add_argument('--weight_decay', type=float, default=0)

        # training hyperparameters
        parser.add_argument('--phase', type=str, default='train')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--epochs', type=int, default=38)
        parser.add_argument('--resume', type=str, default='')
        parser.add_argument('--fine_tune', type=self.str2bool, default='no')
        parser.add_argument('--pretrained_dir', type=str, default=' ')


        self.initialized = True
        return parser

    def str2bool(self, v):
        return v.lower() in ("yes", "true", "t", "1")

    def parse(self):

        parser = argparse.ArgumentParser(description='mesh generator')
        self.initialize(parser)
        args = parser.parse_args()

        return args
