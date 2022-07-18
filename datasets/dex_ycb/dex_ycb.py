from collections import defaultdict
import torch
import torch.utils.data as data
from utils.humbi_utils import *
from utils.vis import base_transform, inv_base_tranmsform, uv2map
from termcolor import cprint
from utils.vis import crop_roi
from utils.augmentation import Augmentation
from src.network import Pool
import pickle
import cv2
import skimage.io as io
import random

random.seed(0)

class DEX_YCB(data.Dataset):
    def __init__(self, root, phase, args, faces, writer=None, down_sample_list=None, img_std=0.5, img_mean=0.5, ms=True, which_split='s0'):
        super(DEX_YCB, self).__init__()
        self.root = root
        self.phase = phase
        self.down_sample_list = down_sample_list
        self.size = args.size
        self.faces = faces
        self.img_std = img_std
        self.img_mean = img_mean
        self.ms = ms
        self.which_split = which_split
        self.which_crop = args.dex_ycb_which_crop
        self.color_aug = Augmentation(size=self.size) if args.color_aug and 'train' in self.phase else None
        self.std = torch.tensor(0.20)
        self.template_mode = args.template_mode
        self.use_small_set = args.humbi_use_small
        self.use_world_or_cam = args.dex_ycb_use_world_or_cam
        assert self.use_world_or_cam in ('world', 'cam')
        if args.template_mode not in ('groundtruth', 'mean_shape','calibrated','random'):
            raise "Not implemented for this specific template mode!"

        print('dataset template mode is ', self.template_mode)
        print('dataset using crop version of ', self.which_crop)
        print('dataset using split version of ', self.which_split)
        print('{} coordinate annotations are being used'.format(self.use_world_or_cam))
        # get data annotation list
        train_file_name = self.which_split+'_train_'+self.which_crop+'.pkl'
        val_file_name = self.which_split+'_val_' +self.which_crop+'.pkl'
        test_file_name = self.which_split+'_test_'+self.which_crop+'.pkl'
        
        if 'train' in self.phase:
            with open(os.path.join(self.root, 'split_annotations', train_file_name), 'rb') as f:
                self.db_data_anno = pickle.load(f)
        elif 'val' in self.phase:
            with open(os.path.join(self.root, 'split_annotations', val_file_name), 'rb') as f:
                self.db_data_anno = pickle.load(f)
        elif 'test' in self.phase:
            with open(os.path.join(self.root, 'split_annotations', test_file_name), 'rb') as f:
                self.db_data_anno = pickle.load(f)
        else:
            raise "phase setting for the dataset is incorrect!!" 

        if args.template_mode == 'groundtruth':
            with open(os.path.join(self.root, 'split_annotations', 'unique_shapes_dex_ycb.pkl'), 'rb') as f:
                self.hand_shapes = pickle.load(f)
        elif args.template_mode == 'calibrated':
            print('loading calibrated hand shape ....')
            with open('out/dex_ycb/reproduce_train_conf_branch/calibrated_shapes/calibrated_randomly_20_shape_all_subjects.pkl','rb') as f:
                self.hand_shapes = pickle.load(f)        
        # get template mesh for all the subjects
        if args.template_mode == 'groundtruth':
            with open(os.path.join(self.root, 'split_annotations', 'rest_pose_template.pkl'), 'rb') as f:
                self.rest_pose_templates = pickle.load(f)
        elif args.template_mode == 'calibrated':
            print('loading calibrated rest hand meshes....')
            with open('out/dex_ycb/reproduce_train_conf_branch/calibrated_shapes/calibrated_randomly_20_rest_pose_template.pkl','rb') as f:
                self.rest_pose_templates = pickle.load(f)

        if 'train' in self.phase and args.template_mode == 'calibrated':
            raise "Wrong configuration. Calibrated template should not be used in training phase!"     
        cprint('Loaded DEX_YCB {} {} samples'.format(self.phase, str(len(self.db_data_anno))), 'red')
        

    def __getitem__(self, idx):
        if 'train' in self.phase:
            return self.get_training_sample(idx)
        elif 'val' in self.phase or 'test' in self.phase:
            return self.get_eval_sample(idx)
        else:
            raise Exception('phase error')

    def get_training_sample(self, idx):
        anno_item = self.db_data_anno[idx]
        image_rela_path = anno_item['cropped_img_path'][19:]
        uv = anno_item['cropped_joint_2d']
        xyz_provided = anno_item['joint_3d']
        if self.use_world_or_cam == 'cam':
            xyz = anno_item['regressed_joint_3d']
            v0 = anno_item['mesh']
            mano = anno_item['mano_params']
        elif self.use_world_or_cam == 'world':
            xyz = anno_item['world_regressed_joint_3d']
            v0 = anno_item['world_mesh']
            mano = anno_item['world_mano_params']

        subject_id = int(image_rela_path.split('/')[1].split('-')[-1])

        img = io.imread(os.path.join(self.root, image_rela_path))
        orig_size = img.shape[0]
        
        if self.color_aug is not None:
            img = self.color_aug(img)
        img = base_transform(img, size=self.size, mean=self.img_mean, std=self.img_std)
        uv[:, 0] = uv[:, 0] * self.size/orig_size
        uv[:, 1] = uv[:, 1] * self.size/orig_size        
        
        uv_map = uv2map(uv.astype(np.int), img.shape[1:])
        uv_map = cv2.resize(uv_map.transpose(1, 2, 0), (img.shape[2]//2, img.shape[1]//2)).transpose(2, 0, 1)
        img, xyz, uv, uv_map, v0, mano, xyz_provided = [torch.from_numpy(x).float() for x in [img, xyz, uv, uv_map, v0, mano, xyz_provided]]

        xyz_root = xyz[0]
        xyz = (xyz - xyz_root) / self.std

        translation = mano[48:51]
        # xyz_provided = xyz_provided - xyz_provided[0]
        # print('xyz_provided', xyz_provided)
        # print('translation', translation)
        xyz_provided = (xyz_provided - translation[np.newaxis, :]) / self.std

        v0 = (v0 - translation[np.newaxis, :]) / self.std
        
        if self.ms:
            v1 = Pool(v0.unsqueeze(0), self.down_sample_list[0])[0]
            v2 = Pool(v1.unsqueeze(0), self.down_sample_list[1])[0]
            v3 = Pool(v2.unsqueeze(0), self.down_sample_list[2])[0]
            gt = [v0, v1, v2, v3]
        else:
            gt = [v0, ]

        # get rest_pose_template 
        if self.template_mode == 'groundtruth':
            template_v0 = torch.from_numpy(self.rest_pose_templates[subject_id]).float() / self.std
        elif self.template_mode == 'mean_shape':
            template_v0 = torch.from_numpy(self.rest_pose_templates['mean_shape']).float() / self.std
        if self.ms:
            template_v1 = Pool(template_v0.unsqueeze(0), self.down_sample_list[0])[0]
            template_v2 = Pool(template_v1.unsqueeze(0), self.down_sample_list[1])[0]
            template_v3 = Pool(template_v2.unsqueeze(0), self.down_sample_list[2])[0]
            template_v4 = Pool(template_v3.unsqueeze(0), self.down_sample_list[3])[0]
            rest_pose_template = [template_v0, template_v1, template_v2, template_v3, template_v4][::-1]
        else:
            rest_pose_template = [template_v0]
        
        data = {'img': img,
                'mesh_gt': gt,
                'mesh_template':rest_pose_template,
                'xyz_gt': xyz,
                'xyz_gt_provided':xyz_provided,
                'uv_point': uv,
                'uv_gt': uv_map,
                'xyz_root': xyz_root,
                'shape_params': torch.from_numpy(self.hand_shapes[subject_id]).float(),
                'pose_params': mano[:48],
                'mano_translation':mano[48:51],
                }
        data['meta'] = image_rela_path
        return data

    def get_eval_sample(self, idx):
        anno_item = self.db_data_anno[idx]
        image_rela_path = anno_item['cropped_img_path'][19:]
        uv = anno_item['cropped_joint_2d']
        xyz_provided = anno_item['joint_3d']

        if self.use_world_or_cam == 'cam':
            xyz = anno_item['regressed_joint_3d']
            v0 = anno_item['mesh']
            mano = anno_item['mano_params']
        elif self.use_world_or_cam == 'world':
            xyz = anno_item['world_regressed_joint_3d']
            v0 = anno_item['world_mesh']
            mano = anno_item['world_mano_params']

        subject_id = int(image_rela_path.split('/')[1].split('-')[-1])

        img = io.imread(os.path.join(self.root, image_rela_path))
        orig_size = img.shape[0]

        img = base_transform(img, size=self.size, mean=self.img_mean, std=self.img_std)
        uv[:, 0] = uv[:, 0] * self.size/orig_size
        uv[:, 1] = uv[:, 1] * self.size/orig_size    
        
        uv_map = uv2map(uv.astype(np.int), img.shape[1:])
        uv_map = cv2.resize(uv_map.transpose(1, 2, 0), (img.shape[2]//2, img.shape[1]//2)).transpose(2, 0, 1)
        img, xyz, uv, uv_map, v0, mano, xyz_provided = [torch.from_numpy(x).float() for x in [img, xyz, uv, uv_map, v0, mano, xyz_provided]]

        xyz_root = xyz[0]
        xyz = (xyz - xyz_root) / self.std

        # xyz_provided = xyz_provided - xyz_provided[0]        
        translation = mano[48:51]
        xyz_provided = (xyz_provided - translation[np.newaxis, :]) / self.std
        v0 = (v0 - translation[np.newaxis, :]) / self.std
        
        if self.ms:
            v1 = Pool(v0.unsqueeze(0), self.down_sample_list[0])[0]
            v2 = Pool(v1.unsqueeze(0), self.down_sample_list[1])[0]
            v3 = Pool(v2.unsqueeze(0), self.down_sample_list[2])[0]
            gt = [v0, v1, v2, v3]
        else:
            gt = [v0, ]

        # get rest_pose_template 
        if self.template_mode == 'groundtruth':
            template_v0 = torch.from_numpy(self.rest_pose_templates[subject_id]).float() / self.std
        elif self.template_mode == 'calibrated':
            template_v0 = torch.from_numpy(self.rest_pose_templates[subject_id]).float() / self.std
        elif self.template_mode == 'random':
            template_v0 = torch.from_numpy(self.rest_pose_templates[subject_id]).float() / self.std 
        elif self.template_mode == 'mean_shape':
            template_v0 = torch.from_numpy(self.rest_pose_templates['mean_shape']).float() / self.std
        if self.ms:
            template_v1 = Pool(template_v0.unsqueeze(0), self.down_sample_list[0])[0]
            template_v2 = Pool(template_v1.unsqueeze(0), self.down_sample_list[1])[0]
            template_v3 = Pool(template_v2.unsqueeze(0), self.down_sample_list[2])[0]
            template_v4 = Pool(template_v3.unsqueeze(0), self.down_sample_list[3])[0]
            rest_pose_template = [template_v0, template_v1, template_v2, template_v3, template_v4][::-1]
        else:
            rest_pose_template = [template_v0]

        data = {'img': img,
                'mesh_gt': gt,
                'mesh_template':rest_pose_template,
                'xyz_gt': xyz,
                'xyz_gt_provided':xyz_provided,
                'uv_point': uv,
                'uv_gt': uv_map,
                'xyz_root': xyz_root,
                'shape_params': torch.from_numpy(self.hand_shapes[subject_id]).float(),
                'pose_params': mano[:48],
                'mano_translation':mano[48:51],
                }
        data['meta'] = image_rela_path
        data['crop_bbox'] = torch.from_numpy(anno_item['crop_bbox'])
        return data

    def __len__(self):

        return len(self.db_data_anno)

    def get_face(self):
        return self.faces




if __name__ == '__main__':
    import pickle
    import utils
    from utils import utils
    from options.base_options import BaseOptions
    import cv2
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    args = BaseOptions().parse()
    with open('template/transform.pkl', 'rb') as f:
        tmp = pickle.load(f, encoding='latin1')

    down_transform_list = [
        utils.to_sparse(down_transform)
        for down_transform in tmp['down_transform']
    ]

    args.phase = 'val'
    args.size = 224
    args.work_dir = './'

    if True:
        dataset = DEX_YCB('data/dex_ycb', args.phase, args, tmp['face'], writer=None,
                        down_sample_list=down_transform_list, img_mean=args.img_mean, img_std=args.img_std, ms=args.ms_mesh)

        data = dataset.get_training_sample(4)
        print(data['meta'])
        print(data['img'].shape)
        print(data['uv_gt'].shape)
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(inv_base_tranmsform(data['img'].numpy())[:, :, ::-1])
        ax[0].scatter(data['uv_point'].numpy()[:, 0], data['uv_point'].numpy()[:, 1])
        ax[1].imshow(np.sum(data['uv_gt'].numpy(), axis=0))
        fig.savefig('datasets/dex_ycb/test.png')
    

