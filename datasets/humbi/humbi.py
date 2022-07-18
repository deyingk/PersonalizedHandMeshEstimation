from collections import defaultdict
import torch
import torch.utils.data as data
from utils.humbi_utils import *
from utils.vis import base_transform, inv_base_tranmsform, cnt_area, uv2map
from termcolor import cprint
from utils.vis import crop_roi
from utils.augmentation import Augmentation, crop_roi, rotate, get_m1to1_gaussian_rand
from cmr.network import Pool
import pickle
import cv2
import skimage.io as io
import random

random.seed(0)

class HUMBI(data.Dataset):
    def __init__(self, root, phase, args, faces, writer=None, down_sample_list=None, img_std=0.5, img_mean=0.5, ms=True, which_split='split_1'):
        super(HUMBI, self).__init__()
        self.root = root
        self.phase = phase
        self.down_sample_list = down_sample_list
        self.size = args.size
        self.faces = faces
        self.img_std = img_std
        self.img_mean = img_mean
        self.ms = ms
        self.which_split = which_split
        self.pos_aug = args.pos_aug if 'train' in self.phase else 0
        self.rot_aug = args.rot_aug if 'train' in self.phase else 0
        self.use_rotate = args.use_rotate
        assert 0 <= self.rot_aug <= 180, 'rotaion limit must be in [0, 180]'
        self.color_aug = Augmentation(size=self.size) if args.color_aug and 'train' in self.phase else None
        # self.std = torch.tensor(0.20)
        self.std = 0.20
        self.template_mode = args.template_mode
        self.use_small_set = args.humbi_use_small
        self.use_world_or_cam = args.humbi_use_world_or_cam
        if args.template_mode not in ('groundtruth', 'mean_shape','calibrated'):
            raise "Not implemented for this specific template mode!"

        print('dataset template mode is ', self.template_mode)
        print('is using small dtaset?', self.use_small_set)
        print('is using world or camera annotation?', self.use_world_or_cam)
        # get data annotation list
        if self.use_small_set:
            print('here 1')
            train_file_name = 'training_small_'+self.which_split+'.pkl'
            test_file_name = 'test_small_'+self.which_split+'.pkl'
        else:
            train_file_name = 'training_all_'+self.which_split+'.pkl'
            test_file_name = 'test_all_'+self.which_split+'.pkl'
        if 'train' in self.phase:
            with open(os.path.join(self.root, 'split_annotations', train_file_name), 'rb') as f:
                self.db_data_anno = pickle.load(f)
        else:
            print('loading data from', test_file_name, '....')
            with open(os.path.join(self.root, 'split_annotations', test_file_name), 'rb') as f:
                self.db_data_anno = pickle.load(f)
        # get template mesh for all the subjects
        if args.template_mode in ('groundtruth', 'mean_shape'):
            with open(os.path.join(self.root, 'split_annotations', 'rest_pose_template.pkl'), 'rb') as f:
                self.rest_pose_templates = pickle.load(f)
            with open(os.path.join(self.root, 'split_annotations', 'gt_hand_shapes.pkl'), 'rb') as f:
                self.hand_shapes = pickle.load(f)

        elif args.template_mode == 'calibrated':
            # with open(os.path.join('out',
            #                     'inverse_models',
            #                     # 'experiment_mesh_10m_humbi_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2',
            #                     'experiment_mesh_80m_humbi_synth_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2',
            #                     # 'experiment_mesh_10m_humbi_synth_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2_permutation_10',
            #                     # 'experiment_mesh_80m_humbi_synth_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2_center_align',
            #                     # 'calibrate_humbi',
            #                     # 'calibrate_humbi_cmr_pg_no_mask_full_dataset_epochs_25_30',
            #                     # 'cmr_pg_no_mask_full_dataset_epochs_25_30',
            #                     # 'cmr_pg_no_mask_full_dataset_cam_split_5_epochs_10_15',
            #                     # 'cmr_pg_no_mask_full_dataset_cam_split_5_epochs_10_15_last_epoch',
            #                     'cmr_pg_no_mask_full_dataset_cam_split_5_epochs_10_15_rerun_4_cvpr',
            #                     # 'calibrate_humbi_from_mean_shape_result',
            #                     # 'calibrated_rest_pose_template.pkl'), 'rb') as f:
            #                     'calibrated_randomly_20_shape_all_subjects.pkl'), 'rb') as f:
            with open('out/humbi/mano_based_model_iterative_pose_conf_2nd_stage_full_dataset_cam_split_5_epochs_10_15_both_mesh_and_pose_loss_resnet50_multiply_ranking_loss_weight_1_margine_1_layers_1_bs_128/calibrated_shapes/calibrated_top_10_shape_all_subjects.pkl', 'rb') as f:

            # with open('out/humbi/mano_based_model_iterative_pose_without_gt_shape_full_dataset_cam_split_5_epochs_10_15_both_mesh_and_pose_loss_resnet50_multiply/calibrated_shapes/calibrated_randomly_20_shape_all_subjects.pkl', 'rb') as f:
            # with open('out/inverse_models/experiment_mesh_80m_humbi_synth_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2/mano_after_optimization/calibrated_shape_all_subjects.pkl', 'rb') as f:
                self.hand_shapes = pickle.load(f)


            if self.use_small_set:
                print('loading calibrated rest hand meshes....')
                # with open(os.path.join(self.root, 'split_annotations', 'calibrated_rest_pose_template.pkl'), 'rb') as f:
                # with open(os.path.join(self.root, '..', '..', 'out',
                with open(os.path.join('out',
                                    'inverse_models',
                                    'experiment_mesh_10m_humbi_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2',
                                    'cmr_pg_no_mask_train_better_visual',
                                    # 'calibrate_humbi_cmr_pg_no_mask_full_dataset_epochs_25_30',
                                    # 'calibrate_humbi_from_mean_shape_result',
                                    'calibrated_randomly_50_rest_pose_template_small.pkl'), 'rb') as f:
                    self.rest_pose_templates = pickle.load(f)
            else:
                print('loading calibrated full test hand meshes....')
                # with open(os.path.join(self.root, 'split_annotations', 'calibrated_rest_pose_template.pkl'), 'rb') as f:
                # with open(os.path.join(self.root, '..', '..', 'out',
                # with open(os.path.join('out',
                #                     'inverse_models',
                #                     # 'experiment_mesh_10m_humbi_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2',
                #                     'experiment_mesh_80m_humbi_synth_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2',
                #                     # 'experiment_mesh_10m_humbi_synth_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2_permutation_10',
                #                     # 'experiment_mesh_80m_humbi_synth_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2_center_align',
                #                     # 'calibrate_humbi',
                #                     # 'calibrate_humbi_cmr_pg_no_mask_full_dataset_epochs_25_30',
                #                     # 'cmr_pg_no_mask_full_dataset_epochs_25_30',
                #                     # 'cmr_pg_no_mask_full_dataset_cam_split_5_epochs_10_15',
                #                     # 'cmr_pg_no_mask_full_dataset_cam_split_5_epochs_10_15_last_epoch',
                #                     'cmr_pg_no_mask_full_dataset_cam_split_5_epochs_10_15_rerun_4_cvpr',
                #                     # 'calibrate_humbi_from_mean_shape_result',
                #                     # 'calibrated_rest_pose_template.pkl'), 'rb') as f:
                #                     'calibrated_randomly_20_rest_pose_template.pkl'), 'rb') as f:
                with open('out/humbi/mano_based_model_iterative_pose_conf_2nd_stage_full_dataset_cam_split_5_epochs_10_15_both_mesh_and_pose_loss_resnet50_multiply_ranking_loss_weight_1_margine_1_layers_1_bs_128/calibrated_shapes/calibrated_top_10_rest_pose_template.pkl', 'rb') as f:
                # with open('out/humbi/mano_based_model_iterative_pose_without_gt_shape_full_dataset_cam_split_5_epochs_10_15_both_mesh_and_pose_loss_resnet50_multiply/calibrated_shapes/calibrated_randomly_20_rest_pose_template.pkl','rb') as f:
                # with open('out/inverse_models/experiment_mesh_80m_humbi_synth_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2/mano_after_optimization/calibrated_rest_pose_template.pkl','rb') as f:
                    self.rest_pose_templates = pickle.load(f)


        if 'train' in self.phase and args.template_mode == 'calibrated':
            raise "Wrong configuration. Calibrated template should not be used in training phase!"     
        cprint('Loaded HUMBI {} {} samples'.format(self.phase, str(len(self.db_data_anno))), 'red')
        

    def __getitem__(self, idx):
        if self.phase == 'training':
            return self.get_training_sample(idx)
        elif self.phase in  ('evaluation', 'test') :
            return self.get_eval_sample(idx)
        else:
            raise Exception('phase error')

    def get_training_sample(self, idx):
        if self.use_world_or_cam == 'world':
            image_rela_path, uv, xyz, v0, mano = self.db_data_anno[idx][:5]
        elif self.use_world_or_cam == 'cam':
            image_rela_path = self.db_data_anno[idx][0]
            uv = self.db_data_anno[idx][1]
            xyz, v0, mano = self.db_data_anno[idx][5:]
        else:
            raise 'Use world or camera annnotation?'

        subject_id = int(image_rela_path.split('/')[0].split('_')[-1])

        img = io.imread(os.path.join(self.root, image_rela_path))
        orig_size = img.shape[0]

        translation = mano[:3]
        v0 = (v0 - translation[np.newaxis, :]) / self.std
        # rotate image
        if self.use_rotate and random.random()>0.4:
            # print('inside the rotation......')
            img, uv, v0, xyz = self.rotate_data(img, uv, v0, xyz)
        
        if self.color_aug is not None:
            img = self.color_aug(img)
        img = base_transform(img, size=self.size, mean=self.img_mean, std=self.img_std)
        uv[:, 0] = uv[:, 0] * self.size/orig_size
        uv[:, 1] = uv[:, 1] * self.size/orig_size

   
        
        uv_map = uv2map(uv.astype(np.int), img.shape[1:])
        uv_map = cv2.resize(uv_map.transpose(1, 2, 0), (img.shape[2]//2, img.shape[1]//2)).transpose(2, 0, 1)
        img, xyz, uv, uv_map, v0, mano = [torch.from_numpy(x).float() for x in [img, xyz, uv, uv_map, v0, mano]]

        xyz_root = xyz[0]
        xyz = (xyz - xyz_root) / self.std
        

        
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
            shape_params = self.hand_shapes[subject_id].float()
        elif self.template_mode == 'mean_shape':
            template_v0 = torch.from_numpy(self.rest_pose_templates['mean_shape']).float() / self.std
            shape_params = torch.zeros(10).float()
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
                'uv_point': uv,
                'uv_gt': uv_map,
                'xyz_root': xyz_root,
                'shape_params': shape_params,
                'pose_params': mano[3:26],
                'mano_translation':mano[0:3],
                }
        data['meta'] = image_rela_path
        return data

    def get_eval_sample(self, idx):
        # image_rela_path, uv, xyz, v0, mano = self.db_data_anno[idx]
        if self.use_world_or_cam == 'world':
            image_rela_path, uv, xyz, v0, mano = self.db_data_anno[idx][:5]
        elif self.use_world_or_cam == 'cam':
            image_rela_path = self.db_data_anno[idx][0]
            uv = self.db_data_anno[idx][1]
            xyz, v0, mano = self.db_data_anno[idx][5:]
        else:
            raise 'Use world or camera annnotation?'
        subject_id = int(image_rela_path.split('/')[0].split('_')[-1])

        img = io.imread(os.path.join(self.root, image_rela_path))
        orig_size = img.shape[0]

        img = base_transform(img, size=self.size, mean=self.img_mean, std=self.img_std)
        uv[:, 0] = uv[:, 0] * self.size/orig_size
        uv[:, 1] = uv[:, 1] * self.size/orig_size    
        
        uv_map = uv2map(uv.astype(np.int), img.shape[1:])
        uv_map = cv2.resize(uv_map.transpose(1, 2, 0), (img.shape[2]//2, img.shape[1]//2)).transpose(2, 0, 1)
        img, xyz, uv, uv_map, v0, mano = [torch.from_numpy(x).float() for x in [img, xyz, uv, uv_map, v0, mano]]

        xyz_root = xyz[0]
        xyz = (xyz - xyz_root) / self.std
        
        translation = mano[:3]
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
            shape_params = self.hand_shapes[subject_id].float()

        elif self.template_mode == 'calibrated':
            template_v0 = torch.from_numpy(self.rest_pose_templates[subject_id]).float() / self.std 
            shape_params = torch.from_numpy(self.hand_shapes[subject_id]).float()
        elif self.template_mode == 'mean_shape':
            template_v0 = torch.from_numpy(self.rest_pose_templates['mean_shape']).float() / self.std
            shape_params = torch.zeros(10).float()

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
                'uv_point': uv,
                'uv_gt': uv_map,
                'xyz_root': xyz_root,
                'shape_params': shape_params,
                'pose_params': mano[3:26],
                'mano_translation':mano[0:3],
                }
        data['meta'] = image_rela_path
        return data
        
    def rotate_data(self, img, uv=None, v0=None, xyz=None):
        if 'train' in self.phase:
            assert v0 is not None and xyz is not None
            if self.rot_aug > 0:
                angle = np.random.randint(-self.rot_aug, self.rot_aug)
                rot_mapping = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1.0)  # 12
                img = rotate(img, rot_mapping)
                rot_point = np.array([[np.cos(angle / 180. * np.pi), np.sin(angle / 180. * np.pi), 0],
                                      [-np.sin(angle / 180. * np.pi), np.cos(angle / 180. * np.pi), 0],
                                      [0, 0, 1]])
                uv = np.matmul(rot_point[:2, :2], (uv - np.array([[img.shape[1] // 2, img.shape[0] // 2]])).T).T + np.array(
                    [[img.shape[1] // 2, img.shape[0] // 2]])
                v0 = np.matmul(rot_point, v0.T).T
                xyz = np.matmul(rot_point, xyz.T).T
                return img, uv, v0, xyz
        else:
            raise "Won't rotate for validataion"

    def __len__(self):

        return len(self.db_data_anno)

    def get_face(self):
        return self.faces


class HUMBI_Group(data.Dataset):
    """
    For this class, each data point would contain a sequence of images from the same subject.
    """
    def __init__(self, root, phase, args, faces, writer=None, down_sample_list=None, img_std=0.5, img_mean=0.5, ms=True):
        super(HUMBI_Group, self).__init__()
        self.root = root
        self.group_size = args.group_size
        self.phase = phase
        self.down_sample_list = down_sample_list
        self.size = args.size
        self.faces = faces
        self.img_std = img_std
        self.img_mean = img_mean
        self.ms = ms
        self.pos_aug = args.pos_aug if 'train' in self.phase else 0
        self.rot_aug = args.rot_aug if 'train' in self.phase else 0
        assert 0 <= self.rot_aug <= 180, 'rotaion limit must be in [0, 180]'
        self.color_aug = Augmentation(size=self.size) if args.color_aug and 'train' in self.phase else None

        self.std = torch.tensor(0.20)
        # get data annotation list
        if 'train' in self.phase:
            with open(os.path.join(self.root, 'split_annotations', 'training_small_split_1.pkl'), 'rb') as f:
                self.db_data_anno = pickle.load(f)

            # group these annotations into different subjects
            all_unique_subjects =defaultdict(list)
            for item in self.db_data_anno:
                subject = item[0].split('/')[0]
                all_unique_subjects[subject].append(item)
            
            self.db_data_anno = []
            for _, value in all_unique_subjects.items():
                random.shuffle(value)
                self.db_data_anno.append(value)
                # db_data_anno now is a list of lists.

            self._split_into_chunks()

    
        else:
            with open(os.path.join(self.root, 'split_annotations', 'test_small_split_1.pkl'), 'rb') as f:
                self.db_data_anno = pickle.load(f)            
        cprint('Loaded HUMBI {} {} samples'.format(self.phase, str(len(self.db_data_anno))), 'red')

    def _split_into_chunks(self):
        anno = []
        for subject_anno in self.db_data_anno:
            anno.extend([subject_anno[i*self.group_size: (i+1)*self.group_size] for i in range(len(subject_anno)//self.group_size)])
        self.db_data_anno = anno       

    def __getitem__(self, idx):
        if 'train' in self.phase:
            return self.get_training_sample(idx)
        elif self.phase in  ('evaluation', 'test') :
            return self.get_eval_sample(idx)
        else:
            raise Exception('phase error')
    
    def get_training_sample_from_single_annotation(self, annotation):
        """
        Args:
            annnotation: a list, [image_rela_path, uv, xyz, v0, mano]
        """

        image_rela_path, uv, xyz, v0, mano = annotation
        img = io.imread(os.path.join(self.root, image_rela_path))
        orig_size = img.shape[0]
        
        if self.color_aug is not None:
            img = self.color_aug(img)
        img = base_transform(img, size=self.size, mean=self.img_mean, std=self.img_std)
        uv[:, 0] = uv[:, 0] * self.size/orig_size
        uv[:, 1] = uv[:, 1] * self.size/orig_size        
        
        uv_map = uv2map(uv.astype(np.int), img.shape[1:])
        uv_map = cv2.resize(uv_map.transpose(1, 2, 0), (img.shape[2]//2, img.shape[1]//2)).transpose(2, 0, 1)
        img, xyz, uv, uv_map, v0, mano = [torch.from_numpy(x).float() for x in [img, xyz, uv, uv_map, v0, mano]]

        xyz_root = xyz[0]
        xyz = (xyz - xyz_root) / self.std
        
        translation = mano[:3]
        v0 = (v0 - translation[np.newaxis, :]) / self.std
        
        if self.ms:
            v1 = Pool(v0.unsqueeze(0), self.down_sample_list[0])[0]
            v2 = Pool(v1.unsqueeze(0), self.down_sample_list[1])[0]
            v3 = Pool(v2.unsqueeze(0), self.down_sample_list[2])[0]
            gt = [v0, v1, v2, v3]
        else:
            gt = [v0, ]

        data = {'img': img,
                'mesh_gt': gt,
                'xyz_gt': xyz,
                'uv_point': uv,
                'uv_gt': uv_map,
                'xyz_root': xyz_root,
                }
        return data 

    def get_training_sample(self, idx):
        data = defaultdict(list)
        for anno in self.db_data_anno[idx]:
            this_data = self.get_training_sample_from_single_annotation(anno)
            for key, value in this_data.items():
                data[key].append(value)

        for key, value in data.items():
            if key != 'mesh_gt':
                data[key] = torch.stack(value)
            else:
                number_ms = len(value[0])
                mesh_list = []
                for i in range(number_ms):
                    cur_scale = []
                    for j in range(self.group_size):
                        cur_scale.append(value[j][i])
                    mesh_list.append(torch.stack(cur_scale))
                data[key] = mesh_list
        return data



    def get_eval_sample(self, idx):
        image_rela_path, uv, xyz, v0, mano = self.db_data_anno[idx]

        img = io.imread(os.path.join(self.root, image_rela_path))
        orig_size = img.shape[0]

        img = base_transform(img, size=self.size, mean=self.img_mean, std=self.img_std)
        uv[:, 0] = uv[:, 0] * self.size/orig_size
        uv[:, 1] = uv[:, 1] * self.size/orig_size    
        
        uv_map = uv2map(uv.astype(np.int), img.shape[1:])
        uv_map = cv2.resize(uv_map.transpose(1, 2, 0), (img.shape[2]//2, img.shape[1]//2)).transpose(2, 0, 1)
        img, xyz, uv, uv_map, v0, mano = [torch.from_numpy(x).float() for x in [img, xyz, uv, uv_map, v0, mano]]

        xyz_root = xyz[0]
        xyz = (xyz - xyz_root) / self.std
        
        translation = mano[:3]
        v0 = (v0 - translation[np.newaxis, :]) / self.std
        
        if self.ms:
            v1 = Pool(v0.unsqueeze(0), self.down_sample_list[0])[0]
            v2 = Pool(v1.unsqueeze(0), self.down_sample_list[1])[0]
            v3 = Pool(v2.unsqueeze(0), self.down_sample_list[2])[0]
            gt = [v0, v1, v2, v3]
        else:
            gt = [v0, ]

        data = {'img': img,
                'mesh_gt': gt,
                'xyz_gt': xyz,
                'uv_point': uv,
                'uv_gt': uv_map,
                'xyz_root': xyz_root,
                }
        return data

    def __len__(self):

        return len(self.db_data_anno)

    def get_face(self):
        return self.faces



class HUMBI_Group_V2(data.Dataset):
    """
    For this class, each data point would contain a sequence of images from the same subject.
    Coparing to HUMBI_GROUP, this V2 version returns more information
    """
    def __init__(self, root, phase, args, faces, writer=None, down_sample_list=None, img_std=0.5, img_mean=0.5, ms=True, which_split='split_1'):
        super(HUMBI_Group_V2, self).__init__()
        self.root = root
        self.group_size = args.group_size
        self.phase = phase
        self.down_sample_list = down_sample_list
        self.size = args.size
        self.faces = faces
        self.img_std = img_std
        self.img_mean = img_mean
        self.ms = ms
        self.which_split = which_split
        self.pos_aug = args.pos_aug if 'train' in self.phase else 0
        self.rot_aug = args.rot_aug if 'train' in self.phase else 0
        self.use_rotate = args.use_rotate
        assert 0 <= self.rot_aug <= 180, 'rotaion limit must be in [0, 180]'
        self.color_aug = Augmentation(size=self.size) if args.color_aug and 'train' in self.phase else None
        # self.std = torch.tensor(0.20)
        self.std = 0.20
        self.template_mode = args.template_mode
        self.use_small_set = args.humbi_use_small
        self.use_world_or_cam = args.humbi_use_world_or_cam
        if args.template_mode not in ('groundtruth', 'mean_shape','calibrated'):
            raise "Not implemented for this specific template mode!"

        print('dataset template mode is ', self.template_mode)
        print('is using small dtaset?', self.use_small_set)
        print('is using world or camera annotation?', self.use_world_or_cam)
        # get data annotation list
        if self.use_small_set:
            print('here 1')
            train_file_name = 'training_small_'+self.which_split+'.pkl'
            test_file_name = 'test_small_'+self.which_split+'.pkl'
        else:
            train_file_name = 'training_all_'+self.which_split+'.pkl'
            test_file_name = 'test_all_'+self.which_split+'.pkl'
        if 'train' in self.phase:
            with open(os.path.join(self.root, 'split_annotations', train_file_name), 'rb') as f:
                self.db_data_anno = pickle.load(f)
        else:
            print('loading data from', test_file_name, '....')
            with open(os.path.join(self.root, 'split_annotations', test_file_name), 'rb') as f:
                self.db_data_anno = pickle.load(f)

        # group these annotations into different subjects
        all_unique_subjects =defaultdict(list)
        for item in self.db_data_anno:
            subject = item[0].split('/')[0]
            all_unique_subjects[subject].append(item)
        
        self.db_data_anno = []
        for _, value in all_unique_subjects.items():
            random.shuffle(value)
            self.db_data_anno.append(value)
            # db_data_anno now is a list of lists.

        self._split_into_chunks()        

        # get template mesh for all the subjects
        if args.template_mode in ('groundtruth', 'mean_shape'):
            with open(os.path.join(self.root, 'split_annotations', 'rest_pose_template.pkl'), 'rb') as f:
                self.rest_pose_templates = pickle.load(f)
            with open(os.path.join(self.root, 'split_annotations', 'gt_hand_shapes.pkl'), 'rb') as f:
                self.hand_shapes = pickle.load(f)

        elif args.template_mode == 'calibrated':
            with open(os.path.join('out',
                                'inverse_models',
                                # 'experiment_mesh_10m_humbi_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2',
                                'experiment_mesh_80m_humbi_synth_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2',
                                # 'experiment_mesh_10m_humbi_synth_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2_permutation_10',
                                # 'experiment_mesh_80m_humbi_synth_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2_center_align',
                                # 'calibrate_humbi',
                                # 'calibrate_humbi_cmr_pg_no_mask_full_dataset_epochs_25_30',
                                # 'cmr_pg_no_mask_full_dataset_epochs_25_30',
                                # 'cmr_pg_no_mask_full_dataset_cam_split_5_epochs_10_15',
                                # 'cmr_pg_no_mask_full_dataset_cam_split_5_epochs_10_15_last_epoch',
                                'cmr_pg_no_mask_full_dataset_cam_split_5_epochs_10_15_rerun_4_cvpr',
                                # 'calibrate_humbi_from_mean_shape_result',
                                # 'calibrated_rest_pose_template.pkl'), 'rb') as f:
                                'calibrated_randomly_20_shape_all_subjects.pkl'), 'rb') as f:

            # with open('out/humbi/mano_based_model_iterative_pose_without_gt_shape_full_dataset_cam_split_5_epochs_10_15_both_mesh_and_pose_loss_resnet50_multiply/calibrated_shapes/calibrated_randomly_20_shape_all_subjects.pkl', 'rb') as f:
            # with open('out/inverse_models/experiment_mesh_80m_humbi_synth_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2/mano_after_optimization/calibrated_shape_all_subjects.pkl', 'rb') as f:
                self.hand_shapes = pickle.load(f)


            if self.use_small_set:
                print('loading calibrated rest hand meshes....')
                # with open(os.path.join(self.root, 'split_annotations', 'calibrated_rest_pose_template.pkl'), 'rb') as f:
                # with open(os.path.join(self.root, '..', '..', 'out',
                with open(os.path.join('out',
                                    'inverse_models',
                                    'experiment_mesh_10m_humbi_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2',
                                    'cmr_pg_no_mask_train_better_visual',
                                    # 'calibrate_humbi_cmr_pg_no_mask_full_dataset_epochs_25_30',
                                    # 'calibrate_humbi_from_mean_shape_result',
                                    'calibrated_randomly_50_rest_pose_template_small.pkl'), 'rb') as f:
                    self.rest_pose_templates = pickle.load(f)
            else:
                print('loading calibrated full test hand meshes....')
                # with open(os.path.join(self.root, 'split_annotations', 'calibrated_rest_pose_template.pkl'), 'rb') as f:
                # with open(os.path.join(self.root, '..', '..', 'out',
                with open(os.path.join('out',
                                    'inverse_models',
                                    # 'experiment_mesh_10m_humbi_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2',
                                    'experiment_mesh_80m_humbi_synth_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2',
                                    # 'experiment_mesh_10m_humbi_synth_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2_permutation_10',
                                    # 'experiment_mesh_80m_humbi_synth_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2_center_align',
                                    # 'calibrate_humbi',
                                    # 'calibrate_humbi_cmr_pg_no_mask_full_dataset_epochs_25_30',
                                    # 'cmr_pg_no_mask_full_dataset_epochs_25_30',
                                    # 'cmr_pg_no_mask_full_dataset_cam_split_5_epochs_10_15',
                                    # 'cmr_pg_no_mask_full_dataset_cam_split_5_epochs_10_15_last_epoch',
                                    'cmr_pg_no_mask_full_dataset_cam_split_5_epochs_10_15_rerun_4_cvpr',
                                    # 'calibrate_humbi_from_mean_shape_result',
                                    # 'calibrated_rest_pose_template.pkl'), 'rb') as f:
                                    'calibrated_randomly_20_rest_pose_template.pkl'), 'rb') as f:
                # with open('out/humbi/mano_based_model_iterative_pose_without_gt_shape_full_dataset_cam_split_5_epochs_10_15_both_mesh_and_pose_loss_resnet50_multiply/calibrated_shapes/calibrated_randomly_20_rest_pose_template.pkl','rb') as f:
                # with open('out/inverse_models/experiment_mesh_80m_humbi_synth_on_the_fly_batch_size_512_[256_512_1024_1024_512_256]_dropout_0_lr_1e-4_epochs_50_80_with_norm_meter_0_2/mano_after_optimization/calibrated_rest_pose_template.pkl','rb') as f:
                    self.rest_pose_templates = pickle.load(f)


        if 'train' in self.phase and args.template_mode == 'calibrated':
            raise "Wrong configuration. Calibrated template should not be used in training phase!"     
        cprint('Loaded HUMBI {} {} samples'.format(self.phase, str(len(self.db_data_anno))), 'red')
        
    def _split_into_chunks(self):
        anno = []
        for subject_anno in self.db_data_anno:
            anno.extend([subject_anno[i*self.group_size: (i+1)*self.group_size] for i in range(len(subject_anno)//self.group_size)])
        self.db_data_anno = anno 


    def __getitem__(self, idx):
        if self.phase == 'training':
            return self.get_training_sample(idx)
        elif self.phase in  ('evaluation', 'test') :
            return self.get_eval_sample(idx)
        else:
            raise Exception('phase error')

    def get_training_sample_from_single_annotation(self, annotation):
        if self.use_world_or_cam == 'world':
            image_rela_path, uv, xyz, v0, mano = annotation[:5]
        elif self.use_world_or_cam == 'cam':
            image_rela_path = annotation[0]
            uv = annotation[1]
            xyz, v0, mano = annotation[5:]
        else:
            raise 'Use world or camera annnotation?'

        subject_id = int(image_rela_path.split('/')[0].split('_')[-1])

        img = io.imread(os.path.join(self.root, image_rela_path))
        orig_size = img.shape[0]

        translation = mano[:3]
        v0 = (v0 - translation[np.newaxis, :]) / self.std
        # rotate image
        if self.use_rotate and random.random()>0.4:
            # print('inside the rotation......')
            img, uv, v0, xyz = self.rotate_data(img, uv, v0, xyz)
        
        if self.color_aug is not None:
            img = self.color_aug(img)
        img = base_transform(img, size=self.size, mean=self.img_mean, std=self.img_std)
        uv[:, 0] = uv[:, 0] * self.size/orig_size
        uv[:, 1] = uv[:, 1] * self.size/orig_size

   
        
        uv_map = uv2map(uv.astype(np.int), img.shape[1:])
        uv_map = cv2.resize(uv_map.transpose(1, 2, 0), (img.shape[2]//2, img.shape[1]//2)).transpose(2, 0, 1)
        img, xyz, uv, uv_map, v0, mano = [torch.from_numpy(x).float() for x in [img, xyz, uv, uv_map, v0, mano]]

        xyz_root = xyz[0]
        xyz = (xyz - xyz_root) / self.std
        

        
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
            shape_params = self.hand_shapes[subject_id].float()
        elif self.template_mode == 'mean_shape':
            template_v0 = torch.from_numpy(self.rest_pose_templates['mean_shape']).float() / self.std
            shape_params = torch.zeros(10).float()
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
                'uv_point': uv,
                'uv_gt': uv_map,
                'xyz_root': xyz_root,
                'shape_params': shape_params,
                'pose_params': mano[3:26],
                'mano_translation':mano[0:3],
                }
        data['meta'] = image_rela_path
        return data

    def get_training_sample(self, idx):
        data = defaultdict(list)
        for anno in self.db_data_anno[idx]:
            this_data = self.get_training_sample_from_single_annotation(anno)
            for key, value in this_data.items():
                data[key].append(value)

        for key, value in data.items():
            if key not in ('mesh_gt', 'meta', 'mesh_template'):
                data[key] = torch.stack(value)
            elif key == 'mesh_gt':
                number_ms = len(value[0])
                mesh_list = []
                for i in range(number_ms):
                    cur_scale = []
                    for j in range(self.group_size):
                        cur_scale.append(value[j][i])
                    mesh_list.append(torch.stack(cur_scale))
                data[key] = mesh_list
        return data

    def get_eval_sample_from_single_annotation(self, annotation):
        # image_rela_path, uv, xyz, v0, mano = self.db_data_anno[idx]
        if self.use_world_or_cam == 'world':
            image_rela_path, uv, xyz, v0, mano = annotation[:5]
        elif self.use_world_or_cam == 'cam':
            image_rela_path = annotation[0]
            uv = annotation[1]
            xyz, v0, mano = annotation[5:]
        else:
            raise 'Use world or camera annnotation?'
        subject_id = int(image_rela_path.split('/')[0].split('_')[-1])

        img = io.imread(os.path.join(self.root, image_rela_path))
        orig_size = img.shape[0]

        img = base_transform(img, size=self.size, mean=self.img_mean, std=self.img_std)
        uv[:, 0] = uv[:, 0] * self.size/orig_size
        uv[:, 1] = uv[:, 1] * self.size/orig_size    
        
        uv_map = uv2map(uv.astype(np.int), img.shape[1:])
        uv_map = cv2.resize(uv_map.transpose(1, 2, 0), (img.shape[2]//2, img.shape[1]//2)).transpose(2, 0, 1)
        img, xyz, uv, uv_map, v0, mano = [torch.from_numpy(x).float() for x in [img, xyz, uv, uv_map, v0, mano]]

        xyz_root = xyz[0]
        xyz = (xyz - xyz_root) / self.std
        
        translation = mano[:3]
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
            shape_params = self.hand_shapes[subject_id].float()

        elif self.template_mode == 'calibrated':
            template_v0 = torch.from_numpy(self.rest_pose_templates[subject_id]).float() / self.std 
            shape_params = torch.from_numpy(self.hand_shapes[subject_id]).float()
        elif self.template_mode == 'mean_shape':
            template_v0 = torch.from_numpy(self.rest_pose_templates['mean_shape']).float() / self.std
            shape_params = torch.zeros(10).float()

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
                'uv_point': uv,
                'uv_gt': uv_map,
                'xyz_root': xyz_root,
                'shape_params': shape_params,
                'pose_params': mano[3:26],
                'mano_translation':mano[0:3],
                }
        data['meta'] = image_rela_path
        return data

    def get_eval_sample(self, idx):
        data = defaultdict(list)
        for anno in self.db_data_anno[idx]:
            this_data = self.get_eval_sample_from_single_annotation(anno)
            for key, value in this_data.items():
                data[key].append(value)
        
        for key, value in data.items():
            print(key, type(value[0]))
        for key, value in data.items():
            """
            "mesh_template" leaves unprocessed
            """
            print('current key is ', key)
            if key not in ('mesh_gt', 'meta', 'mesh_template'):
                data[key] = torch.stack(value)
            elif key == 'mesh_gt':
                number_ms = len(value[0])
                mesh_list = []
                for i in range(number_ms):
                    cur_scale = []
                    for j in range(self.group_size):
                        cur_scale.append(value[j][i])
                    mesh_list.append(torch.stack(cur_scale))
                data[key] = mesh_list
        return data
  
    def rotate_data(self, img, uv=None, v0=None, xyz=None):
        if 'train' in self.phase:
            assert v0 is not None and xyz is not None
            if self.rot_aug > 0:
                angle = np.random.randint(-self.rot_aug, self.rot_aug)
                rot_mapping = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1.0)  # 12
                img = rotate(img, rot_mapping)
                rot_point = np.array([[np.cos(angle / 180. * np.pi), np.sin(angle / 180. * np.pi), 0],
                                      [-np.sin(angle / 180. * np.pi), np.cos(angle / 180. * np.pi), 0],
                                      [0, 0, 1]])
                uv = np.matmul(rot_point[:2, :2], (uv - np.array([[img.shape[1] // 2, img.shape[0] // 2]])).T).T + np.array(
                    [[img.shape[1] // 2, img.shape[0] // 2]])
                v0 = np.matmul(rot_point, v0.T).T
                xyz = np.matmul(rot_point, xyz.T).T
                return img, uv, v0, xyz
        else:
            raise "Won't rotate for validataion"

    def __len__(self):

        return len(self.db_data_anno)

    def get_face(self):
        return self.faces


class HUMBI_Confidence(data.Dataset):
    def __init__(self, root, phase, args, faces, writer=None, down_sample_list=None, img_std=0.5, img_mean=0.5, ms=True, which_split='split_1'):
        super(HUMBI_Confidence, self).__init__()
        assert 'p' in which_split
        self.root = root
        self.phase = phase
        self.down_sample_list = down_sample_list
        self.size = args.size
        self.faces = faces
        self.img_std = img_std
        self.img_mean = img_mean
        self.ms = ms
        self.which_split = which_split
        self.pos_aug = args.pos_aug if 'train' in self.phase else 0
        self.rot_aug = args.rot_aug if 'train' in self.phase else 0
        self.use_rotate = args.use_rotate
        assert 0 <= self.rot_aug <= 180, 'rotaion limit must be in [0, 180]'
        self.color_aug = Augmentation(size=self.size) if args.color_aug and 'train' in self.phase else None
        # self.std = torch.tensor(0.20)
        self.std = 0.20
        self.template_mode = args.template_mode
        self.use_small_set = args.humbi_use_small
        self.use_world_or_cam = args.humbi_use_world_or_cam
        if args.template_mode not in ('groundtruth', 'mean_shape','calibrated'):
            raise "Not implemented for this specific template mode!"

        # get data annotation list
        train_file_name = 'training_all_'+self.which_split+'.pkl'
        test_file_name = 'test_all_'+self.which_split+'.pkl'
        if 'train' in self.phase:
            with open(os.path.join(self.root, 'split_annotations', train_file_name), 'rb') as f:
                self.db_data_anno = pickle.load(f)
        else:
            print('loading data from', test_file_name, '....')
            with open(os.path.join(self.root, 'split_annotations', test_file_name), 'rb') as f:
                self.db_data_anno = pickle.load(f)



    def __getitem__(self, idx):
        if self.phase == 'training':
            return self.get_training_sample(idx)
        elif self.phase in  ('evaluation', 'test') :
            return self.get_eval_sample(idx)
        else:
            raise Exception('phase error')

    def get_training_sample(self, idx):
        image_rela_path = self.db_data_anno['image_relative_paths'][idx]
        subject_id = int(image_rela_path.split('/')[0].split('_')[-1])
        gt_mano_shape = self.db_data_anno['gt_shape_params'][idx]
        pred_mano_shape = self.db_data_anno['pred_shape_params'][idx]
        img = io.imread(os.path.join(self.root, image_rela_path))
        orig_size = img.shape[0]        
        if self.color_aug is not None:
            img = self.color_aug(img)
        img = base_transform(img, size=self.size, mean=self.img_mean, std=self.img_std)

        gt_mano_shape, pred_mano_shape = [torch.from_numpy(x).float() for x in [gt_mano_shape, pred_mano_shape]]
        
        data = {'img': img,
                'gt_mano_shape': gt_mano_shape,
                'pred_mano_shape':pred_mano_shape,
                'subject_id': torch.tensor(subject_id).long(),
                }
        data['meta'] = image_rela_path
        return data

    def get_eval_sample(self, idx):
        image_rela_path = self.db_data_anno['image_relative_paths'][idx]
        subject_id = int(image_rela_path.split('/')[0].split('_')[-1])
        gt_mano_shape = self.db_data_anno['gt_shape_params'][idx]
        pred_mano_shape = self.db_data_anno['pred_shape_params'][idx]
        img = io.imread(os.path.join(self.root, image_rela_path))
        orig_size = img.shape[0]        
        img = base_transform(img, size=self.size, mean=self.img_mean, std=self.img_std)

        gt_mano_shape, pred_mano_shape = [torch.from_numpy(x).float() for x in [gt_mano_shape, pred_mano_shape]]
        
        data = {'img': img,
                'gt_mano_shape': gt_mano_shape,
                'pred_mano_shape':pred_mano_shape,
                'subject_id': torch.tensor(subject_id).long(),
                }
        data['meta'] = image_rela_path
        return data

    def __len__(self):

        return len(self.db_data_anno['image_relative_paths'])

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

    args.phase = 'test'
    args.humbi_use_world_or_cam = 'cam'
    args.humbi_use_small = False
    args.use_rotate =False
    args.size = 224
    args.work_dir = './'

    if False:
        dataset = HUMBI('data/humbi', args.phase, args, tmp['face'], writer=None,
                        down_sample_list=down_transform_list, img_mean=args.img_mean, img_std=args.img_std, ms=args.ms_mesh)

        data = dataset.get_training_sample(4)
        print(data['img'].shape)
        print(data['uv_gt'].shape)
        fig, ax = plt.subplots(1,2)
        ax[0].imshow(inv_base_tranmsform(data['img'].numpy())[:, :, ::-1])
        ax[0].scatter(data['uv_point'].numpy()[:, 0], data['uv_point'].numpy()[:, 1])
        ax[1].imshow(np.sum(data['uv_gt'].numpy(), axis=0))
        fig.savefig('datasets/humbi/test.png')
    
    if False:
        dataset = HUMBI('data/humbi', args.phase, args, tmp['face'], writer=None,
                down_sample_list=down_transform_list, img_mean=args.img_mean, img_std=args.img_std, ms=args.ms_mesh, which_split='split_5')
        # print(dataset[0])
        # print(dataset.get_training_sample(0))
        # data = dataset.get_training_sample(4)

        # print(data['img'].shape)
        # print(data['uv_gt'].shape)
        # print(data['img'].shape)
        # print(data['uv_gt'].shape)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=False, num_workers=16, drop_last=True)
        for data_i in dataloader:
            with open('training_samples.pkl', 'wb') as f:
                pickle.dump(data_i, f)

            print (len(data_i['mesh_gt']))
            print(len(data_i['mesh_gt'][0]))
            print(data_i['mesh_gt'][0][0])
            # print(data_i['shape_params'])
            print(data_i['img'].shape)
            print(data_i['uv_gt'].shape)
            break
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(inv_base_tranmsform(data['img'].numpy())[:, :, ::-1])
        # ax[0].scatter(data['uv_point'].numpy()[:, 0], data['uv_point'].numpy()[:, 1])
        # ax[1].imshow(np.sum(data['uv_gt'].numpy(), axis=0))
        # fig.savefig('datasets/humbi/test.png')

    if True:
        args.group_size = 5

        dataset = HUMBI_Group_V2('data/humbi', args.phase, args, tmp['face'], writer=None,
                down_sample_list=down_transform_list, img_mean=args.img_mean, img_std=args.img_std, ms=args.ms_mesh, which_split='split_5')
        
        print(dataset[0]['img'].shape)
        stop
        # print(dataset[0])
        # print(dataset.get_training_sample(0))
        # data = dataset.get_training_sample(4)

        
        # print(data['uv_gt'].shape)
        # print(data['img'].shape)
        # print(data['uv_gt'].shape)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, pin_memory=False, num_workers=16, drop_last=True)
        for data_i in dataloader:
            stop

            print (len(data_i['mesh_gt']))
            print(len(data_i['mesh_gt'][0]))
            print(data_i['mesh_gt'][0][0])
            # print(data_i['shape_params'])
            print(data_i['img'].shape)
            print(data_i['uv_gt'].shape)
            break
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(inv_base_tranmsform(data['img'].numpy())[:, :, ::-1])
        # ax[0].scatter(data['uv_point'].numpy()[:, 0], data['uv_point'].numpy()[:, 1])
        # ax[1].imshow(np.sum(data['uv_gt'].numpy(), axis=0))
        # fig.savefig('datasets/humbi/test.png')
    
    # print(data)
    # for i in range(len(dataset)):
    #     data = dataset.get_training_sample(i)
    #     cv2.imshow('test', inv_base_tranmsform(data['img'].numpy())[:, :, ::-1])
    #     cv2.waitKey(0)

