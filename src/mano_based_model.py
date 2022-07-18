# ------------------------------------------------------------------------------
# Copyright (c) 2021
# Licensed under the MIT License.
# Written by Xingyu Chen(chenxingyusean@foxmail.com)
# ------------------------------------------------------------------------------
"""
CMR_PG_Pure_Pose_Identity_Aware
"""
import os
from re import S
from tempfile import template
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from manopth.manolayer import ManoLayer
from .network import ConvBlock, SpiralConv, Pool, ParallelDeblock, SelfAttention
from .resnet import resnet18, resnet50
from .resnet_orig import resnet18 as resnet18_orig
from .loss import l1_loss, bce_loss, normal_loss, edge_length_loss, mse_loss, prepare_ranking_data
from src.network import Pool, make_linear_layers_with_dropout
import pickle
import scipy
from .rotation_conversions import *


class MANO_Based_Model_Iterative_Pose(nn.Module):
    def __init__(self, args):
        super().__init__()

        # get feature_extractor
        if '50' in args.backbone:
            self.feature_extractor = resnet50(pretrained=True)
        elif '18' in args.backbone:
            self.feature_extractor = resnet18(pretrained=True)
        else:
            raise Exception("Not supported", args.backbone)

        self.iteration = args.iteration
        self.pose_agg_mode = args.pose_agg_mode # ('add', 'multiply')

        # pose regressor
        self.pose_6d_extractor = nn.ModuleList()
        for _ in range(self.iteration):
            self.pose_6d_extractor.append(make_linear_layers_with_dropout([1000 + 96, 512, 256, 128, 16*6], dropout_prob=0.0))

        # mano related buffers
        mano_pca_layer = ManoLayer(
                            mano_root='template', side='right', use_pca=True, ncomps=args.mano_pose_comps, flat_hand_mean=False)
        self.register_buffer('th_selected_comps', mano_pca_layer.th_selected_comps)
        self.register_buffer('th_hands_mean', mano_pca_layer.th_hands_mean)

        with open(os.path.join(args.work_dir, 'template', 'humbi_j_regressor.npy'), 'rb') as f:
            j_regressor = torch.from_numpy(np.load(f)).float()
            self.register_buffer('j_regressor', j_regressor.unsqueeze(0))
        
        # mano layer
        self.mano_layer = ManoLayer(mano_root='template', side='right', use_pca=False, flat_hand_mean=True)

    def mano_pca_to_rotation_6d(self, pose):
        """
        Args:
            pose:, mano pose PCA representation, first 3 corresponds to global rotation
        """
        th_full_hand_pose = pose[:,3:].mm(self.th_selected_comps)
        pose_axisang = torch.cat([
            pose[:, :3],
            self.th_hands_mean + th_full_hand_pose
        ], 1)
        pose_matrix = axis_angle_to_matrix(pose_axisang.view(-1,16,3)) # bs x 16 x 3 x 3
        pose_6d = matrix_to_rotation_6d(pose_matrix) #bs x 16 x 6
        pose_6d_vector = pose_6d.view(-1, 96)  
        return pose_6d_vector
    
    def pose_6d_to_axis_angle(self, pose_6d_vector):
        pose_6d = pose_6d_vector.view(-1, 16, 6)
        pose_matrix = rotation_6d_to_matrix(pose_6d)
        pose_axisang = matrix_to_axis_angle(pose_matrix).view(-1, 48)
        return pose_axisang

    def rotation_6d_to_matrix(self, d6):
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalization per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6)
        Returns:
            batch of rotation matrices of size (*, 3, 3)
        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1, eps=1e-6)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1, eps=1e-6)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)


    def matrix_to_rotation_6d(self, matrix):
        """
        Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
        by dropping the last row. Note that 6D representation is not unique.
        Args:
            matrix: batch of rotation matrices of size (*, 3, 3)
        Returns:
            6D rotation representation, of size (*, 6)
        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """
        return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

    def iter_decoder(self, feature):
        batch_size = feature.shape[0]
        pose_6d = torch.zeros(batch_size, 16, 3, 3).to(feature.device)
        pose_6d += torch.eye(3).unsqueeze(0).unsqueeze(0).to(feature.device)
        pose_6d = pose_6d[:,:,:2,:]
        pose_6d = pose_6d.reshape(batch_size, 16*6)

        pred_pose_6d = []
        for i in range(self.iteration):
            delta_pose = self.pose_6d_extractor[i](torch.cat((feature, pose_6d), dim=1))
            if self.pose_agg_mode == 'add':
                pose_6d = pose_6d + delta_pose
            elif self.pose_agg_mode == 'multiply':
                pose_6d = self.matrix_to_rotation_6d(torch.matmul(self.rotation_6d_to_matrix(delta_pose.view(batch_size, 16, 6)), 
                                    self.rotation_6d_to_matrix(pose_6d.view(batch_size, 16, 6)))).view(batch_size, 16*6)                
            pred_pose_6d.append(pose_6d)
        
        return pred_pose_6d

    def forward(self, img, shape_params):
        feature = self.feature_extractor(img)[0]
        pred_pose_6d = self.iter_decoder(feature)
        pred_mesh, pred_keypoints = self.mano_layer(self.pose_6d_to_axis_angle(pred_pose_6d[-1]), shape_params)


        return {'mesh_pred': [pred_mesh/1000/0.2,],
                'keypoints_pred':  pred_keypoints/1000/0.2,
                'mano_pose_pred': pred_pose_6d[::-1]    # note! the predicted pose is in the format of rotation 6d
                }


    def loss(self, **kwargs):
        loss_dict = dict()
        loss = 0.
        loss += l1_loss(kwargs['pred'][0], kwargs['gt'][0])
        loss_dict['l1_loss'] = loss.clone()

        loss_dict['normal_loss'] = 0.1 * normal_loss(kwargs['pred'][0], kwargs['gt'][0], kwargs['face'])
        loss_dict['edge_loss'] = edge_length_loss(kwargs['pred'][0], kwargs['gt'][0], kwargs['face'])            
        
        mano_pose_loss = 0.0
        mano_pose_loss += l1_loss(kwargs['mano_pose_pred'][0], self.mano_pca_to_rotation_6d(kwargs['mano_pose_gt']))
        loss_dict['mano_pose_loss'] = mano_pose_loss

        loss += loss_dict['normal_loss'] + loss_dict['edge_loss'] + loss_dict['mano_pose_loss']

        loss_dict['loss'] = loss

        return loss_dict



class MANO_Based_Model_Iterative_Pose_Without_GT_Shape(nn.Module):
    def __init__(self, args):
        super().__init__()

        # get feature_extractor
        if '50' in args.backbone:
            self.feature_extractor = resnet50(pretrained=True)
        elif '18' in args.backbone:
            self.feature_extractor = resnet18(pretrained=True)
        else:
            raise Exception("Not supported", args.backbone)

        self.iteration = args.iteration

        # pose regressor
        self.pose_6d_extractor = nn.ModuleList()
        for _ in range(self.iteration):
            self.pose_6d_extractor.append(make_linear_layers_with_dropout([1000 + 96, 512, 256, 128, 16*6], dropout_prob=0.0))

        # shape regressor
        self.shape_extractor = make_linear_layers_with_dropout([1000, 512, 256, 128, 10], dropout_prob=0.0)
        
        # mano related buffers
        mano_pca_layer = ManoLayer(
                            mano_root='template', side='right', use_pca=True, ncomps=args.mano_pose_comps, flat_hand_mean=False)
        self.register_buffer('th_selected_comps', mano_pca_layer.th_selected_comps)
        self.register_buffer('th_hands_mean', mano_pca_layer.th_hands_mean)

        with open(os.path.join(args.work_dir, 'template', 'humbi_j_regressor.npy'), 'rb') as f:
            j_regressor = torch.from_numpy(np.load(f)).float()
            self.register_buffer('j_regressor', j_regressor.unsqueeze(0))
        # mano layer
        self.mano_layer = ManoLayer(mano_root='template', side='right', use_pca=False, flat_hand_mean=True)

    def mano_pca_to_rotation_6d(self, pose):
        """
        Args:
            pose:, mano pose PCA representation, first 3 corresponds to global rotation
        """
        th_full_hand_pose = pose[:,3:].mm(self.th_selected_comps)
        pose_axisang = torch.cat([
            pose[:, :3],
            self.th_hands_mean + th_full_hand_pose
        ], 1)
        pose_matrix = axis_angle_to_matrix(pose_axisang.view(-1,16,3)) # bs x 16 x 3 x 3
        pose_6d = matrix_to_rotation_6d(pose_matrix) #bs x 16 x 6
        pose_6d_vector = pose_6d.view(-1, 96)  
        return pose_6d_vector
    
    def pose_6d_to_axis_angle(self, pose_6d_vector):
        pose_6d = pose_6d_vector.view(-1, 16, 6)
        pose_matrix = rotation_6d_to_matrix(pose_6d)
        pose_axisang = matrix_to_axis_angle(pose_matrix).view(-1, 48)
        return pose_axisang

    def rotation_6d_to_matrix(self, d6):
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalization per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6)
        Returns:
            batch of rotation matrices of size (*, 3, 3)
        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)


    def matrix_to_rotation_6d(self, matrix):
        """
        Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
        by dropping the last row. Note that 6D representation is not unique.
        Args:
            matrix: batch of rotation matrices of size (*, 3, 3)
        Returns:
            6D rotation representation, of size (*, 6)
        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """
        return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

    def iter_decoder(self, feature):
        batch_size = feature.shape[0]
        pose_6d = torch.zeros(batch_size, 16, 3, 3).to(feature.device)
        pose_6d += torch.eye(3).unsqueeze(0).unsqueeze(0).to(feature.device)
        pose_6d = pose_6d[:,:,:,:2]
        pose_6d = pose_6d.reshape(batch_size, 16*6)

        pred_pose_6d = []
        for i in range(self.iteration):
            delta_pose = self.pose_6d_extractor[i](torch.cat((feature, pose_6d), dim=1))
            pose_6d = pose_6d + delta_pose
            pred_pose_6d.append(pose_6d)
        
        return pred_pose_6d

    def forward(self, img):
        feature = self.feature_extractor(img)[0]
        pred_pose_6d = self.iter_decoder(feature)
        pred_shape = self.shape_extractor(feature)
        pred_mesh, pred_keypoints = self.mano_layer(self.pose_6d_to_axis_angle(pred_pose_6d[-1]), pred_shape)


        return {'mesh_pred': [pred_mesh/1000/0.2,], 
                'mano_pose_pred': pred_pose_6d[::-1],    # note! the predicted pose is in the format of rotation 6d
                'keypoints_pred':  pred_keypoints/1000/0.2,
                'mano_shape_pred': [pred_shape, ],
                }


    def loss(self, **kwargs):

        loss_dict = dict()
        loss = 0.
        loss += l1_loss(kwargs['pred'][0], kwargs['gt'][0])
        loss_dict['l1_loss'] = loss.clone()

        loss_dict['normal_loss'] = 0.1 * normal_loss(kwargs['pred'][0], kwargs['gt'][0], kwargs['face'])
        loss_dict['edge_loss'] = edge_length_loss(kwargs['pred'][0], kwargs['gt'][0], kwargs['face'])            
        
        mano_pose_loss = 0.0
        mano_pose_loss += l1_loss(kwargs['mano_pose_pred'][0], self.mano_pca_to_rotation_6d(kwargs['mano_pose_gt']))
        loss_dict['mano_pose_loss'] = mano_pose_loss
        loss_dict['mano_shape_loss'] = l1_loss(kwargs['mano_shape_pred'][0], kwargs['mano_shape_gt'])

        loss += loss_dict['normal_loss'] + loss_dict['edge_loss'] + loss_dict['mano_pose_loss'] + loss_dict['mano_shape_loss']

        loss_dict['loss'] = loss

        return loss_dict



class MANO_Based_Model_Iterative_Pose_Without_GT_Shape_With_Conf(nn.Module):
    def __init__(self, args):
        super().__init__()

        # get feature_extractor
        if '50' in args.backbone:
            self.feature_extractor = resnet50(pretrained=True)
        elif '18' in args.backbone:
            self.feature_extractor = resnet18(pretrained=True)
        else:
            raise Exception("Not supported", args.backbone)

        self.iteration = args.iteration

        # pose regressor
        self.pose_6d_extractor = nn.ModuleList()
        for _ in range(self.iteration):
            self.pose_6d_extractor.append(make_linear_layers_with_dropout([1000 + 96, 512, 256, 128, 16*6], dropout_prob=0.0))

        # shape regressor
        self.shape_extractor = make_linear_layers_with_dropout([1000, 512, 256, 128, 10], dropout_prob=0.0)

        if args.conf_hidden_layers == 1:
            if args.conf_hidden_layers_drop_out:
                self.conf_branch = [nn.Dropout(p=0.2), nn.Linear(1000, 1)]
            else:
                self.conf_branch = [nn.Linear(1000, 1)]

        elif args.conf_hidden_layers == 2:
            self.conf_branch = [nn.Linear(1000, 256),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(256, 1)]
        elif args.conf_hidden_layers == 3:
            self.conf_branch = [nn.Linear(1000, 256),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(256, 64),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(64, 16),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(16, 1)]
        elif args.conf_hidden_layers == 6:
            self.conf_branch = [nn.Linear(1000, 512),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(512, 256),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(256, 128),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(128, 64),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(64, 32),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(32, 16),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(16, 1)]
        if args.conf_use_sigmoid:
            self.conf_branch.append(nn.Sigmoid())
        self.conf_branch = nn.Sequential(*self.conf_branch)

        # mano related buffers
        mano_pca_layer = ManoLayer(
                            mano_root='template', side='right', use_pca=True, ncomps=args.mano_pose_comps, flat_hand_mean=False)
        self.register_buffer('th_selected_comps', mano_pca_layer.th_selected_comps)
        self.register_buffer('th_hands_mean', mano_pca_layer.th_hands_mean)

        with open(os.path.join(args.work_dir, 'template', 'humbi_j_regressor.npy'), 'rb') as f:
            j_regressor = torch.from_numpy(np.load(f)).float()
            self.register_buffer('j_regressor', j_regressor.unsqueeze(0))
        # mano layer
        self.mano_layer = ManoLayer(mano_root='template', side='right', use_pca=False, flat_hand_mean=True)

        # loss function
        self.conf_weight_scalar = torch.tensor(args.conf_weight_scalar)

        self.use_ranking_loss = args.use_ranking_loss
        if args.use_ranking_loss:
            self.ranking_loss = nn.MarginRankingLoss(margin=args.ranking_loss_margin)
        
        self.only_train_conf_branch = args.mano_based_conf_second_stage_train_conf_branch

    def mano_pca_to_rotation_6d(self, pose):
        """
        Args:
            pose:, mano pose PCA representation, first 3 corresponds to global rotation
        """
        th_full_hand_pose = pose[:,3:].mm(self.th_selected_comps)
        pose_axisang = torch.cat([
            pose[:, :3],
            self.th_hands_mean + th_full_hand_pose
        ], 1)
        pose_matrix = axis_angle_to_matrix(pose_axisang.view(-1,16,3)) # bs x 16 x 3 x 3
        pose_6d = matrix_to_rotation_6d(pose_matrix) #bs x 16 x 6
        pose_6d_vector = pose_6d.view(-1, 96)  
        return pose_6d_vector
    
    def pose_6d_to_axis_angle(self, pose_6d_vector):
        pose_6d = pose_6d_vector.view(-1, 16, 6)
        pose_matrix = rotation_6d_to_matrix(pose_6d)
        pose_axisang = matrix_to_axis_angle(pose_matrix).view(-1, 48)
        return pose_axisang

    def rotation_6d_to_matrix(self, d6):
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalization per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6)
        Returns:
            batch of rotation matrices of size (*, 3, 3)
        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)


    def matrix_to_rotation_6d(self, matrix):
        """
        Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
        by dropping the last row. Note that 6D representation is not unique.
        Args:
            matrix: batch of rotation matrices of size (*, 3, 3)
        Returns:
            6D rotation representation, of size (*, 6)
        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """
        return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

    def iter_decoder(self, feature):
        batch_size = feature.shape[0]
        pose_6d = torch.zeros(batch_size, 16, 3, 3).to(feature.device)
        pose_6d += torch.eye(3).unsqueeze(0).unsqueeze(0).to(feature.device)
        pose_6d = pose_6d[:,:,:,:2]
        pose_6d = pose_6d.reshape(batch_size, 16*6)

        pred_pose_6d = []
        for i in range(self.iteration):
            delta_pose = self.pose_6d_extractor[i](torch.cat((feature, pose_6d), dim=1))
            pose_6d = pose_6d + delta_pose
            pred_pose_6d.append(pose_6d)
        
        return pred_pose_6d

    def forward(self, img):
        
        if self.only_train_conf_branch:
            with torch.no_grad():
                feature = self.feature_extractor(img)[0]
                pred_pose_6d = self.iter_decoder(feature)
                pred_shape = self.shape_extractor(feature)
                pred_mesh, pred_keypoints = self.mano_layer(self.pose_6d_to_axis_angle(pred_pose_6d[-1]), pred_shape)
            confidence= self.conf_branch(feature.detach())     
        else:
            feature = self.feature_extractor(img)[0]
            pred_pose_6d = self.iter_decoder(feature)
            pred_shape = self.shape_extractor(feature)
            pred_mesh, pred_keypoints = self.mano_layer(self.pose_6d_to_axis_angle(pred_pose_6d[-1]), pred_shape)
            confidence= self.conf_branch(feature.detach())


        return {'mesh_pred': [pred_mesh/1000/0.2,], 
                'mano_pose_pred': pred_pose_6d[::-1],    # note! the predicted pose is in the format of rotation 6d
                'keypoints_pred':  pred_keypoints/1000/0.2,
                'mano_shape_pred': [pred_shape, ],
                'conf_pred': confidence,
                }


    def loss(self, **kwargs):

        loss_dict = dict()
        loss = 0.
        loss += l1_loss(kwargs['pred'][0], kwargs['gt'][0])
        loss_dict['l1_loss'] = loss.clone()

        loss_dict['normal_loss'] = 0.1 * normal_loss(kwargs['pred'][0], kwargs['gt'][0], kwargs['face'])
        loss_dict['edge_loss'] = edge_length_loss(kwargs['pred'][0], kwargs['gt'][0], kwargs['face'])            
        
        mano_pose_loss = 0.0
        mano_pose_loss += l1_loss(kwargs['mano_pose_pred'][0], self.mano_pca_to_rotation_6d(kwargs['mano_pose_gt']))
        loss_dict['mano_pose_loss'] = mano_pose_loss
        loss_dict['mano_shape_loss'] = l1_loss(kwargs['mano_shape_pred'][0], kwargs['mano_shape_gt'])
        loss += loss_dict['normal_loss'] + loss_dict['edge_loss'] + loss_dict['mano_pose_loss'] + loss_dict['mano_shape_loss']
        
        loss_dict['loss'] = loss

        shape_gt = kwargs['mano_shape_gt']
        shape_pred = kwargs['mano_shape_pred'][0].detach()
        l1_distance = torch.mean(torch.abs(shape_gt-shape_pred), dim=1)            

        if self.use_ranking_loss:
            x1, x2, y = prepare_ranking_data(kwargs['conf_pred'].squeeze(), l1_distance.squeeze())
            loss_dict['conf_loss'] =  self.conf_weight_scalar * self.ranking_loss(x1, x2, y)
        else:       
            loss_dict['conf_loss'] =  self.conf_weight_scalar * l1_loss(kwargs['conf_pred'], (-1*l1_distance).unsqueeze(-1))       
        loss += loss_dict['conf_loss']

        if self.only_train_conf_branch:
            loss = loss_dict['conf_loss']

        loss_dict['loss'] = loss
        return loss_dict

