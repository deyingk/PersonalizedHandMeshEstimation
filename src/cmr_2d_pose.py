# ------------------------------------------------------------------------------
# Copyright (c) 2021
# Licensed under the MIT License.
# Written by Xingyu Chen(chenxingyusean@foxmail.com)
# ------------------------------------------------------------------------------
"""
CMR_PG_Pure_Pose_Identity_Aware
"""

from re import S
from tempfile import template
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from manopth.manolayer import ManoLayer
from .network import ConvBlock, SpiralConv, Pool, ParallelDeblock, SelfAttention
from .resnet import resnet18, resnet50
from .loss import l1_loss, bce_loss, normal_loss, edge_length_loss, mse_loss
from src.network import Pool, make_linear_layers_with_dropout
import pickle
import scipy
from .rotation_conversions import *

class EncodeStage1(nn.Module):
    def __init__(self, backbone):
        super(EncodeStage1, self).__init__()
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    def forward(self, x):
        x0 = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x0, x4, x3, x2, x1


class EncodeStage2(nn.Module):
    def __init__(self, backbone, in_channel):
        super(EncodeStage2, self).__init__()
        self.reduce = nn.Sequential(ConvBlock(in_channel, in_channel, relu=True, norm='bn'),
                                    ConvBlock(in_channel, 128, relu=True, norm='bn'),
                                    ConvBlock(128, 64, kernel_size=1, padding=0, relu=False, norm='bn'))
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc

    def forward(self, x):
        x0 = self.reduce(x)
        x = self.maxpool(x0)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4, x3, x2, x1


class EncodeStage3(nn.Module):
    def __init__(self, backbone, in_channel):
        super(EncodeStage3, self).__init__()
        self.reduce = nn.Sequential(ConvBlock(in_channel, in_channel, relu=True, norm='bn'),
                                    ConvBlock(in_channel, 128, relu=True, norm='bn'),
                                    ConvBlock(128, 64, kernel_size=1, padding=0, relu=False, norm='bn'))
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc

    def forward(self, x):
        x = self.reduce(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x, x4, x3, x2, x1


class CMR_PG_PP_2D_Pose(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.uv_channel = 21
        self.relation = [[4, 8], [4, 12], [4, 16], [4, 20], [8, 12], [8, 16], [8, 20], [12, 16], [12, 20], [16, 20]]

        backbone, self.latent_size = self.get_backbone(args.backbone)
        self.backbone1 = EncodeStage1(backbone)

        backbone2, _ = self.get_backbone(args.backbone)
        self.backbone2 = EncodeStage2(backbone2, 64+self.uv_channel)

        backbone3, _ = self.get_backbone(args.backbone)
        self.backbone3 = EncodeStage3(backbone3, 64+self.uv_channel+len(self.relation))

        self.uv_prior_delayer = nn.ModuleList([ConvBlock(self.latent_size[2] + self.latent_size[1], self.latent_size[2], kernel_size=3, relu=True, norm='bn'),
                                               ConvBlock(self.latent_size[3] + self.latent_size[2], self.latent_size[3], kernel_size=3, relu=True, norm='bn'),
                                               ConvBlock(self.latent_size[4] + self.latent_size[3], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
                                               ConvBlock(self.latent_size[4], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
                                               ])
        self.uv_prior_head = ConvBlock(self.latent_size[4], self.uv_channel, kernel_size=3, padding=1, relu=False, norm=None)

        self.uv_delayer = nn.ModuleList([ConvBlock(self.latent_size[2] + self.latent_size[1], self.latent_size[2], kernel_size=3, relu=True, norm='bn'),
                                         ConvBlock(self.latent_size[3] + self.latent_size[2], self.latent_size[3], kernel_size=3, relu=True, norm='bn'),
                                         ConvBlock(self.latent_size[4] + self.latent_size[3], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
                                         ConvBlock(self.latent_size[4], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
                                         ])
        self.uv_head = ConvBlock(self.latent_size[4], self.uv_channel, kernel_size=3, padding=1, relu=False, norm=None)

        self.uvm_delayer = nn.ModuleList([ConvBlock(self.latent_size[2] + self.latent_size[1], self.latent_size[2], kernel_size=3, relu=True, norm='bn'),
                                          ConvBlock(self.latent_size[3] + self.latent_size[2], self.latent_size[3], kernel_size=3, relu=True, norm='bn'),
                                          ConvBlock(self.latent_size[4] + self.latent_size[3], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
                                          ConvBlock(self.latent_size[4], self.latent_size[4], kernel_size=3, relu=True, norm='bn'),
                                          ])
        # self.uvm_head = ConvBlock(self.latent_size[4], self.uv_channel+1, kernel_size=3, padding=1, relu=False, norm=None)
        self.uvm_head = ConvBlock(self.latent_size[4], self.uv_channel, kernel_size=3, padding=1, relu=False, norm=None)



    def get_backbone(self, backbone, pretrained=True):
        if '50' in backbone:
            basenet = resnet50(pretrained=pretrained)
            latent_channel = (1000, 2048, 1024, 512, 256)
        elif '18' in backbone:
            basenet = resnet18(pretrained=pretrained)
            latent_channel = (1000, 512, 256, 128, 64)
        else:
            raise Exception("Not supported", backbone)

        return basenet, latent_channel


    def uv_prior_decoder(self, z):
        x = z[0]
        for i, de in enumerate(self.uv_prior_delayer):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if i < 3:
                x = torch.cat((x, z[i+1]), dim=1)
            x = de(x)
        pred = torch.sigmoid(self.uv_prior_head(x))

        return pred

    def uv_decoder(self, z):
        x = z[0]
        for i, de in enumerate(self.uv_delayer):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if i < 3:
                x = torch.cat((x, z[i+1]), dim=1)
            x = de(x)
        pred = torch.sigmoid(self.uv_head(x))

        return pred

    def uvm_decoder(self, z):
        x = z[0]
        for i, de in enumerate(self.uvm_delayer):
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            if i < 3:
                x = torch.cat((x, z[i + 1]), dim=1)
            x = de(x)
        pred = torch.sigmoid(self.uvm_head(x))

        return pred


    def forward(self, x):
        z1 = self.backbone1(x)
        pred1 = self.uv_prior_decoder(z1[1:])
        z2 = self.backbone2(torch.cat([z1[0], pred1], 1))
        pred2 = self.uv_decoder(z2)
        z3 = self.backbone3(torch.cat([z1[0], pred2] + [pred2[:, i].sum(dim=1, keepdim=True) for i in self.relation], 1))
        pred4 = self.uvm_decoder(z3[1:])

        out= {'uv_pred': pred4,
                # 'uv_pred': pred4[:, :self.uv_channel],
                # 'mask_pred': pred4[:, self.uv_channel],
                'uv_prior': pred1,
                'uv_prior2': pred2,
                }
        return out

    def loss(self, **kwargs):                  
        loss_dict = dict()
        loss = 0.

        loss_dict['uv_loss'] = 10 * bce_loss(kwargs['uv_pred'], kwargs['uv_gt'])
        loss_dict['uv_prior_loss'] = 10 * bce_loss(kwargs['uv_prior'], kwargs['uv_gt'])
        loss_dict['uv_prior_loss2'] = 10 * bce_loss(kwargs['uv_prior2'], kwargs['uv_gt'])
        loss += loss_dict['uv_loss'] + loss_dict['uv_prior_loss'] + loss_dict['uv_prior_loss2']

        loss_dict['loss'] = loss
        return loss_dict



def main():
    import os.path as osp
    from utils.read import spiral_tramsform
    class Args:
        pass
    args = Args()
    args.in_channels = 3
    args.out_channels = [64, 128, 256, 512]
    args.iteration = 3
    args.backbone = 'ResNet18'
    args.work_dir = osp.dirname(osp.realpath(__file__))


    # dummy inputs
    img = torch.rand(12,3,224,224)
    shape = torch.rand(12, 10)

    pose_detector = CMR_PG_PP_2D_Pose(args)

    out = pose_detector(img)
    print(out['uv_pred'].shape)
    
if __name__ == '__main__':
    main()