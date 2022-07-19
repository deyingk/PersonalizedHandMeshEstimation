import os
import os.path as op
import json
import cv2
import base64
import numpy as np
import code
import torch
from tqdm import tqdm
from src.utils.tsv_file_ops import tsv_reader, tsv_writer
from src.utils.tsv_file_ops import generate_linelist_file
from src.utils.tsv_file_ops import generate_hw_file
from src.utils.tsv_file import TSVFile
from src.utils.image_ops import img_from_base64
from src.modeling._mano import MANO
from src.tools.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle

from manopth.manolayer import ManoLayer

import pickle
# mano = MANO().cuda()

from collections import defaultdict

tsv_file = "{}/{}.img.tsv"
hw_file = "{}/{}.hw.tsv"
label_file = "{}/{}.label.tsv"
linelist_file = "{}/{}.linelist.tsv"



mano_pca_layer = ManoLayer(mano_root='Your_MANO_ROOT', side='right', use_pca=True, ncomps=48, flat_hand_mean=False)
th_selected_comps = mano_pca_layer.th_selected_comps
th_hands_mean = mano_pca_layer.th_hands_mean

print(th_selected_comps.shape)  # 45 x 45
print(th_hands_mean.shape) # 1 x 45

def mano_pca_to_rotation_6d(pose):
    """
    Args:
        pose:, mano pose PCA representation, first 3 corresponds to global rotation
    """
    th_full_hand_pose = pose[:,3:].mm(th_selected_comps)
    pose_axisang = torch.cat([
        pose[:, :3],
        th_hands_mean + th_full_hand_pose
    ], 1)
    pose_matrix = axis_angle_to_matrix(pose_axisang.view(-1,16,3)) # bs x 16 x 3 x 3
    pose_6d = matrix_to_rotation_6d(pose_matrix) #bs x 16 x 6
    pose_6d_vector = pose_6d.view(-1, 96)  
    return pose_6d_vector

def pose_6d_to_axis_angle(pose_6d_vector):
    pose_6d = pose_6d_vector.view(-1, 16, 6)
    pose_matrix = rotation_6d_to_matrix(pose_6d)
    pose_axisang = matrix_to_axis_angle(pose_matrix).view(-1, 48)
    return pose_axisang


def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2]) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2]) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def preproc(dataset_folder, dataset_tsv_folder, split):
    # init SMPL
    smpl_mesh_model = MANO()

    # bbox expansion factor
    scaleFactor = 1

    imgfiles_folder = dataset_folder+'/cropped_hand_size_2'

    # annotation loading
    rows, rows_label, rows_hw = [], [], []


    # get annotations
    if 'train' in split:
        with open(os.path.join(dataset_folder, 'split_annotations', 's1_train_size_2.pkl'), 'rb') as f:
            db_data_anno = pickle.load(f)
    elif 'val' in split:
        with open(os.path.join(dataset_folder, 'split_annotations', 's1_val_size_2.pkl'), 'rb') as f:
            db_data_anno = pickle.load(f)
    elif 'test' in split:
        with open(os.path.join(dataset_folder, 'split_annotations', 's1_test_size_2.pkl'), 'rb') as f:
            db_data_anno = pickle.load(f)
    else:
        raise "phase setting for the dataset is incorrect!!"     

    with open(os.path.join(dataset_folder, 'split_annotations', 'unique_shapes_dex_ycb.pkl'), 'rb') as f:
        hand_shapes = pickle.load(f)


    for anno_item in tqdm(db_data_anno):

        image_rela_path = anno_item['cropped_img_path'][19:]
        subject_id = int(image_rela_path.split('/')[1].split('-')[-1])
        imgname = image_rela_path

        img_data = cv2.imread(os.path.join(dataset_folder, image_rela_path))
        img_encoded_str = base64.b64encode(cv2.imencode('.jpg', img_data)[1])
        height = img_data.shape[0]
        width = img_data.shape[1]
        center = [height/2,width/2]

        scale = scaleFactor*max(height, width)/200

        mano = anno_item['mano_params']
        mano_pose_pca = mano[:48]
        mano_pose_pca = torch.from_numpy(mano_pose_pca).unsqueeze(0)
        th_full_hand_pose = mano_pose_pca[:,3:].mm(th_selected_comps)
        mano_pose = torch.cat([
            mano_pose_pca[:, :3],
            th_hands_mean + th_full_hand_pose
        ], 1)         # pose_axisang
        mano_pose = mano_pose.squeeze().numpy()

        mano_shape = hand_shapes[subject_id]  # ..... need modification

        gt_3d_joints = anno_item['regressed_joint_3d']
        
        mano_pose_tensor = torch.FloatTensor(mano_pose).view(1,-1)
        mano_shape_tensor = torch.FloatTensor(mano_shape).view(1,-1)


        gt_3d_joints_tag = np.ones((1,21,4))
        gt_3d_joints_tag[0,:,0:3] = gt_3d_joints - anno_item['mano_params'][48:51][np.newaxis,:]  # units in meters

        gt_2d_joints_tag = np.ones([21,3])
        gt_2d_joints_tag[:,:2] = anno_item['cropped_joint_2d']

        mano_pose_camera_corrd = np.asarray(mano_pose_tensor).tolist()[0]
        mano_shape_camera_corrd = np.asarray(mano_shape_tensor).tolist()[0]
                        
        labels = []
        labels.append({"center": center, "scale": scale,
            "2d_joints": gt_2d_joints_tag.tolist(), "has_2d_joints": 1,
            "3d_joints": gt_3d_joints_tag.tolist(), "has_3d_joints": 1,
            "pose": mano_pose_camera_corrd, "betas": mano_shape_camera_corrd, "has_smpl": 1 })

        row_label = [imgname, json.dumps(labels)]
        rows_label.append(row_label)
        row = [imgname, img_encoded_str]
        rows.append(row)
        height = img_data.shape[0]
        width = img_data.shape[1]
        row_hw = [imgname, json.dumps([{"height":height, "width":width}])]
        rows_hw.append(row_hw)


    resolved_label_file = label_file.format(dataset_tsv_folder, split)
    print('save to',resolved_label_file)
    tsv_writer(rows_label, resolved_label_file)

    resolved_linelist_file = linelist_file.format(dataset_tsv_folder, split)
    print('save to',resolved_linelist_file)
    generate_linelist_file(resolved_label_file, save_file=resolved_linelist_file)

    resolved_tsv_file = tsv_file.format(dataset_tsv_folder, split)
    print('save to',resolved_tsv_file)
    tsv_writer(rows, resolved_tsv_file)

    resolved_hw_file = hw_file.format(dataset_tsv_folder, split)
    print('save to',resolved_hw_file)
    tsv_writer(rows_hw, resolved_hw_file)



def main():

    print('Set the correct path for the mano model!')
    print('To run this file, please install METRO.')

    datasets = ['train','test', 'val']
    dataset_img_folder = "./datasets/dex_ycb"
    dataset_tsv_folder = "./datasets/dex_ycb_tsv_reproduce"
    for split in datasets:
        preproc(dataset_img_folder, dataset_tsv_folder, split)

if __name__ == '__main__':
    main()