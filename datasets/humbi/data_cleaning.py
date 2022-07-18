'''
The goal of this script is to:

1) generate 2D annotation from the released 3D annotions
'''
import os
import numpy as np
from manopth.manolayer import ManoLayer
from scipy.spatial.transform import Rotation as R
import torch

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat

def batch_rodrigues(axisang):
    #axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat


def load_camera_params(param_path):
    """
    Load camera params from the provided txt file.
    Could be used to extract /project matrices.

    Args:
        param_path : the txt file containing the camera params.
    """
    with open(param_path, 'r') as f:
        contents = f.readlines()
    camera_dict = {}
    for line in range(3, len(contents),4):
        current_line = line
        camera_id = contents[current_line].strip().split(' ')[-1]
        
        current_line += 1
        project_matrix = np.zeros((3, 4))
        current_row = 0
        while current_row < 3:
            entries = contents[current_line].strip().split(' ')
            for i in range(4):
                project_matrix[current_row, i] = float(entries[i])
            current_row += 1
            current_line += 1
        camera_dict[camera_id.zfill(7)] = project_matrix
    return camera_dict

def load_extrinsic_params(param_path):
    """
    Load camera params from the provided txt file.
    Could be used to extract extrinsic matrices.

    Args:
        param_path : the txt file containing the camera params.
    """
    with open(param_path, 'r') as f:
        contents = f.readlines()
    camera_dict = {}
    for line in range(3, len(contents),5):
        current_line = line
        camera_id = contents[current_line].strip().split(' ')[-1]

        current_line += 1   
        t = np.zeros((3,1))
        rotation_matrix = np.zeros((3, 3))
        entries = contents[current_line].strip().split(' ')
        for i in range(3):
            t[i, 0] = float(entries[i])
        current_line += 1

        current_row = 0
        while current_row < 3:
            entries = contents[current_line].strip().split(' ')
            for i in range(3):
                rotation_matrix[current_row, i] = float(entries[i])
            current_row += 1
            current_line += 1
        camera_dict[camera_id.zfill(7)] = np.concatenate((rotation_matrix, -rotation_matrix@t), -1)
    return camera_dict

def load_crop_params(params_path):
    with open(params_path, 'r') as f:
        contents = f.readlines()
    params_dict = {}
    for line in contents:
        camera_id, xmin, xmax, ymin, ymax = line.strip().split(' ')
        params_dict[camera_id.zfill(7)] = [int(xmin), int(xmax), int(ymin), int(ymax)]
    return params_dict



data_dir = '../../data/humbi'

if_save = False
if_save_3d = True
device = torch.device('cuda')
cpu = torch.device('cpu')


for subject in sorted(os.listdir(data_dir)):
    if 'subject' not in subject:
        continue
    print('....processing....', subject)
    if subject == 'subject_52':
        # the image folders under subject_52 are actually empty
        continue
    subject_dir = os.path.join(data_dir, subject)
    hand_dir = os.path.join(subject_dir, 'hand')

    camera_dict = load_camera_params(os.path.join(hand_dir, 'project.txt'))
    extrinsic_dict = load_extrinsic_params(os.path.join(hand_dir, 'extrinsic.txt'))

    all_frames = sorted(os.listdir(hand_dir))
    all_frame_directories = []
    for item in all_frames:
        if os.path.isdir(os.path.join(hand_dir, item)):
            all_frame_directories.append(os.path.join(hand_dir, item))

    ncomps = 20
    mano_layer = ManoLayer(
        mano_root='../../template', use_pca=True, ncomps=ncomps, flat_hand_mean=False)
    mano_layer = mano_layer.to(device)

    for frame_dir in all_frame_directories:
        # get 3D keypoints
        print(frame_dir)
        save_dir = os.path.join(frame_dir, 'reconstruction/keypoints_r_2d')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        keypoints = np.loadtxt(os.path.join(frame_dir, 'reconstruction/keypoints_r.txt'))

        cam_vertex_save_dir = os.path.join(frame_dir, 'reconstruction/cam_vertex_r')
        if not os.path.isdir(cam_vertex_save_dir):
            os.mkdir(cam_vertex_save_dir)
        cam_mano_save_dir = os.path.join(frame_dir, 'reconstruction/cam_mano_r')
        if not os.path.isdir(cam_mano_save_dir):
            os.mkdir(cam_mano_save_dir)

        r_params_raw = np.loadtxt(os.path.join(frame_dir, 'reconstruction/mano_params_r.txt'))
        world_mano_hand_verts = np.loadtxt(os.path.join(frame_dir, 'reconstruction/vertices_r.txt'))

        right_hand_image_dir = os.path.join(frame_dir, 'image_cropped', 'right')
        images_list = sorted(os.listdir(os.path.join(frame_dir, 'image_cropped', 'right')))
        images_list = list(filter(lambda x: x.endswith('.png'), images_list))

        crop_params = load_crop_params(os.path.join(right_hand_image_dir, 'list.txt'))
        
        for image_name in images_list:
            camera_id = image_name[5:-4]
            # project
            project_matrix = camera_dict[camera_id]
            keypoints_2d_homo = project_matrix @ np.concatenate((keypoints.T, np.ones((1, 21))), 0)
            keypoints_2d_homo = keypoints_2d_homo/ (keypoints_2d_homo[-1:, :])
            keypoints_2d = keypoints_2d_homo[:2]
            # crop and rescale
            xmin, xmax, ymin, ymax = crop_params[camera_id]
            keypoints_2d_crop = keypoints_2d - np.array([[xmin],[ymin]])
            keypoints_2d_crop_and_resize = np.zeros(keypoints_2d_crop.shape)
            keypoints_2d_crop_and_resize[0] = keypoints_2d_crop[0] * 250/(xmax - xmin)
            keypoints_2d_crop_and_resize[1] = keypoints_2d_crop[1] * 250/(ymax - ymin)
            keypoints_2d_crop_and_resize = keypoints_2d_crop_and_resize.T
            if if_save:
                with open(os.path.join(save_dir, camera_id+'.npy'), 'wb') as f:
                    np.save(f, keypoints_2d_crop_and_resize)

            extrinsic_matrix = extrinsic_dict[camera_id]
            cam_vertices = (extrinsic_matrix @ (np.concatenate((world_mano_hand_verts.T, np.ones((1,778))),axis=0))).T

            # get camera cood mano params
            r_params = np.copy(r_params_raw)
            r_params = r_params[np.newaxis, :]
            translation = r_params[:,:3]
            pose_params = r_params[:,3:26]
            global_pose = r_params[:, 3:6]
            hand_pose = r_params[:,6:26]
            shape_params = r_params[:, 26:]          

            global_pose_rotation = batch_rodrigues(torch.from_numpy(global_pose))[0].view(3,3).numpy()
            new_global_pose_rotation = extrinsic_matrix[:3,:3] @ global_pose_rotation
            r = R.from_matrix(new_global_pose_rotation)
            new_global_pose = r.as_rotvec()
            new_pose_params = np.concatenate((new_global_pose[np.newaxis, :], hand_pose), axis=1)
            new_shape_params = np.array(shape_params)

            mano_hand_verts, mano_hand_joints = mano_layer(torch.from_numpy(new_pose_params).float().to(device), 
                                                        torch.from_numpy(new_shape_params).float().to(device))
            mano_hand_verts = mano_hand_verts/1000

            mano_hand_verts = mano_hand_verts.squeeze().to(cpu).numpy()
            # print(cam_vertices - mano_hand_verts)
            new_translation = np.mean(cam_vertices - mano_hand_verts, 0)
            # print(new_translation)
            # print(cam_vertices - (mano_hand_verts+new_translation[np.newaxis, :]))
            cam_mano_r_params = np.concatenate((new_translation.squeeze(), new_pose_params.squeeze(), new_shape_params.squeeze()))
            
            if if_save_3d:
                np.savetxt(os.path.join(cam_vertex_save_dir, camera_id+'.txt'), cam_vertices)
                np.savetxt(os.path.join(cam_mano_save_dir, camera_id+'.txt'), cam_mano_r_params)
