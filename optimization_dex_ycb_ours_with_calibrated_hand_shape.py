import pickle
import shutil
import numpy as np
import os
import yaml
import torch
from manopth.manolayer import ManoLayer
import numpy as np
import pickle
import os
import torch
import yaml
import time
from torch.utils.tensorboard import SummaryWriter

import cv2
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from src.rotation_conversions import *

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def pose_6d_to_axis_angle(pose_6d_vector):
    pose_6d = pose_6d_vector.view(-1, 16, 6)
    pose_matrix = rotation_6d_to_matrix(pose_6d)
    pose_axisang = matrix_to_axis_angle(pose_matrix).view(-1, 48)
    return pose_axisang


use_ground_truth_2d = False
experiment_name = 'reproduce_ours_with_calibrated_hand_shape_and_optimization'
if_save = False
save_file_name = '#####' # set the save file name here.

start_from_flat = False

lr_stage_1 = 1e-2
epochs_stage_1 = 200
lr_stage_2_root = 1e-3
lr_stage_2_pose = 1e-3
epochs_stage_2 = 100
milestones = [400]


which_optimizer = 'Adam'  #'Adam' or 'SGD'
with_gt_shape = False
with_calibrated_shape = True
with_baseline_pred_shape = False
with_boukhayma = False
assert sum((with_gt_shape, with_calibrated_shape, with_baseline_pred_shape, with_boukhayma)) == 1

use_confidence = False
confidence_threshold = 0.1



file_predictions_2d ='out/dex_ycb/reproduce_2d_model/test/predictions_2d_on_test_set.pkl'
if with_gt_shape:
    file_predictions_mesh = 'out/dex_ycb/reproduce_mano_our_model_with_gt_shape/test_with_groundtruth/predictions_on_test_set.pkl'
elif with_calibrated_shape:
    file_predictions_mesh = 'out/dex_ycb/reproduce_mano_our_model_with_gt_shape/test_with_calibrated/predictions_on_test_set.pkl'
elif with_baseline_pred_shape:
    pass
elif with_boukhayma:
    pass

file_intrinsic = 'out/dex_ycb/reproduce_2d_model/intrinsics_test.pt'


data_dir = 'data/dex_ycb'

run_dir = 'out/dex_ycb_with_optimization/' + experiment_name
writer = SummaryWriter(run_dir)
shutil.copy(__file__, run_dir+'/running_script') 

"""get 2d predictions"""

with open(file_predictions_2d, 'rb') as f:
    predictions_2d = pickle.load(f)

crop_width = predictions_2d['crop_bbox'][:,2] - predictions_2d['crop_bbox'][:,0]
crop_height = predictions_2d['crop_bbox'][:,3] - predictions_2d['crop_bbox'][:,1]

pred_2d_on_orig_img = predictions_2d['uv_pred'].copy()
pred_2d_on_orig_img[:,:,:2] = pred_2d_on_orig_img[:,:,:2]/112*crop_width[:,np.newaxis, np.newaxis]
pred_2d_on_orig_img[:,:,0] = pred_2d_on_orig_img[:,:,0] + predictions_2d['crop_bbox'][:,0, np.newaxis]
pred_2d_on_orig_img[:,:,1] = pred_2d_on_orig_img[:,:,1] + predictions_2d['crop_bbox'][:,1, np.newaxis]

gt_2d_on_orig_img = predictions_2d['uv_gt'].copy()
gt_2d_on_orig_img[:,:,:2] = gt_2d_on_orig_img[:,:,:2]/112*crop_width[:,np.newaxis, np.newaxis]
gt_2d_on_orig_img[:,:,0] = gt_2d_on_orig_img[:,:,0] + predictions_2d['crop_bbox'][:,0, np.newaxis]
gt_2d_on_orig_img[:,:,1] = gt_2d_on_orig_img[:,:,1] + predictions_2d['crop_bbox'][:,1, np.newaxis]

loss_mask = (predictions_2d['uv_pred'][:,:, -1] > confidence_threshold)

with open(file_predictions_mesh, 'rb') as f:
    predictions = pickle.load(f)
dex_ycb_j_regressor = torch.from_numpy(np.load('template/dex_ycb_j_regressor.npy')).float().unsqueeze(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model
mano_layer_1 = ManoLayer(mano_root='template', side='right', use_pca=False, ncomps=45, flat_hand_mean=True)
mano_layer_pca = ManoLayer(mano_root='template', side='right', use_pca=True, ncomps=45, flat_hand_mean=False)

gt_pose = torch.from_numpy(predictions['gt_pose_params']).float()

if not start_from_flat:
    pred_pose = torch.from_numpy(predictions['pred_pose_params']).float()
else:
    batch_size = gt_pose.shape[0]
    pred_pose = torch.rand(batch_size, 16*6)

if with_baseline_pred_shape:
    gt_shape = torch.from_numpy(predictions['pred_shape_params']).float()
else:
    gt_shape = torch.from_numpy(predictions['gt_shape_params']).float()
gt_mano_translations = torch.from_numpy(predictions['mano_translations']).float()
gt_meshes = torch.from_numpy(predictions['gt_meshes']).float()*0.2 + gt_mano_translations.unsqueeze(1)
loss_mask = torch.from_numpy(loss_mask)



def load_camera_intrinsic_params(param_path):
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
        project_matrix = np.zeros((3, 3))
        current_row = 0
        while current_row < 3:
            entries = contents[current_line].strip().split(' ')
            for i in range(3):
                project_matrix[current_row, i] = float(entries[i])
            current_row += 1
            current_line += 1
        camera_dict[camera_id.zfill(7)] = project_matrix
    return camera_dict

n = len(predictions['image_relative_paths'])

if os.path.exists(file_intrinsic):
    intrinsic_matrices = torch.load(file_intrinsic)
else:
    intrinsic_matrices = []

    for idx in range(n):
        print(idx)

        image_relative_path = predictions['image_relative_paths'][idx]
        image_relative_path = '/'.join(image_relative_path.split('/')[1:])

        image_path = os.path.join(data_dir, image_relative_path)

        # get intrinsics
        image_relative_path
        s = image_relative_path.split('/')[-2]

        intr_file = os.path.join(data_dir, 'calibration', "intrinsics", "{}_640x480.yml".format(s))
        with open(intr_file, 'r') as f:
            intr = yaml.load(f, Loader=yaml.FullLoader)
        intr = intr['color']
        fx = intr['fx']
        fy = intr['fy']
        ppx = intr['ppx']
        ppy = intr['ppy']
        intrinsic_matrices.append(np.array([[fx, 0, ppx],
                                            [0, fy, ppy],
                                            [0, 0, 1]
                                                    ]))

    intrinsic_matrices = np.array(intrinsic_matrices)
    intrinsic_matrices = torch.from_numpy(intrinsic_matrices).float()
    torch.save(intrinsic_matrices, file_intrinsic)

_, gt_keypoints_3d_direct = mano_layer_pca(gt_pose, gt_shape)

gt_keypoints_3d_direct = gt_keypoints_3d_direct/1000 + gt_mano_translations.unsqueeze(1)

gt_keypoints_3d_direct = gt_keypoints_3d_direct.to(device)
def batch_project(intrinsics, joints_3d):
    """
    intrinsics, bs x 3 x 3
    joints_3d, bs x 21 x 3
    """
    joints_3d = torch.transpose(joints_3d, 1, 2)
    joints_2d = torch.bmm(intrinsics, joints_3d)
    joints_2d = joints_2d / (joints_2d[:,-1:, :] + 1e-8)
    joints_2d = joints_2d[:,:2,:]
    joints_2d = torch.transpose(joints_2d, 2,1)
    return joints_2d


mano_layer_1_cuda = mano_layer_1.to(device)
gt_shape = gt_shape.to(device)
gt_mano_translations = gt_mano_translations.to(device)
intrinsic_matrices = intrinsic_matrices.to(device)
# get gt 2d_direct
gt_2d_direct = batch_project(intrinsic_matrices, gt_keypoints_3d_direct)

if use_ground_truth_2d:
    target_2d = gt_2d_direct.to(device)
else:
    target_2d = torch.from_numpy(pred_2d_on_orig_img[:,:,:2]).float().to(device)

loss_mask = loss_mask.to(device)

def pose_to_2d(pose, mano_translations):
    mesh, keypoints_3d = mano_layer_1_cuda(pose_6d_to_axis_angle(pose), gt_shape)
    keypoints_3d = keypoints_3d/1000 + mano_translations.unsqueeze(1)
    pose_2d = batch_project(intrinsic_matrices, keypoints_3d)
    return mesh, pose_2d

def map_z_to_cam_3d(Z, root_2d, intrinsic_matrices):
    X = Z * (root_2d[:,0] - intrinsic_matrices[:,0,2])/(intrinsic_matrices[:,0,0])
    Y = Z * (root_2d[:,1] - intrinsic_matrices[:,1,2])/(intrinsic_matrices[:,1,1])
    return torch.stack((X, Y, Z), dim=-1)


# starting time
start = time.time()

""" stage 1, optimize root position """

root_2d = target_2d[:,0,:]
z_var = 0.85*torch.ones(root_2d.shape[0]).to(device)
pose_var = pred_pose.detach()
pose_var = pose_var.to(device)
# pose_var.requires_grad = True
pose_var.requires_grad = False

# initial_mano_translations = torch.tensor([[-0.0292309 ,  0.14668225,  1.2578207]]).repeat(pose_var.shape[0], 1)
initial_mano_translations = map_z_to_cam_3d(z_var, root_2d, intrinsic_matrices)
mano_translations_var = initial_mano_translations.detach().to(device)
mano_translations_var.requires_grad = True

loss_fn = torch.nn.MSELoss(reduction='mean')

# optimizer = torch.optim.Adam([pose_var])
if which_optimizer == 'Adam':
    optimizer = torch.optim.Adam([mano_translations_var], lr=lr_stage_1)
elif which_optimizer =='SGD':
    optimizer = torch.optim.SGD([mano_translations_var], lr=lr_stage_1)
else:
    raise "optimizer not supported!"

for iter_i in range(epochs_stage_1):
    pred_mesh, pred_pose_2d = pose_to_2d(pose_var, mano_translations_var)
    if use_confidence:
        loss = loss_fn(torch.masked_select(pred_pose_2d, loss_mask.unsqueeze(-1).repeat(1,1,2)),
                        torch.masked_select(target_2d, loss_mask.unsqueeze(-1).repeat(1,1,2)))
    else:
        loss = loss_fn(pred_pose_2d, target_2d)

    error = (pred_mesh.cpu().detach().numpy() - predictions['gt_meshes']*0.2*1000)
    mjpve = np.mean(np.sum(error**2, axis=-1)**0.5)

    print(iter_i, loss.item(), mjpve)
    # print(torch.mean(torch.sum((mano_translations_var - gt_mano_translations)**2, dim=-1)**0.5))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    writer.add_scalar('loss', loss.item(), iter_i)
    writer.add_scalar('mjpve', mjpve, iter_i)



""" stage 2, optimize all parameters """
if not start_from_flat:
    pred_pose = torch.from_numpy(predictions['pred_pose_params']).float()
else:
    batch_size = gt_pose.shape[0]
    # pose_6d = torch.zeros(batch_size, 16, 3, 3).to(gt_pose.device)
    # pose_6d += torch.eye(3).unsqueeze(0).unsqueeze(0).to(gt_pose.device)
    # pose_6d = pose_6d[:,:,:2,:]
    # pose_6d = pose_6d.reshape(batch_size, 16*6)
    # pred_pose = pose_6d
    pred_pose = torch.rand(batch_size, 16*6)

pose_var = pred_pose.detach()
pose_var = pose_var.to(device)
pose_var.requires_grad = True
mano_translations_var_second_stage = mano_translations_var.detach()
mano_translations_var_second_stage.requires_grad = True

loss_fn = torch.nn.MSELoss(reduction='mean')

# optimizer = torch.optim.Adam([pose_var], lr=0.001)

if which_optimizer == 'Adam':
    optimizer = torch.optim.Adam([{'params':mano_translations_var_second_stage, 'lr':lr_stage_2_root},
                                {'params': pose_var, 'lr': lr_stage_2_pose}])
elif which_optimizer == 'SGD':
    optimizer = torch.optim.SGD([{'params':mano_translations_var_second_stage, 'lr':lr_stage_2_root},
                                {'params': pose_var, 'lr': lr_stage_2_pose}])
else:
    raise "Optimizer not supported!"
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

gt_rela_mesh = predictions['gt_meshes']*0.2*1000
gt_rela_mesh = torch.from_numpy(gt_rela_mesh).float().to(device)
dex_ycb_j_regressor = dex_ycb_j_regressor.to(device)

for iter_i in range(epochs_stage_1, epochs_stage_1 + epochs_stage_2):
    pred_mesh, pred_pose_2d = pose_to_2d(pose_var, mano_translations_var_second_stage)
    if use_confidence:
        loss = loss_fn(torch.masked_select(pred_pose_2d, loss_mask.unsqueeze(-1).repeat(1,1,2)),
                        torch.masked_select(target_2d, loss_mask.unsqueeze(-1).repeat(1,1,2)))
    else:
        loss = loss_fn(pred_pose_2d, target_2d)
    
    error = (pred_mesh.cpu().detach().numpy() - predictions['gt_meshes']*0.2*1000)
    mjpve = np.mean(np.sum(error**2, axis=-1)**0.5)

    error = (pred_mesh.cpu().detach().numpy() - gt_rela_mesh.cpu().numpy())
    mjpve = np.mean(np.sum(error**2, axis=-1)**0.5)
    
    gt_3d = torch.bmm(dex_ycb_j_regressor.repeat(gt_rela_mesh.shape[0], 1,1), gt_rela_mesh)
    gt_3d = gt_3d - gt_3d[:, :1, :]
    pred_3d =  torch.bmm(dex_ycb_j_regressor.repeat(pred_mesh.shape[0], 1,1), pred_mesh.detach())
    pred_3d = pred_3d - pred_3d[:, :1, :]
    error = gt_3d - pred_3d
    mjpje = np.mean(np.sum(error.cpu().numpy()**2, axis=-1)**0.5)
    
    # get absolute error
    pred_mesh_abs = pred_mesh.detach() + mano_translations_var_second_stage.detach().unsqueeze(1) * 1000
    pred_3d_abs = torch.bmm(dex_ycb_j_regressor.repeat(pred_mesh.shape[0], 1,1), pred_mesh_abs)
    gt_mesh_abs = gt_rela_mesh + gt_mano_translations.unsqueeze(1) * 1000
    gt_3d_abs = torch.bmm(dex_ycb_j_regressor.repeat(pred_mesh.shape[0], 1,1), gt_mesh_abs)

    
    print(iter_i, loss.item(), mjpve, mjpje)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    writer.add_scalar('loss', loss.item(), iter_i)
    writer.add_scalar('mjpje', mjpje, iter_i)
    writer.add_scalar('mjpve', mjpve, iter_i)
    scheduler.step()

if if_save:
    save_predictions = predictions.copy()
    save_predictions['pred_meshes'] = pred_mesh.cpu().detach().numpy()/1000/0.2
    with open(save_file_name, 'wb') as f:
        pickle.dump(save_predictions, f)

# end time
end = time.time()
# total time taken
print(f"Runtime of the program is {end - start}")