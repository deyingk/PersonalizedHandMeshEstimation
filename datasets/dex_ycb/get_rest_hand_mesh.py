"""
"""
import os
import numpy as np
import torch
import torch.nn as nn
from manopth.manolayer import ManoLayer
import pickle
import yaml

mano_layer = ManoLayer(mano_root='../../template', side='right', use_pca=True, ncomps=20, flat_hand_mean=False)

data_dir = '../../data/dex_ycb'
annotation_path = os.path.join(data_dir, 'calibration')
rest_pose_template = {}
unique_shapes = {}
for file_name in os.listdir(annotation_path):
    if not file_name.startswith('mano'):
        continue
    subject = int(file_name.split('_')[-2].split('-')[-1])
    mano_calib_file = os.path.join(annotation_path, file_name, "mano.yml")
    with open(mano_calib_file, 'r') as f:
        mano_calib = yaml.load(f, Loader=yaml.FullLoader)
    shape = np.array(mano_calib['betas'])
    unique_shapes[subject] = shape

    # feed into mano model
    shape = torch.from_numpy(shape).float().unsqueeze(0)
    pose = torch.zeros(1,20+3)
    out_mesh, joints = mano_layer(pose, shape)
    rest_pose_template[subject] = out_mesh.squeeze().numpy()/1000 # rescale to meter

with open(os.path.join(data_dir, 'split_annotations', 'rest_pose_template.pkl'), 'wb') as f:
    pickle.dump(rest_pose_template,f)

with open(os.path.join(data_dir,'split_annotations', 'unique_shapes_dex_ycb.pkl'), 'wb') as f:
    pickle.dump(unique_shapes,f)
