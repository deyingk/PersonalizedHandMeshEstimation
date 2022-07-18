"""
Generata a rest hand mesh, with zero shape parameters.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from manopth.manolayer import ManoLayer
import pickle

mano_layer = ManoLayer(mano_root='../../template', side='right', use_pca=True, ncomps=20, flat_hand_mean=False)

rest_pose_template = {}
# also add mean shape to the dictionary
mean_shape = torch.zeros(1, 10)
pose = torch.zeros(1, 20+3)
out_mesh, joints = mano_layer(pose, mean_shape)
rest_pose_template['mean_shape'] = out_mesh.squeeze().numpy()/1000
with open(os.path.join('../../data/FreiHAND', 'rest_pose_template.pkl'), 'wb') as f:
    pickle.dump(rest_pose_template,f)
