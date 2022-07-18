"""
Assume we already have annotations for each subject in separate pickle files.
I.e., file 'subject_1_anno.pkl' contains all annotations for subject_1.
"""
import os
import numpy as np
import torch
import torch.nn as nn
from manopth.manolayer import ManoLayer
import pickle

mano_layer = ManoLayer(mano_root='../../template', side='right', use_pca=True, ncomps=20, flat_hand_mean=False)

annotation_path = '../../data/humbi/annotations'
rest_pose_template = {}
for file_name in os.listdir(annotation_path):
    subject = int(file_name.split('_')[1])
    with open(os.path.join(annotation_path, file_name),'rb') as f:
        anno = pickle.load(f)
        shape = torch.from_numpy(anno[0][-1][-10:]).float().unsqueeze(0)
        pose = torch.zeros(1,20+3)
        out_mesh, joints = mano_layer(pose, shape)
        rest_pose_template[subject] = out_mesh.squeeze().numpy()/1000 # rescale to meter
# also add mean shape to the dictionary
mean_shape = torch.zeros(1, 10)
pose = torch.zeros(1, 20+3)
out_mesh, joints = mano_layer(pose, mean_shape)
rest_pose_template['mean_shape'] = out_mesh.squeeze().numpy()/1000
with open(os.path.join('../../data/humbi/split_annotations', 'rest_pose_template.pkl'), 'wb') as f:
    pickle.dump(rest_pose_template,f)
