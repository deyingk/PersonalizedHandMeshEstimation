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

# get ground truth shape params
gt_shape_dict ={}
annotation_path = '../../data/humbi/annotations'
rest_pose_template = {}
for file_name in os.listdir(annotation_path):
    subject = int(file_name.split('_')[1])
    with open(os.path.join(annotation_path, file_name),'rb') as f:
        anno = pickle.load(f)
        shape = torch.from_numpy(anno[0][-1][-10:]).float().unsqueeze(0)
        gt_shape_dict[subject] = shape

# print(gt_shape_dict)

# get calibrated shape params
with open('../../data/humbi/my_calibrated_hand/pred_meshes_small_test_from_mesh_10m_humbi_on_the_fly_meter_0_2_out_calibrated_shape_all_subjects.pkl', 'rb') as f:
    calibrated_shape_dict = pickle.load(f)
# print(calibrated_shape_dict)

max_range = 0
for shape in gt_shape_dict.values():
    max_range = max(max_range, max(torch.max(shape), torch.max(-1*shape)))
print(max_range)

for subject, shape in calibrated_shape_dict.items():
    print(subject)
    print(np.sum(np.abs(shape.squeeze() - gt_shape_dict[subject].squeeze().numpy()))/max_range)

# print(calibrated_shape_dict[118])
# print(gt_shape_dict[118])

calibrated_rest_pose_template = {}
for subject, shape in calibrated_shape_dict.items():
    print('---', shape.shape)
    shape = torch.from_numpy(shape).float().unsqueeze(0)
    pose = torch.zeros(1,20+3)
    out_mesh, joints = mano_layer(pose, shape)
    calibrated_rest_pose_template[subject] = out_mesh.squeeze().numpy()/1000 # rescale to meter    

# also add mean shape to the dictionary
mean_shape = torch.zeros(1, 10)
pose = torch.zeros(1, 20+3)
out_mesh, joints = mano_layer(pose, mean_shape)
calibrated_rest_pose_template['mean_shape'] = out_mesh.squeeze().numpy()/1000
with open(os.path.join('../../data/humbi/split_annotations', 'calibrated_rest_pose_template.pkl'), 'wb') as f:
    pickle.dump(calibrated_rest_pose_template,f)
