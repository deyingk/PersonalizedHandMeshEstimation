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


annotation_path = '../../data/humbi/annotations'
hand_shape_dict = {}
for file_name in os.listdir(annotation_path):
    subject = int(file_name.split('_')[1])
    with open(os.path.join(annotation_path, file_name),'rb') as f:
        anno = pickle.load(f)
        shape = torch.from_numpy(anno[0][-1][-10:])
        hand_shape_dict[subject] = shape
# also add mean shape to the dictionary
with open(os.path.join('../../data/humbi/split_annotations', 'gt_hand_shapes.pkl'), 'wb') as f:
    pickle.dump(hand_shape_dict, f)
