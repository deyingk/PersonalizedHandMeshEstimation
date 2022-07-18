# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""DexYCB dataset."""

import os
import yaml
import numpy as np
import torch
from manopth.manolayer import ManoLayer
import pickle
from PIL import Image

import random
random.seed(0)

anno_dir = '../../data/dex_ycb/split_annotations'
with open(os.path.join(anno_dir, 's0_train_size_2.pkl'), 'rb') as f:
    train_anno = pickle.load(f)

with open(os.path.join(anno_dir, 's0_val_size_2.pkl'), 'rb') as f:
    val_anno = pickle.load(f)

with open(os.path.join(anno_dir, 's0_test_size_2.pkl'), 'rb') as f:
    test_anno = pickle.load(f)

all_anno = []
all_anno.extend(train_anno)
all_anno.extend(val_anno)
all_anno.extend(test_anno)

print(len(all_anno), len(train_anno), len(val_anno), len(test_anno))

random.shuffle(all_anno)
total_length = len(all_anno)
training_anno = all_anno[: int(0.8*total_length)]
val_anno = all_anno[int(0.8*total_length):int(0.85*total_length)]
test_anno = all_anno[int(0.85*total_length):]

with open(os.path.join(anno_dir, 's4_train_size_2.pkl'), 'wb') as f:
    pickle.dump(training_anno, f)

with open(os.path.join(anno_dir, 's4_val_size_2.pkl'), 'wb') as f:
    pickle.dump(val_anno, f)

with open(os.path.join(anno_dir, 's4_test_size_2.pkl'), 'wb') as f:
    pickle.dump(test_anno, f)

print(len(all_anno), len(training_anno), len(val_anno), len(test_anno))
