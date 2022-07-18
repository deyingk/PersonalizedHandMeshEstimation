import pickle
import os
import numpy as np


anno_dir = '../../data/dex_ycb/split_annotations'
with open(os.path.join(anno_dir, 's0_train.pkl'), 'rb') as f:
    annos = pickle.load(f)

with open(os.path.join(anno_dir, 'unique_shapes_dex_ycb.pkl'), 'rb') as f:
    unique_shapes = pickle.load(f)
unique_shapes = np.stack(unique_shapes.values())

all_anno_to_save = []  # [keypoints_3d, mesh, mano]
all_mano_anno = []
for anno in annos:
    all_anno_to_save.append([anno['joint_3d'], anno['mesh'], anno['mano_params']])
    all_mano_anno.append(anno['mano_params'])

all_mano_anno = np.array(all_mano_anno)
all_mano_pose = all_mano_anno[:,3:48]
pose_mean = np.mean(all_mano_pose, 0)
pose_std = np.std(all_mano_pose, 0)

print(pose_mean)
print(pose_std)
print(np.max(all_mano_pose), np.min(all_mano_pose))

dex_ycb_stats = {
    'all_shapes': unique_shapes,
    'pose_mean': pose_mean,
    'pose_std': pose_std
}

with open(os.path.join(anno_dir, 'all_mano_anno.pkl'), 'wb') as f:
    pickle.dump(all_anno_to_save, f)
with open(os.path.join(anno_dir, 'dex_ycb_mano_stats.pkl'), 'wb') as f:
    pickle.dump(dex_ycb_stats, f)