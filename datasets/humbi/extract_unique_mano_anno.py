'''
The goal of this script is to:

Extract unique mano annos from humbi dataset.
The content in the pickle file is a list of lists:
[ [keypoints_3d, mesh, mano],
  [keypoints_3d, mesh, mano],
  ...,
]

'''
import os
import numpy as np
import pickle

def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

data_dir = '../../data/humbi'
final_annotation_dir = '../../data/humbi/annotations_unique_mano'
create_dir(final_annotation_dir)

all_anno = []
all_shape_anno = []
all_mano_params = []

for subject in sorted(os.listdir(data_dir)):
    if 'subject' not in subject:
        continue
    print('....processing....', subject)
    if subject == 'subject_52':
        # the image folders under subject_52 are actually empty
        continue
    subject_dir = os.path.join(data_dir, subject)
    subject_anno = []
    hand_dir = os.path.join(subject_dir, 'hand')

    all_frames = sorted(os.listdir(hand_dir))
    all_frame_directories = []
    for item in all_frames:
        if os.path.isdir(os.path.join(hand_dir, item)):
            all_frame_directories.append(os.path.join(hand_dir, item))

    for frame_dir in all_frame_directories:
        # get 3D keypoints

        uv_save_dir = os.path.join(frame_dir, 'reconstruction/keypoints_r_2d')

        keypoints_3d = np.loadtxt(os.path.join(frame_dir, 'reconstruction/keypoints_r.txt'))
        mesh = np.loadtxt(os.path.join(frame_dir, 'reconstruction/vertices_r.txt'))
        mano = np.loadtxt(os.path.join(frame_dir, 'reconstruction/mano_params_r.txt'))
        all_mano_params.append(mano)
        subject_anno.append([keypoints_3d, mesh, mano])
    all_shape_anno.append(mano[-10:])
    all_anno.extend(subject_anno)

# print(all_shape_anno)
all_mano_params = np.stack(all_mano_params)
all_shape_anno = np.stack(all_shape_anno)
mano_mean = np.mean(all_mano_params, 0)
mano_std = np.std(all_mano_params, 0)
print('mano_mean', mano_mean)
print('mano_std', mano_std)
humbi_mano_stats = {
    'all_shapes': all_shape_anno,
    'mano_mean': mano_mean,
    'mano_std': mano_std
}

with open(os.path.join(final_annotation_dir, 'all_mano_anno.pkl'), 'wb') as f:
    pickle.dump(all_anno, f)
with open(os.path.join(final_annotation_dir, 'humbi_mano_stats.pkl'), 'wb') as f:
    pickle.dump(humbi_mano_stats, f)


