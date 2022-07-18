'''
The goal of this script is to:

Collect the annotations of images into a single pickle file for each subject.
The content in the pickle file is a list of lists:
[ [image_rela_path, uv_annotation, keypoints_3d, mesh, mano, cam_keypoints_3d, cam_mesh, cam_mano],
  [image_rela_path, uv_annotation, keypoints_3d, mesh, mano, cam_keypoints_3d, cam_mesh, cam_mano],
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
final_annotation_dir = '../../data/humbi/annotations'
create_dir(final_annotation_dir)

humbi_regressor = np.load('../../template/humbi_j_regressor.npy')

for subject in sorted(os.listdir(data_dir)):
    if 'subject' not in subject:
        continue
    print('....processing....', subject)
    if subject == 'subject_52':
        # the image folders under subject_52 are actually empty
        continue
    subject_dir = os.path.join(data_dir, subject)
    anno = []
    hand_dir = os.path.join(subject_dir, 'hand')

    all_frames = sorted(os.listdir(hand_dir))
    all_frame_directories = []
    for item in all_frames:
        if os.path.isdir(os.path.join(hand_dir, item)):
            all_frame_directories.append(os.path.join(hand_dir, item))

    for frame_dir in all_frame_directories:
        # get 3D keypoints

        uv_save_dir = os.path.join(frame_dir, 'reconstruction/keypoints_r_2d')
        cam_mano_dir = os.path.join(frame_dir, 'reconstruction/cam_mano_r')
        cam_vertex_dir = os.path.join(frame_dir, 'reconstruction/cam_vertex_r')

        keypoints_3d = np.loadtxt(os.path.join(frame_dir, 'reconstruction/keypoints_r.txt'))
        mesh = np.loadtxt(os.path.join(frame_dir, 'reconstruction/vertices_r.txt'))
        mano = np.loadtxt(os.path.join(frame_dir, 'reconstruction/mano_params_r.txt'))

        right_hand_image_dir = os.path.join(frame_dir, 'image_cropped', 'right')
        images_list = sorted(os.listdir(os.path.join(frame_dir, 'image_cropped', 'right')))
        images_list = list(filter(lambda x: x.endswith('.png'), images_list))
        
        for image_name in images_list:
            image_rela_path = '/'.join([subject, 'hand', frame_dir.split('/')[-1], 'image_cropped', 'right', image_name])
            camera_id = image_name[5:-4]
            # load uv_annotation
            uv_annotation = np.load(os.path.join(uv_save_dir, camera_id+'.npy'))
            # anno.append([image_rela_path, uv_annotation, keypoints_3d, mesh, mano])

            cam_mesh = np.loadtxt(os.path.join(cam_vertex_dir, camera_id+'.txt'))
            cam_mano = np.loadtxt(os.path.join(cam_mano_dir, camera_id+'.txt'))
            cam_keypoints_3d = humbi_regressor @ cam_mesh
            # anno.append({
            #     'image_rela_path':image_rela_path,
            #     'uv': uv_annotation,
            #     'xyz_world': keypoints_3d,
            #     'mesh_world':mesh,
            #     'mano_world':mano,
            #     'xyz_cam':cam_keypoints_3d,
            #     'mesh_cam':cam_mesh,
            #     'mano_cam': cam_mano,

            # })
            anno.append([image_rela_path, uv_annotation, keypoints_3d, mesh, mano, cam_keypoints_3d, cam_mesh, cam_mano])
    
    with open(os.path.join(final_annotation_dir, subject+'_anno.pkl'), 'wb') as f:
        pickle.dump(anno, f)
    



