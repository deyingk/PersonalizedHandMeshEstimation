import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
import torch
import pickle
import numpy as np
from manopth.manolayer import ManoLayer

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def load_model(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['network'])
    return model

def main():

    model_dir = 'out/dex_ycb/reproduce_train_conf_branch'
    data_path = 'out/dex_ycb/reproduce_train_conf_branch/test/predictions_on_test_set.pkl'
    save_folder = os.path.join(model_dir, 'calibrated_shapes')

    calibrate_randomly = True
    sort_with_pred_quality = False

    make_folder(save_folder)
    with open(data_path, 'rb') as f:
        annos = pickle.load(f)
    print(annos.keys())
    print(len(annos['subject_ids']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meshes = torch.from_numpy(annos['pred_meshes']).float().to(device)
    gt_meshes = torch.from_numpy(annos['gt_meshes']).float().to(device)
    subject_ids = torch.from_numpy(annos['subject_ids']).float().to(device)
    unique_subject_ids = np.unique(annos['subject_ids'])

    print(meshes.shape)
    print(torch.mean(torch.sum((meshes - gt_meshes)**2, dim=-1) **0.5)*0.2*1000)

    out_shape = annos['pred_shape_params']
    print('---------------')
    out_shape_dict = {}
    for subject_id in unique_subject_ids:
        same_subject_mask = subject_ids.cpu().numpy()==subject_id
        this_subject_shape = np.array(out_shape[same_subject_mask])
        out_shape_dict[subject_id] = this_subject_shape
        print('subject ', subject_id, '----', len(out_shape_dict[subject_id]))
    
    with open(os.path.join(save_folder, 'shape_all_subjects.pkl'), 'wb') as f:
        pickle.dump(out_shape_dict, f, protocol=pickle.HIGHEST_PROTOCOL)


    calibrated_shape_dict = {}
    for key, value in out_shape_dict.items():
        calibrated_shape_dict[key] = np.mean(value, 0)
        print(key, calibrated_shape_dict[key])
    # with open(os.path.join(save_folder, 'calibrated_shape_all_subjects_small.pkl'), 'wb') as f:
    with open(os.path.join(save_folder, 'calibrated_shape_all_subjects.pkl'), 'wb') as f:
        pickle.dump(calibrated_shape_dict, f, protocol=pickle.HIGHEST_PROTOCOL)    



    # generate calibrated rest hand mesh
    mano_layer = ManoLayer(mano_root='template', side='right', use_pca=True, ncomps=20, flat_hand_mean=False)

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
    with open(os.path.join(save_folder, 'calibrated_rest_pose_template.pkl'), 'wb') as f:
        pickle.dump(calibrated_rest_pose_template,f)

    # partially calibrated shape, for purpose of ablation study

    if sort_with_pred_quality:

        for cali_number in [1,2,3,4,5,10, 20, 50, 100, 200]:
            partially_calibrated_shape_dict = {}
            for key, value in out_shape_dict.items():
                partially_calibrated_shape_dict[key] = np.mean(value[:cali_number, :], 0)
                print(key, partially_calibrated_shape_dict[key])
            with open(os.path.join(save_folder, 'calibrated_best_{}_shape_all_subjects.pkl'.format(cali_number)), 'wb') as f:
                pickle.dump(partially_calibrated_shape_dict, f, protocol=pickle.HIGHEST_PROTOCOL) 


            # generate calibrated rest hand mesh
            mano_layer = ManoLayer(mano_root='template', side='right', use_pca=True, ncomps=20, flat_hand_mean=False)

            partially_calibrated_rest_pose_template = {}
            for subject, shape in partially_calibrated_shape_dict.items():
                print('---', shape.shape)
                shape = torch.from_numpy(shape).float().unsqueeze(0)
                pose = torch.zeros(1,20+3)
                out_mesh, joints = mano_layer(pose, shape)
                partially_calibrated_rest_pose_template[subject] = out_mesh.squeeze().numpy()/1000 # rescale to meter    

            # also add mean shape to the dictionary
            mean_shape = torch.zeros(1, 10)
            pose = torch.zeros(1, 20+3)
            out_mesh, joints = mano_layer(pose, mean_shape)
            partially_calibrated_rest_pose_template['mean_shape'] = out_mesh.squeeze().numpy()/1000

            with open(os.path.join(save_folder, 'calibrated_best_{}_rest_pose_template.pkl'.format(cali_number)), 'wb') as f:
                pickle.dump(partially_calibrated_rest_pose_template,f)




    # partially calibrated shape, for purpose of ablation study
    if calibrate_randomly:
        np.random.seed(0)
        for key, value in out_shape_dict.items():
            np.random.shuffle(value)

        for cali_number in [1,2,3,4,5,10, 20, 50, 100, 200]:
            partially_calibrated_shape_dict = {}
            for key, value in out_shape_dict.items():
                partially_calibrated_shape_dict[key] = np.mean(value[:cali_number, :], 0)
                print(key, partially_calibrated_shape_dict[key])
            if calibrate_randomly:
                with open(os.path.join(save_folder, 'calibrated_randomly_{}_shape_all_subjects.pkl'.format(cali_number)), 'wb') as f:
                    pickle.dump(partially_calibrated_shape_dict, f, protocol=pickle.HIGHEST_PROTOCOL)    
            else:
                with open(os.path.join(save_folder, 'calibrated_{}_shape_all_subjects.pkl'.format(cali_number)), 'wb') as f:
                    pickle.dump(partially_calibrated_shape_dict, f, protocol=pickle.HIGHEST_PROTOCOL)    


            # generate calibrated rest hand mesh
            mano_layer = ManoLayer(mano_root='template', side='right', use_pca=True, ncomps=20, flat_hand_mean=False)

            partially_calibrated_rest_pose_template = {}
            for subject, shape in partially_calibrated_shape_dict.items():
                print('---', shape.shape)
                shape = torch.from_numpy(shape).float().unsqueeze(0)
                pose = torch.zeros(1,20+3)
                out_mesh, joints = mano_layer(pose, shape)
                partially_calibrated_rest_pose_template[subject] = out_mesh.squeeze().numpy()/1000 # rescale to meter    

            # also add mean shape to the dictionary
            mean_shape = torch.zeros(1, 10)
            pose = torch.zeros(1, 20+3)
            out_mesh, joints = mano_layer(pose, mean_shape)
            partially_calibrated_rest_pose_template['mean_shape'] = out_mesh.squeeze().numpy()/1000

            if calibrate_randomly:
                with open(os.path.join(save_folder, 'calibrated_randomly_{}_rest_pose_template.pkl'.format(cali_number)), 'wb') as f:
                    pickle.dump(partially_calibrated_rest_pose_template,f)
            else:
                with open(os.path.join(save_folder, 'calibrated_{}_rest_pose_template.pkl'.format(cali_number)), 'wb') as f:
                    pickle.dump(partially_calibrated_rest_pose_template,f)




if __name__ == '__main__':
    main()