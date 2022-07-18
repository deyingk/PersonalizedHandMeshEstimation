import os
from tracemalloc import stop
os.environ["CUDA_VISIBLE_DEVICES"]='7'
import os.path as osp
from collections import defaultdict
import torch
import torch.backends.cudnn as cudnn
from src.cmr_pg_pp import CMR_PG_PP
from src.mano_based_model import *
from src.boukhayma_model import resnet34_Mano
from utils.read import spiral_tramsform
from utils import utils, writer
from options.base_options import BaseOptions
from datasets.dex_ycb.dex_ycb import DEX_YCB

from torch.utils.data import DataLoader
from termcolor import cprint
from tensorboardX import SummaryWriter
import numpy as np
import pickle

if __name__ == '__main__':
    # get config
    args = BaseOptions().parse()
    # dir prepare
    args.work_dir = osp.dirname(osp.realpath(__file__))
    args.dataset ='dex_ycb'
    args.dex_ycb_which_crop='size_2'
    args.dex_ycb_which_split='s1'
    args.dex_ycb_use_world_or_cam='cam'
    # args.model = 'mano_based_model_iterative_pose_without_gt_shape'
    input_option= 0
    args.size = 256



    data_fp = osp.join(args.work_dir, 'data', args.dataset)
    args.batch_size = 32
    args.model_dir = 'out/dex_ycb/reproduce_boukhayma_model'

    test_or_val = 'test'

    template_fp = osp.join(args.work_dir, 'template', 'template.ply')
    transform_fp = osp.join(args.work_dir, 'template', 'transform.pkl')
    spiral_indices_list, down_transform_list, up_transform_list, tmp = spiral_tramsform(transform_fp, template_fp, args.ds_factors, args.seq_length, args.dilation)


    # device set
    if args.device_idx=='cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    torch.set_num_threads(args.n_threads)

    # deterministic
    cudnn.benchmark = True
    cudnn.deterministic = True

    feed_both = False
    feed_template = False
    feed_shape_vector = False

    
    # model = torch.nn.DataParallel(resnet34_Mano(input_option=input_option)) 
    # model.load_state_dict(torch.load(osp.join(args.work_dir, args.model_dir, 'model-0.pth')))

    model = resnet34_Mano(input_option=input_option)
    check_point = torch.load(osp.join(args.work_dir, args.model_dir, 'checkpoints', 'checkpoint_last_old.pt'))
    print('loading epoch {}'.format(check_point['epoch']))
    model.load_state_dict(check_point['model_state_dict'], strict=False)


    model = torch.nn.DataParallel(model).cuda()

    # training data
    dataset = DEX_YCB(data_fp, test_or_val, args, tmp['face'], writer=writer,
                        down_sample_list=down_transform_list,  img_std=args.img_std, img_mean=args.img_mean, ms=args.ms_mesh, which_split=args.dex_ycb_which_split)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=16, drop_last=True)
    

    def phrase_data( data):
        for key, val in data.items():
            if key == 'meta':
                continue
            if isinstance(val, list):
                data[key] = [d.to(device) for d in data[key]]
            else:
                data[key] = data[key].to(device)
        return data
    
    model.eval()
    predictions = defaultdict(list)
    predictions.update({'subject_ids':[],
                    'image_relative_paths':[],
                    'mano_translations':[],
                    'pred_meshes':[],
                    'gt_xyz_provided': [],
                    'gt_meshes':[],
                    # 'pred_pose_params':[],
                    'gt_shape_params':[],
                    'gt_pose_params':[],
                    })
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            print(step)
            # if step > 60:
            #     break
            data = phrase_data(data)
            out = model(data['img'])
            pred_mesh = out['mesh_pred'][0]

            error = np.mean(np.sum((out['mesh_pred'][0].cpu().numpy() - data['mesh_gt'][0].cpu().numpy())**2, axis=-1)**0.5)*0.2*1000
            print('*'*10, error)
            predictions['pred_meshes'].append(pred_mesh.cpu().numpy())
            predictions['gt_meshes'].append(data['mesh_gt'][0].cpu().numpy())
            predictions['mano_translations'].append(data['mano_translation'].cpu().numpy())
            predictions['gt_shape_params'].append(data['shape_params'].cpu().numpy())
            predictions['gt_pose_params'].append(data['pose_params'].cpu().numpy())
            predictions['gt_xyz_provided'].append(data['xyz_gt_provided'].cpu().numpy())
            if 'mano_pose_pred' in out:
                predictions['pred_pose_params'].append(out['mano_pose_pred'][0].cpu().numpy())
            if 'mano_shape_pred' in out:
                predictions['pred_shape_params'].append(out['mano_shape_pred'][0].cpu().numpy())
            if 'conf_pred' in out:
                predictions['conf_pred'].append(out['conf_pred'].cpu().numpy())
            subject_list = []
            img_rela_path = []
            for meta in data['meta']: 
                subject_list.append(int(meta.split('/')[1].split('-')[-1]))
                img_rela_path.append(meta)
                # subject_list.append(int(meta.split('/')[0].strip('subject_')))
            predictions['image_relative_paths'].extend(img_rela_path)
            predictions['subject_ids'].append(np.array(subject_list))
    for key, value in predictions.items():
        if key != 'image_relative_paths':
            predictions[key] = np.concatenate(value)
            print(predictions[key].shape)
    
    # save_dir = osp.join(args.model_dir, test_or_val+'_with_'+args.template_mode+'_shape_top_20')
    save_dir = osp.join(args.model_dir, test_or_val+'_with_'+args.template_mode)
    utils.makedirs(save_dir)
    print(osp.join(save_dir,'predictions_on_'+test_or_val+'_set.pkl'))
    with open(osp.join(save_dir,'predictions_on_'+test_or_val+'_set.pkl'),'wb') as f:
        pickle.dump(predictions,f)
