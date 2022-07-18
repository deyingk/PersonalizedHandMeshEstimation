import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import os.path as osp
import torch
import torch.backends.cudnn as cudnn
from src.cmr_pg_pp import CMR_PG_PP
from src.mano_based_model import *
from src.cmr_2d_pose import *
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
    args.backbone = 'ResNet18'
    args.model = 'cmr_2d_pose'
    data_fp = osp.join(args.work_dir, 'data', args.dataset)
    args.batch_size = 32
    args.model_dir = 'out/dex_ycb/reproduce_2d_model'
    
    test_or_val = 'test'

    # device set
    if args.device_idx=='cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    torch.set_num_threads(args.n_threads)

    # deterministic
    cudnn.benchmark = True
    cudnn.deterministic = True

    template_fp = osp.join(args.work_dir, 'template', 'template.ply')
    transform_fp = osp.join(args.work_dir, 'template', 'transform.pkl')
    spiral_indices_list, down_transform_list, up_transform_list, tmp = spiral_tramsform(transform_fp, template_fp, args.ds_factors, args.seq_length, args.dilation)


    # model
    if args.model == 'cmr_2d_pose':
        model = CMR_PG_PP_2D_Pose(args)
    else:
        raise('not implemented')


    check_point = torch.load(osp.join(args.work_dir, args.model_dir, 'checkpoints', 'checkpoint_best.pt'))
    print('loading epoch {}'.format(check_point['epoch']))
    model.load_state_dict(check_point['model_state_dict'])

    # model = model.to(device)
    # if device == torch.device('cuda'):
        # model = torch.nn.DataParallel(model).cuda()
    model = torch.nn.DataParallel(model).cuda()

    # training data
    dataset = DEX_YCB(data_fp, test_or_val, args, tmp['face'], writer=writer,
                        down_sample_list=down_transform_list,  img_std=args.img_std, img_mean=args.img_mean, ms=args.ms_mesh, which_split=args.dex_ycb_which_split)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=False, num_workers=16, drop_last=True)

    
    def uv_map_to_coord_conf(uv_map):
        bs, joints, h, w = uv_map.shape[0], uv_map.shape[1], uv_map.shape[2], uv_map.shape[3]
        idx = torch.argmax(uv_map.view(bs, joints, -1), dim=-1)
        confidence, _ = torch.max(uv_map.view(bs, joints, -1), dim=-1)
        y = idx // w
        x = idx % w
        return torch.stack((x,y, confidence), -1)

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
    predictions = {'subject_ids':[],
                    'image_relative_paths':[],
                    'uv_pred':[],
                    'uv_gt':[],
                    'crop_bbox':[],
                    }
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            print(step)
            # if step > 60:
            #     break
            data = phrase_data(data)
            out = model(data['img'])


            predictions['uv_pred'].append(uv_map_to_coord_conf(out['uv_pred']).cpu().numpy())
            predictions['uv_gt'].append(uv_map_to_coord_conf(data['uv_gt']).cpu().numpy())
            predictions['crop_bbox'].append(data['crop_bbox'].cpu().numpy())

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
    
    utils.makedirs(osp.join(args.model_dir,test_or_val))
    with open(osp.join(args.model_dir,test_or_val,'predictions_2d_on_'+test_or_val+'_set.pkl'),'wb') as f:
        pickle.dump(predictions,f)
