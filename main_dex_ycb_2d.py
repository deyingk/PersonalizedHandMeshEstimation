import os
import os.path as osp
import torch
import torch.backends.cudnn as cudnn
from src.mano_based_model import *
from src.cmr_2d_pose import *
from utils.read import spiral_tramsform
from utils import utils, writer
from options.base_options import BaseOptions
from datasets.dex_ycb.dex_ycb import DEX_YCB
from torch.utils.data import DataLoader
# from run_dex_ycb import Runner
from termcolor import cprint
from run_dex_ycb_2d import Runner
from tensorboardX import SummaryWriter
import json
import random
import numpy as np



if __name__ == '__main__':
    # get config
    args = BaseOptions().parse()
    # dir prepare
    args.work_dir = osp.dirname(osp.realpath(__file__))
    data_fp = osp.join(args.work_dir, 'data', args.dataset)
    args.out_dir = osp.join(args.work_dir, 'out', args.dataset, args.exp_name)
    args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')
    if args.phase in ['eval', 'demo']:
        utils.makedirs(osp.join(args.out_dir, args.phase))
    utils.makedirs(args.out_dir)
    utils.makedirs(args.checkpoints_dir)

    #save args file
    with open(osp.join(args.out_dir, 'args.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # device set
    if args.device_idx=='cpu' or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    torch.set_num_threads(args.n_threads)

    # deterministic
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    template_fp = osp.join(args.work_dir, 'template', 'template.ply')
    transform_fp = osp.join(args.work_dir, 'template', 'transform.pkl')
    spiral_indices_list, down_transform_list, up_transform_list, tmp = spiral_tramsform(transform_fp, template_fp, args.ds_factors, args.seq_length, args.dilation)
    print(len(spiral_indices_list))

    feed_both = False
    feed_template = False
    feed_shape_vector = False
    # model
    if args.model == 'cmr_2d_pose':
        print('Initializing CMR_PG_PP_2D_Pose')
        model = CMR_PG_PP_2D_Pose(args)
        feed_template = False
        feed_shape_vector = False
    else:
        raise Exception('Model {} not support'.format(args.model))

    # load
    epoch = 0
    if args.resume:
        if len(args.resume.split('/')) > 1:
            model_path = args.resume
        else:
            model_path = osp.join(args.checkpoints_dir, args.resume)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', -1) + 1
        cprint('Load checkpoint {}'.format(model_path), 'yellow')     

    if device == torch.device('cuda'):
        model = torch.nn.DataParallel(model).cuda()
    # model = model.to(device)

    # run
    runner = Runner(args, model, tmp['face'], device, feed_template = feed_template, 
                                                        feed_shape_vector = feed_shape_vector,
                                                        feed_both = feed_both)
    print('feed_template', ('_ia' in args.model) and ('warp' not in args.model))
    print('feed_shape_vector', '_ablation' in args.model or 'warp' in args.model)

    if args.phase == 'train':
        # log
        writer = writer.Writer(args)
        writer.print_str(args)
        # dataset

        if args.model.startswith('p'):
            raise 'Not yet'
        else:
            test_dataset = DEX_YCB(data_fp, 'test', args, tmp['face'], down_sample_list=down_transform_list, img_std=args.img_std, 
                                img_mean=args.img_mean, which_split=args.dex_ycb_which_split)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)
        runner.set_test_loader(test_loader)

        if args.model.startswith('p'):
            raise 'Not yet'
        else:
            val_dataset = DEX_YCB(data_fp, 'val', args, tmp['face'], down_sample_list=down_transform_list, img_std=args.img_std, 
                                img_mean=args.img_mean, which_split=args.dex_ycb_which_split)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)
        runner.set_eval_loader(val_loader)    
        
        if args.model.startswith('p'):
            raise 'Not yet'
            # train_dataset = HUMBI_Group(data_fp, 'training', args, tmp['face'], writer=writer,
                                #  down_sample_list=down_transform_list,  img_std=args.img_std, img_mean=args.img_mean, 
                                #  ms=args.ms_mesh)
        else:
            train_dataset = DEX_YCB(data_fp, 'training', args, tmp['face'], writer=writer,
                                 down_sample_list=down_transform_list,  img_std=args.img_std, img_mean=args.img_mean, 
                                 ms=args.ms_mesh, which_split=args.dex_ycb_which_split)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=16, drop_last=True)
        
        print('------------------ loader succeeded--------')
        # optimize
        if args.model.startswith('p'):
            optimizer = torch.optim.Adam(model.module.base_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            print('---setting special optimizer for personalized model------')
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.decay_step, gamma=args.lr_decay)
        # tensorboard
        board = SummaryWriter(osp.join(args.out_dir, 'board'))
        runner.set_train_loader(train_loader, args.epochs, optimizer, scheduler, writer, board, start_epoch=epoch)
        runner.train()
    elif args.phase == 'eval':
        # dataset
        " The code below has never been tested!"
        val_dataset = DEX_YCB(data_fp, 'test', args, tmp['face'], down_sample_list=down_transform_list, img_std=args.img_std, 
                            img_mean=args.img_mean, which_split=args.dex_ycb_which_split)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=0)
        runner.set_eval_loader(val_loader)
        runner.evaluation()

    elif args.phase == 'demo':
        runner.demo()
    else:
        raise Exception('phase error')
