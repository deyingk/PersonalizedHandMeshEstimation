import os
import torch
from utils.vis import cnt_area
import numpy as np
import cv2
from utils.vis import registration, map2uv, inv_base_tranmsform, base_transform, tensor2array
from utils.draw3d import save_a_image_with_mesh_joints
from utils.read import save_mesh
import json
from datasets.FreiHAND.kinematics import mano_to_mpii
from utils.progress.bar import Bar
from termcolor import colored, cprint
import pickle
import time
from collections import defaultdict
from src.loss import prepare_ranking_data

class Runner(object):
    def __init__(self, args, model, faces, device, feed_template=False, feed_shape_vector=False, feed_both=False):
        assert (feed_template and feed_shape_vector) == False
        super(Runner, self).__init__()
        self.args = args
        self.model = model
        self.faces = faces
        self.device = device
        with open(os.path.join(self.args.work_dir, 'template', 'dex_ycb_j_regressor.npy'), 'rb') as f:
            self.j_regressor = torch.from_numpy(np.load(f)).float().to(self.device)
        self.std = torch.tensor(0.20).to(self.device)
        self.face = torch.from_numpy(self.faces[0].astype(np.int64)).to(self.device)
        self.best_root_rela_mpjpe = float('inf')
        self.feed_template = feed_template
        self.feed_shape_vector = feed_shape_vector
        self.feed_both = feed_both

    def set_train_loader(self, train_loader, epochs, optimizer, scheduler, writer, board, start_epoch=0):
        self.train_loader = train_loader
        self.max_epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer
        self.board = board
        self.start_epoch = start_epoch
        self.epoch = max(start_epoch - 1, 0)
        self.total_step = self.start_epoch * (len(self.train_loader.dataset) // self.writer.args.batch_size)
        # self.loss = self.model.loss
        self.loss = self.model.module.loss

    def set_eval_loader(self, eval_loader):
        self.eval_loader = eval_loader
    
    def set_test_loader(self, test_loader):
        self.test_loader = test_loader

    def calculate_rank_acc(self, conf_pred, mano_shape_pred, mano_shape_gt):
        if isinstance(mano_shape_pred, list):
            mano_shape_pred = mano_shape_pred[0]
        with torch.no_grad():
            conf_pred = conf_pred.squeeze()
            mano_shape_pred = mano_shape_pred.squeeze()
            mano_shape_gt = mano_shape_gt.squeeze()
            l1_distance = torch.mean(torch.abs(mano_shape_gt-mano_shape_pred), dim=1)
            x1, x2, y = prepare_ranking_data(conf_pred.squeeze(), l1_distance.squeeze())
            acc = (x1-x2)*y > 0
            acc = torch.sum(acc).float()/len(y)
        return acc.item()

    def train(self):
        for epoch in range(self.start_epoch, self.max_epochs + 1):
            self.epoch = epoch
            t = time.time()
            train_loss_dict,  train_root_relative_mjpje, train_rank_acc= self.train_a_epoch()
            t_duration = time.time() - t
            self.scheduler.step()
            info = {
                'current_epoch': self.epoch,
                'epochs': self.max_epochs,
                'train_loss': train_loss_dict['loss'],
                't_duration': t_duration
            }
            self.writer.print_info(info)
            for key, value in train_loss_dict.items():
                self.board.add_scalar('train/'+key, value, epoch)
            self.board.add_scalar('train/root_relative_mpjpe', train_root_relative_mjpje.item(), epoch)
            self.board.add_scalar('train/train_rank_acc', train_rank_acc, epoch)
            self.writer.save_checkpoint(self.model.module, self.optimizer, self.scheduler, self.epoch, last=True)

            t = time.time()
            print('evaluating......')
            eval_loss_dict, eval_root_relative_mjpje, eval_root_relative_mpvpe, eval_rank_acc = self.evaluate_on_this_epoch()

            t_duration = time.time() - t
            info = {
                'current_epoch': self.epoch,
                'epochs': self.max_epochs,
                'eval_loss': eval_loss_dict['loss'],
                't_duration': t_duration
            }
            self.writer.print_info_eval(info)
            for key, value in eval_loss_dict.items():
                self.board.add_scalar('eval/'+key, value, epoch)
            self.board.add_scalar('eval/root_relative_mpjpe', eval_root_relative_mjpje.item(), epoch)
            self.board.add_scalar('eval/root_relative_mpvpe', eval_root_relative_mpvpe.item(), epoch)
            self.board.add_scalar('eval/eval_rank_acc', eval_rank_acc, epoch)

            t = time.time()
            print('testing......')
            test_loss_dict, test_root_relative_mjpje, test_root_relative_mpvpe, test_rank_acc = self.test_on_this_epoch()

            t_duration = time.time() - t
            info = {
                'current_epoch': self.epoch,
                'epochs': self.max_epochs,
                'test_loss': test_loss_dict['loss'],
                't_duration': t_duration
            }
            self.writer.print_info_test(info)
            for key, value in test_loss_dict.items():
                self.board.add_scalar('test/'+key, value, epoch)
            self.board.add_scalar('test/root_relative_mpjpe', test_root_relative_mjpje.item(), epoch)
            self.board.add_scalar('test/root_relative_mpvpe', test_root_relative_mpvpe.item(), epoch)
            self.board.add_scalar('test/test_rank_acc', test_rank_acc, epoch)

        # if self.eval_loader is not None:
        #     self.evaluation()

    def board_img(self, phase, n_iter, img, **kwargs):
        # print(rendered_mask.shape, rendered_mask.max(), rendered_mask.min())
        self.board.add_image(phase + '/img', tensor2array(img), n_iter)
        if kwargs.get('mask_pred') is not None:
            self.board.add_image(phase + '/mask_gt', tensor2array(kwargs['mask_gt'][0]), n_iter)
            self.board.add_image(phase + '/mask_pred', tensor2array(kwargs['mask_pred'][0]), n_iter)
        if kwargs.get('uv_pred') is not None:
            self.board.add_image(phase + '/uv_gt', tensor2array(kwargs['uv_gt'][0].sum(dim=0).clamp(max=1)), n_iter)
            self.board.add_image(phase + '/uv_pred', tensor2array(kwargs['uv_pred'][0].sum(dim=0).clamp(max=1)), n_iter)
        if kwargs.get('uv_prior') is not None:
            self.board.add_image(phase + '/uv_prior', tensor2array(kwargs['uv_prior'][0].sum(dim=0).clamp(max=1)), n_iter)

    def board_scalar(self, phase, n_iter, lr=None, **kwargs):
        split = '_' if 'train' in phase else '/'
        for key, val in kwargs.items():
            if 'loss' in key:
                self.board.add_scalar(phase + split + key, val.item(), n_iter)
        if lr:
            self.board.add_scalar('lr', lr, n_iter)

    def phrase_data(self, data):
        for key, val in data.items():
            if key == 'meta':
                continue
            if isinstance(val, list):
                data[key] = [d.to(self.device) for d in data[key]]
            else:
                data[key] = data[key].to(self.device)
        return data

    def train_a_epoch(self):
        self.model.train()
        total_loss = 0
        loss_dict = defaultdict(float)
        root_relative_mjpje = 0
        rank_acc = 0
        bar = Bar(colored("TRAIN", color='blue'), max=len(self.train_loader))
        for step, data in enumerate(self.train_loader):
            # if step > 2:
            #     break

            t = time.time()
            data = self.phrase_data(data)

            self.optimizer.zero_grad()
            if self.feed_both:
                out = self.model(data['img'], data['mesh_template'], data['shape_params'])
            elif self.feed_template:
                out = self.model(data['img'], data['mesh_template'])
            elif self.feed_shape_vector:
                out = self.model(data['img'], data['shape_params'])
            else:
                out = self.model(data['img'])

            if 'conf_pred' in out:
                # for baseline model MANO_Based_Model_Iterative_Pose_Without_GT_Shape_With_Conf
                loss = self.loss(pred=out['mesh_pred'], gt=data.get('mesh_gt'), uv_pred=out.get('uv_pred'), uv_gt=data.get('uv_gt'),
                                    face=self.face, uv_prior=out.get('uv_prior'), uv_prior2=out.get('uv_prior2'),
                                    mano_pose_pred=out.get('mano_pose_pred'), mano_pose_gt=data['pose_params'], 
                                    mano_shape_pred=out.get('mano_shape_pred'), mano_shape_gt=data['shape_params'], conf_pred=out.get('conf_pred'))

            elif 'mano_pose_pred' in out and 'mano_shape_pred' in out:
                # for baseline model MANO_Based_Model_Iterative_Pose_Without_GT_Shape and Boukhayma's model
                loss = self.loss(pred=out['mesh_pred'], gt=data.get('mesh_gt'), uv_pred=out.get('uv_pred'), uv_gt=data.get('uv_gt'),
                                    face=self.face, uv_prior=out.get('uv_prior'), uv_prior2=out.get('uv_prior2'),
                                    mano_pose_pred=out.get('mano_pose_pred'), mano_pose_gt=data['pose_params'],   
                                    mano_shape_pred=out.get('mano_shape_pred'), mano_shape_gt=data['shape_params'],
                                    keypoints_pred=out.get('keypoints_pred'), xyz_gt=data['xyz_gt_provided'])
            elif 'mano_pose_pred' in out:
                # for our model MANO_Based_Model_Iterative_Pose
                loss = self.loss(pred=out['mesh_pred'], gt=data.get('mesh_gt'), uv_pred=out.get('uv_pred'), uv_gt=data.get('uv_gt'),
                                    face=self.face, uv_prior=out.get('uv_prior'), uv_prior2=out.get('uv_prior2'),
                                    mano_pose_pred=out.get('mano_pose_pred'), mano_pose_gt=data['pose_params'], 
                                    keypoints_pred=out.get('keypoints_pred'), xyz_gt=data['xyz_gt_provided'])
            else:
                # for CMR
                loss = self.loss(pred=out['mesh_pred'], gt=data.get('mesh_gt'), uv_pred=out.get('uv_pred'), uv_gt=data.get('uv_gt'),
                                    face=self.face, uv_prior=out.get('uv_prior'), uv_prior2=out.get('uv_prior2'))

            loss['loss'].backward()

            for key, value in loss.items():
                loss_dict[key] += value.item()

            # total_loss += loss['loss'].item()
            self.optimizer.step()
            step_duration = time.time() - t
            self.total_step += 1
            self.board_scalar('train', self.total_step, self.optimizer.param_groups[0]['lr'], **loss)
            bar.suffix = (
                '({epoch}/{max_epoch}:{batch}/{size}) '
                'time: {time:.3f} | '
                'loss: {loss:.4f} | '
                'l1_loss: {l1_loss:.4f} | '
                'lr: {lr:.6f} | '
            ).format(epoch=self.epoch, max_epoch=self.max_epochs, batch=step, size=len(self.train_loader),
                     loss=loss['loss'], l1_loss=loss['l1_loss'], time=step_duration,
                     lr=self.optimizer.param_groups[0]['lr'])
            bar.next()
            if self.total_step % 100 == 0:
                info = {
                    'train_loss': loss['loss'],
                    'epoch': self.epoch,
                    'total_step': self.total_step,
                    'step_duration': step_duration,
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                self.writer.print_step(info)
            
            #get 3d keypoints from mesh
            out_mesh = out['mesh_pred'][0].detach() * self.std * 1000
            xyz_out = torch.matmul(self.j_regressor.unsqueeze(0), out_mesh) 
            xyz_out = xyz_out - xyz_out[:, :1, :]
            xyz_gt = data.get('xyz_gt') * self.std * 1000
            if len(xyz_gt.shape)==4:
                batch_size, group_size, n_keypoints = xyz_gt.shape[0], xyz_gt.shape[1], xyz_gt.shape[2]
                xyz_gt = xyz_gt.view(-1, n_keypoints, 3)
            xyz_gt = xyz_gt - xyz_gt[:, :1, :]
            error = xyz_out - xyz_gt
            root_relative_mjpje += torch.mean(torch.sum(error**2, dim=-1) **0.5)

            # get ranking accuracy
            if 'conf_pred' in out:
                rank_acc += self.calculate_rank_acc(conf_pred=out.get('conf_pred'), mano_shape_pred=out.get('mano_shape_pred'), mano_shape_gt=data['shape_params'])
                # print(' rank acc here is ', rank_acc)

        bar.finish()
        if len(data['img'].shape) == 5:
            self.board_img('train', self.epoch, data['img'][0][0], uv_gt=data.get('uv_gt')[0], uv_pred=out.get('uv_pred'), uv_prior=out.get('uv_prior'))
        else:
            self.board_img('train', self.epoch, data['img'][0], uv_gt=data.get('uv_gt'), uv_pred=out.get('uv_pred'), uv_prior=out.get('uv_prior'))

        for key, value in loss_dict.items():
            loss_dict[key] = value/len(self.train_loader)

        return loss_dict, root_relative_mjpje/len(self.train_loader), rank_acc/len(self.train_loader)

    def evaluate_on_this_epoch(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            loss_dict = defaultdict(float)
            root_relative_mjpje = 0
            root_relative_mpvpe = 0
            rank_acc = 0
            bar = Bar(colored("EVAL", color='green'), max=len(self.eval_loader))
            for step, data in enumerate(self.eval_loader):
                
                # print(rank_acc)
                # if step > 2:
                #     break

                t = time.time()
                data = self.phrase_data(data)
                if self.feed_both:
                    out = self.model(data['img'], data['mesh_template'], data['shape_params'])
                elif self.feed_template:
                    out = self.model(data['img'], data['mesh_template'])
                elif self.feed_shape_vector:
                    out = self.model(data['img'], data['shape_params'])
                else:
                    out = self.model(data['img'])

                if 'conf_pred' in out:
                    # for baseline model MANO_Based_Model_Iterative_Pose_Without_GT_Shape_With_Conf
                    loss = self.loss(pred=out['mesh_pred'], gt=data.get('mesh_gt'), uv_pred=out.get('uv_pred'), uv_gt=data.get('uv_gt'),
                                        face=self.face, uv_prior=out.get('uv_prior'), uv_prior2=out.get('uv_prior2'),
                                        mano_pose_pred=out.get('mano_pose_pred'), mano_pose_gt=data['pose_params'], 
                                        mano_shape_pred=out.get('mano_shape_pred'), mano_shape_gt=data['shape_params'], conf_pred=out.get('conf_pred'))

                elif 'mano_pose_pred' in out and 'mano_shape_pred' in out:
                    # for baseline model MANO_Based_Model_Iterative_Pose_Without_GT_Shape and Boukhayma's model
                    loss = self.loss(pred=out['mesh_pred'], gt=data.get('mesh_gt'), uv_pred=out.get('uv_pred'), uv_gt=data.get('uv_gt'),
                                        face=self.face, uv_prior=out.get('uv_prior'), uv_prior2=out.get('uv_prior2'),
                                        mano_pose_pred=out.get('mano_pose_pred'), mano_pose_gt=data['pose_params'],   
                                        mano_shape_pred=out.get('mano_shape_pred'), mano_shape_gt=data['shape_params'],
                                        keypoints_pred=out.get('keypoints_pred'), xyz_gt=data['xyz_gt_provided'])
                elif 'mano_pose_pred' in out:
                    # for our model MANO_Based_Model_Iterative_Pose
                    loss = self.loss(pred=out['mesh_pred'], gt=data.get('mesh_gt'), uv_pred=out.get('uv_pred'), uv_gt=data.get('uv_gt'),
                                        face=self.face, uv_prior=out.get('uv_prior'), uv_prior2=out.get('uv_prior2'),
                                        mano_pose_pred=out.get('mano_pose_pred'), mano_pose_gt=data['pose_params'], 
                                        keypoints_pred=out.get('keypoints_pred'), xyz_gt=data['xyz_gt_provided'])
                else:
                    # for CMR
                    loss = self.loss(pred=out['mesh_pred'], gt=data.get('mesh_gt'), uv_pred=out.get('uv_pred'), uv_gt=data.get('uv_gt'),
                                        face=self.face, uv_prior=out.get('uv_prior'), uv_prior2=out.get('uv_prior2'))

                for key, value in loss.items():
                    loss_dict[key] += value.item()
                # total_loss += loss['loss'].item()
                step_duration = time.time() - t
                self.total_step += 1
                # self.board_scalar('eval', self.total_step, self.optimizer.param_groups[0]['lr'], **loss)
                bar.suffix = (
                    '({epoch}/{max_epoch}:{batch}/{size}) '
                    'time: {time:.3f} | '
                    'loss: {loss:.4f} | '
                    'l1_loss: {l1_loss:.4f} | '
                    'lr: {lr:.6f} | '
                ).format(epoch=self.epoch, max_epoch=self.max_epochs, batch=step, size=len(self.eval_loader),
                        loss=loss['loss'], l1_loss=loss['l1_loss'], time=step_duration,
                        lr=self.optimizer.param_groups[0]['lr'])
                bar.next()
                if self.total_step % 100 == 0:
                    info = {
                        'eval_loss': loss['loss'],
                        'epoch': self.epoch,
                        'total_step': self.total_step,
                        'step_duration': step_duration,
                        'lr': self.optimizer.param_groups[0]['lr']
                    }
                    self.writer.print_step_eval(info)

                #get 3d keypoints from mesh
                xyz_out = torch.matmul(self.j_regressor.unsqueeze(0), out['mesh_pred'][0]) * self.std * 1000
                xyz_out = xyz_out - xyz_out[:, :1, :]
                xyz_gt = data.get('xyz_gt') * self.std * 1000
                xyz_gt = xyz_gt - xyz_gt[:, :1, :]
                error = xyz_out - xyz_gt
                root_relative_mjpje += torch.mean(torch.sum(error**2, dim=-1) **0.5)

                # calculate mesh error
                mesh_pred = out['mesh_pred'][0] * self.std * 1000
                mesh_gt = data.get('mesh_gt')[0] * self.std * 1000
                mesh_pred_center = torch.matmul(self.j_regressor.unsqueeze(0), mesh_pred)[:,0:1,...]
                mesh_gt_center = torch.matmul(self.j_regressor.unsqueeze(0), mesh_gt)[:,0:1,...]
                mesh_pred_aligned = mesh_pred - mesh_pred_center
                mesh_gt_aligned = mesh_gt - mesh_gt_center
                root_relative_mpvpe += torch.mean(torch.sum((mesh_pred_aligned - mesh_gt_aligned)**2, dim=-1) **0.5) 
                
                # get ranking accuracy
                if 'conf_pred' in out:
                    rank_acc += self.calculate_rank_acc(conf_pred=out.get('conf_pred'), mano_shape_pred=out.get('mano_shape_pred'), mano_shape_gt=data['shape_params'])


            bar.finish()
            self.board_img('eval', self.epoch, data['img'][0], uv_gt=data.get('uv_gt'), uv_pred=out.get('uv_pred'), uv_prior=out.get('uv_prior'))

        for key, value in loss_dict.items():
            loss_dict[key] = value/len(self.eval_loader)

        return loss_dict, root_relative_mjpje/len(self.eval_loader), root_relative_mpvpe/len(self.eval_loader), rank_acc/len(self.eval_loader)


    def test_on_this_epoch(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            loss_dict = defaultdict(float)
            root_relative_mjpje = 0
            root_relative_mpvpe = 0
            rank_acc = 0
            bar = Bar(colored("TEST", color='green'), max=len(self.test_loader))
            for step, data in enumerate(self.test_loader):
                # if step > 2:
                #     break
                t = time.time()
                data = self.phrase_data(data)
                if self.feed_both:
                    out = self.model(data['img'], data['mesh_template'], data['shape_params'])
                elif self.feed_template:
                    out = self.model(data['img'], data['mesh_template'])
                elif self.feed_shape_vector:
                    out = self.model(data['img'], data['shape_params'])
                else:
                    out = self.model(data['img'])
                
                if 'conf_pred' in out:
                    # for baseline model MANO_Based_Model_Iterative_Pose_Without_GT_Shape_With_Conf
                    loss = self.loss(pred=out['mesh_pred'], gt=data.get('mesh_gt'), uv_pred=out.get('uv_pred'), uv_gt=data.get('uv_gt'),
                                        face=self.face, uv_prior=out.get('uv_prior'), uv_prior2=out.get('uv_prior2'),
                                        mano_pose_pred=out.get('mano_pose_pred'), mano_pose_gt=data['pose_params'], 
                                        mano_shape_pred=out.get('mano_shape_pred'), mano_shape_gt=data['shape_params'], conf_pred=out.get('conf_pred'))

                elif 'mano_pose_pred' in out and 'mano_shape_pred' in out:
                    # for baseline model MANO_Based_Model_Iterative_Pose_Without_GT_Shape and Boukhayma's model
                    loss = self.loss(pred=out['mesh_pred'], gt=data.get('mesh_gt'), uv_pred=out.get('uv_pred'), uv_gt=data.get('uv_gt'),
                                        face=self.face, uv_prior=out.get('uv_prior'), uv_prior2=out.get('uv_prior2'),
                                        mano_pose_pred=out.get('mano_pose_pred'), mano_pose_gt=data['pose_params'],   
                                        mano_shape_pred=out.get('mano_shape_pred'), mano_shape_gt=data['shape_params'],
                                        keypoints_pred=out.get('keypoints_pred'), xyz_gt=data['xyz_gt_provided'])
                elif 'mano_pose_pred' in out:
                    # for our model MANO_Based_Model_Iterative_Pose
                    loss = self.loss(pred=out['mesh_pred'], gt=data.get('mesh_gt'), uv_pred=out.get('uv_pred'), uv_gt=data.get('uv_gt'),
                                        face=self.face, uv_prior=out.get('uv_prior'), uv_prior2=out.get('uv_prior2'),
                                        mano_pose_pred=out.get('mano_pose_pred'), mano_pose_gt=data['pose_params'], 
                                        keypoints_pred=out.get('keypoints_pred'), xyz_gt=data['xyz_gt_provided'])
                else:
                    # for CMR
                    loss = self.loss(pred=out['mesh_pred'], gt=data.get('mesh_gt'), uv_pred=out.get('uv_pred'), uv_gt=data.get('uv_gt'),
                                        face=self.face, uv_prior=out.get('uv_prior'), uv_prior2=out.get('uv_prior2'))

                for key, value in loss.items():
                    loss_dict[key] += value.item()
                # total_loss += loss['loss'].item()
                step_duration = time.time() - t
                self.total_step += 1
                # self.board_scalar('eval', self.total_step, self.optimizer.param_groups[0]['lr'], **loss)
                bar.suffix = (
                    '({epoch}/{max_epoch}:{batch}/{size}) '
                    'time: {time:.3f} | '
                    'loss: {loss:.4f} | '
                    'l1_loss: {l1_loss:.4f} | '
                    'lr: {lr:.6f} | '
                ).format(epoch=self.epoch, max_epoch=self.max_epochs, batch=step, size=len(self.test_loader),
                        loss=loss['loss'], l1_loss=loss['l1_loss'], time=step_duration,
                        lr=self.optimizer.param_groups[0]['lr'])
                bar.next()
                if self.total_step % 100 == 0:
                    info = {
                        'test_loss': loss['loss'],
                        'epoch': self.epoch,
                        'total_step': self.total_step,
                        'step_duration': step_duration,
                        'lr': self.optimizer.param_groups[0]['lr']
                    }
                    self.writer.print_step_test(info)

                #get 3d keypoints from mesh
                xyz_out = torch.matmul(self.j_regressor.unsqueeze(0), out['mesh_pred'][0]) * self.std * 1000
                xyz_out = xyz_out - xyz_out[:, :1, :]
                xyz_gt = data.get('xyz_gt') * self.std * 1000
                xyz_gt = xyz_gt - xyz_gt[:, :1, :]
                error = xyz_out - xyz_gt
                root_relative_mjpje += torch.mean(torch.sum(error**2, dim=-1) **0.5)

                # calculate mesh error
                mesh_pred = out['mesh_pred'][0] * self.std * 1000
                mesh_gt = data.get('mesh_gt')[0] * self.std * 1000
                mesh_pred_center = torch.matmul(self.j_regressor.unsqueeze(0), mesh_pred)[:,0:1,...]
                mesh_gt_center = torch.matmul(self.j_regressor.unsqueeze(0), mesh_gt)[:,0:1,...]
                mesh_pred_aligned = mesh_pred - mesh_pred_center
                mesh_gt_aligned = mesh_gt - mesh_gt_center
                root_relative_mpvpe += torch.mean(torch.sum((mesh_pred_aligned - mesh_gt_aligned)**2, dim=-1) **0.5) 

                # get ranking accuracy
                if 'conf_pred' in out:
                    rank_acc += self.calculate_rank_acc(conf_pred=out.get('conf_pred'), mano_shape_pred=out.get('mano_shape_pred'), mano_shape_gt=data['shape_params'])


            bar.finish()
            self.board_img('test', self.epoch, data['img'][0], uv_gt=data.get('uv_gt'), uv_pred=out.get('uv_pred'), uv_prior=out.get('uv_prior'))

        for key, value in loss_dict.items():
            loss_dict[key] = value/len(self.test_loader)

        return loss_dict, root_relative_mjpje/len(self.test_loader), root_relative_mpvpe/len(self.test_loader), rank_acc/len(self.test_loader)


    def evaluation(self):
        self.model.eval()
        # print(list(self.model.module.parameters()))
        with torch.no_grad():
            total_loss = 0
            loss_dict = defaultdict(float)
            root_relative_mjpje = 0
            mpvpe = 0
            root_relative_mpvpe = 0
            bar = Bar(colored("EVAL", color='green'), max=len(self.eval_loader))
            for step, data in enumerate(self.eval_loader):
                t = time.time()
                data = self.phrase_data(data)
                if self.feed_both:
                    out = self.model(data['img'], data['mesh_template'], data['shape_params'])
                elif self.feed_template:
                    out = self.model(data['img'], data['mesh_template'])
                elif self.feed_shape_vector:
                    out = self.model(data['img'], data['shape_params'])
                else:
                    out = self.model(data['img'])
                # total_loss += loss['loss'].item()
                #get 3d keypoints from mesh
                xyz_out = torch.matmul(self.j_regressor.unsqueeze(0), out['mesh_pred'][0]) * self.std * 1000
                xyz_out = xyz_out - xyz_out[:, :1, :]
                xyz_gt = data.get('xyz_gt') * self.std * 1000
                xyz_gt = xyz_gt - xyz_gt[:, :1, :]
                error = xyz_out - xyz_gt
                root_relative_mjpje += torch.mean(torch.sum(error**2, dim=-1) **0.5)

                mesh_pred = out['mesh_pred'][0] * self.std * 1000
                mesh_gt = data.get('mesh_gt')[0] * self.std * 1000
                mpvpe += torch.mean(torch.sum((mesh_pred - mesh_gt)**2, dim=-1) **0.5)

                mesh_pred_center = torch.matmul(self.j_regressor.unsqueeze(0), mesh_pred)[:,0:1,...]
                mesh_gt_center = torch.matmul(self.j_regressor.unsqueeze(0), mesh_gt)[:,0:1,...]
                mesh_pred_aligned = mesh_pred - mesh_pred_center
                mesh_gt_aligned = mesh_gt - mesh_gt_center
                root_relative_mpvpe += torch.mean(torch.sum((mesh_pred_aligned - mesh_gt_aligned)**2, dim=-1) **0.5) 

                bar.next()
            bar.finish()
        print('root_relative_mjpje for current epoch is', root_relative_mjpje/len(self.eval_loader))
        print('mpvpe for current epoch is', mpvpe/len(self.eval_loader))
        print('root_relative_mpvpe for current epoch is', root_relative_mpvpe/len(self.eval_loader))


    # def evaluation(self):
    #     if self.eval_loader is None:
    #         raise Exception('Please set_eval_loader before evaluation')
    #     args = self.args
    #     self.model.eval()
    #     xyz_pred_list, verts_pred_list = list(), list()
    #     bar = Bar(colored("EVAL", color='green'), max=len(self.eval_loader))
    #     with torch.no_grad():
    #         for step, data in enumerate(self.eval_loader):
    #             data = self.phrase_data(data)
    #             out = self.model(data['img'])
    #             # silhouette
    #             mask_pred = out.get('mask_pred')
    #             if mask_pred is not None:
    #                 mask_pred = (mask_pred[0] > 0.3).cpu().numpy().astype(np.uint8)
    #                 mask_pred = cv2.resize(mask_pred, (data['img'].size(3), data['img'].size(2)))
    #                 try:
    #                     contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #                     contours.sort(key=cnt_area, reverse=True)
    #                     poly = contours[0].transpose(1, 0, 2).astype(np.int32)
    #                 except:
    #                     poly = None
    #             else:
    #                 mask_pred = np.zeros([data['img'].size(3), data['img'].size(2)])
    #                 poly = None
    #             # vertex
    #             pred = out['mesh_pred'][0] if isinstance(out['mesh_pred'], list) else out['mesh_pred']
    #             vertex = (pred[0].cpu() * self.std.cpu()).numpy()
    #             uv_point_pred, uv_pred_conf = map2uv(out['uv_pred'].cpu().numpy(), (data['img'].size(2), data['img'].size(3)))
    #             vertex, align_state = registration(vertex, uv_point_pred[0], self.j_regressor, data['K'][0].cpu().numpy(), args.size, uv_conf=uv_pred_conf[0], poly=poly)

    #             vertex2xyz = mano_to_mpii(np.matmul(self.j_regressor, vertex))
    #             xyz_pred_list.append(vertex2xyz)
    #             verts_pred_list.append(vertex)
    #             # if args.phase == 'eval':
    #             #     save_a_image_with_mesh_joints(inv_base_tranmsform(data['img'][0].cpu().numpy())[:, :, ::-1], mask_pred, poly, data['K'][0].cpu().numpy(), vertex, self.faces[0], uv_point_pred[0], vertex2xyz,
    #             #                               os.path.join(args.out_dir, 'eval', str(step) + '_plot.jpg'))
    #             bar.suffix = '({batch}/{size})' .format(batch=step+1, size=len(self.eval_loader))
    #             bar.next()
    #     bar.finish()
    #     # save to a json
    #     xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    #     verts_pred_list = [x.tolist() for x in verts_pred_list]
    #     with open(os.path.join(args.out_dir, args.exp_name + '.json'), 'w') as fo:
    #         json.dump([xyz_pred_list, verts_pred_list], fo)
    #     cprint('Save json file at ' + os.path.join(args.out_dir, args.exp_name + '.json'), 'green')

    def demo(self):
        args = self.args
        self.model.eval()
        image_fp = os.path.join(args.work_dir, 'images')
        image_files = [os.path.join(image_fp, i) for i in os.listdir(image_fp) if '_img.jpg' in i]
        bar = Bar(colored("DEMO", color='blue'), max=len(image_files))
        with torch.no_grad():
            for step, image_path in enumerate(image_files):
                image_name = image_path.split('/')[-1].split('_')[0]
                image = cv2.imread(image_path)[..., ::-1]
                input = torch.from_numpy(base_transform(image, size=224)).unsqueeze(0).to(self.device)
                K = np.load(image_path.replace('_img.jpg', '_K.npy'))

                out = self.model(input)
                # silhouette
                mask_pred = out.get('mask_pred')
                if mask_pred is not None:
                    mask_pred = (mask_pred[0] > 0.3).cpu().numpy().astype(np.uint8)
                    mask_pred = cv2.resize(mask_pred, (input.size(3), input.size(2)))
                    try:
                        contours, _ = cv2.findContours(mask_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contours.sort(key=cnt_area, reverse=True)
                        poly = contours[0].transpose(1, 0, 2).astype(np.int32)
                    except:
                        poly = None
                else:
                    mask_pred = np.zeros([input.size(3), input.size(2)])
                    poly = None
                # vertex
                pred = out['mesh_pred'][0] if isinstance(out['mesh_pred'], list) else out['mesh_pred']
                vertex = (pred[0].cpu() * self.std.cpu()).numpy()
                uv_point_pred, uv_pred_conf = map2uv(out['uv_pred'].cpu().numpy(), (input.size(2), input.size(3)))
                vertex, align_state = registration(vertex, uv_point_pred[0], self.j_regressor, K, args.size, uv_conf=uv_pred_conf[0], poly=poly)

                vertex2xyz = mano_to_mpii(np.matmul(self.j_regressor, vertex))
                save_a_image_with_mesh_joints(image[..., ::-1], mask_pred, poly, K, vertex, self.faces[0], uv_point_pred[0], vertex2xyz,
                                              os.path.join(args.out_dir, 'demo', image_name + '_plot.jpg'))
                save_mesh(os.path.join(args.out_dir, 'demo', image_name + '_mesh.ply'), vertex, self.faces[0])

                bar.suffix = '({batch}/{size})' .format(batch=step+1, size=len(image_files))
                bar.next()
        bar.finish()
