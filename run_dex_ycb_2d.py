import os
import torch
from utils.vis import cnt_area
import numpy as np
import cv2
from utils.vis import registration, map2uv, base_transform, tensor2array
from utils.draw3d import save_a_image_with_mesh_joints
from utils.read import save_mesh
import json
from datasets.FreiHAND.kinematics import mano_to_mpii
from utils.progress.bar import Bar
from termcolor import colored, cprint
import pickle
import time
from collections import defaultdict


class Runner(object):
    def __init__(self, args, model, faces, device, feed_template=False, feed_shape_vector=False, feed_both=False):
        assert (feed_template and feed_shape_vector) == False
        super(Runner, self).__init__()
        self.args = args
        self.model = model
        self.faces = faces
        self.device = device
        with open(os.path.join(self.args.work_dir, 'template', 'humbi_j_regressor.npy'), 'rb') as f:
            self.j_regressor = torch.from_numpy(np.load(f)).float().to(self.device)
        self.std = torch.tensor(0.20).to(self.device)
        self.face = torch.from_numpy(self.faces[0].astype(np.int64)).to(self.device)
        self.best_root_rela_mpjpe = float('inf')
        self.best_error_2d = float('inf')
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

    def uv_map_to_coord(self, uv_map):
        bs, joints, h, w = uv_map.shape[0], uv_map.shape[1], uv_map.shape[2], uv_map.shape[3]
        idx = torch.argmax(uv_map.view(bs, joints, -1), dim=-1)
        y = idx // w
        x = idx % w
        return torch.stack((x,y), -1)
    
    def get_2d_error(self, uv_map, pred_uv_map):
        gt_2d = self.uv_map_to_coord(uv_map)
        # print('gt_2d.shape', gt_2d.shape)
        pred_2d = self.uv_map_to_coord(pred_uv_map)
        # print('pred_2d.shape', pred_2d.shape)
        return torch.mean(torch.sum((gt_2d-pred_2d)**2, -1)**0.5)

    def train(self):
        for epoch in range(self.start_epoch, self.max_epochs + 1):
            self.epoch = epoch
            t = time.time()
            train_loss_dict,  train_error_2d= self.train_a_epoch()
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
            self.board.add_scalar('train/error_2d', train_error_2d.item(), epoch)
            self.writer.save_checkpoint(self.model.module, self.optimizer, self.scheduler, self.epoch, last=True)

            t = time.time()
            print('evaluating......')
            eval_loss_dict, eval_error_2d = self.evaluate_on_this_epoch()

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
            self.board.add_scalar('eval/error_2d', eval_error_2d.item(), epoch)
            if self.best_error_2d > eval_error_2d.item():
                self.best_error_2d = eval_error_2d.item()
                self.writer.save_checkpoint(self.model.module, self.optimizer, self.scheduler, self.epoch, best=True)

            t = time.time()
            print('testing......')
            test_loss_dict, test_error_2d = self.test_on_this_epoch()

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
            self.board.add_scalar('test/error_2d', test_error_2d.item(), epoch)

            self.board.add_scalar('test_and_val_combined/error_2d', (eval_error_2d.item() + 3 * test_error_2d.item())/4, epoch)

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
        error_2d = 0
        loss_dict = defaultdict(float)
        root_relative_mjpje = 0
        bar = Bar(colored("TRAIN", color='blue'), max=len(self.train_loader))
        for step, data in enumerate(self.train_loader):
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

            loss = self.loss(uv_pred=out.get('uv_pred'), uv_gt=data.get('uv_gt'),
                            uv_prior=out.get('uv_prior'), uv_prior2=out.get('uv_prior2'))

            loss['loss'].backward()

            for key, value in loss.items():
                loss_dict[key] += value.item()

            # total_loss += loss['loss'].item()
            self.optimizer.step()
            step_duration = time.time() - t
            self.total_step += 1
            self.board_scalar('train', self.total_step, self.optimizer.param_groups[0]['lr'], **loss)

            cur_error = self.get_2d_error(data['uv_gt'], out['uv_pred'])
            # print('--------------')
            # print(data['uv_gt'].shape, out['uv_pred'].shape)
            # stop
            error_2d += cur_error

            # if step == 500:
            #     torch.save(data['uv_gt'].cpu(), 'uv_gt.pt')
            #     torch.save(out['uv_pred'].cpu(), 'uv_pred.pt')

            bar.suffix = (
                '({epoch}/{max_epoch}:{batch}/{size}) '
                'time: {time:.3f} | '
                'loss: {loss:.4f} | '
                'cur_error: {cur_error:.4f} | '
                'lr: {lr:.6f} | '
            ).format(epoch=self.epoch, max_epoch=self.max_epochs, batch=step, size=len(self.train_loader),
                     loss=loss['loss'], cur_error=cur_error.item(), time=step_duration,
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
            


        bar.finish()
        if len(data['img'].shape) == 5:
            self.board_img('train', self.epoch, data['img'][0][0], uv_gt=data.get('uv_gt')[0], uv_pred=out.get('uv_pred'), uv_prior=out.get('uv_prior'))
        else:
            self.board_img('train', self.epoch, data['img'][0], uv_gt=data.get('uv_gt'), uv_pred=out.get('uv_pred'), uv_prior=out.get('uv_prior'))

        for key, value in loss_dict.items():
            loss_dict[key] = value/len(self.train_loader)

        return loss_dict, error_2d/len(self.train_loader)

    def evaluate_on_this_epoch(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            loss_dict = defaultdict(float)
            error_2d = 0
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
                loss = self.loss(uv_pred=out.get('uv_pred'), uv_gt=data.get('uv_gt'),
                            uv_prior=out.get('uv_prior'), uv_prior2=out.get('uv_prior2'))

                for key, value in loss.items():
                    loss_dict[key] += value.item()
                # total_loss += loss['loss'].item()
                step_duration = time.time() - t
                self.total_step += 1

                cur_error = self.get_2d_error(data['uv_gt'], out['uv_pred'])
                error_2d += cur_error
                # self.board_scalar('eval', self.total_step, self.optimizer.param_groups[0]['lr'], **loss)
                bar.suffix = (
                    '({epoch}/{max_epoch}:{batch}/{size}) '
                    'time: {time:.3f} | '
                    'loss: {loss:.4f} | '
                    'cur_error: {cur_error:.4f} | '
                    'lr: {lr:.6f} | '
                ).format(epoch=self.epoch, max_epoch=self.max_epochs, batch=step, size=len(self.eval_loader),
                        loss=loss['loss'], cur_error=cur_error.item(), time=step_duration,
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



            bar.finish()
            self.board_img('eval', self.epoch, data['img'][0], uv_gt=data.get('uv_gt'), uv_pred=out.get('uv_pred'), uv_prior=out.get('uv_prior'))

        for key, value in loss_dict.items():
            loss_dict[key] = value/len(self.eval_loader)

        return loss_dict, error_2d/len(self.eval_loader)


    def test_on_this_epoch(self):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0
            loss_dict = defaultdict(float)
            error_2d = 0
            bar = Bar(colored("TEST", color='green'), max=len(self.test_loader))
            for step, data in enumerate(self.test_loader):

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
                loss = self.loss(uv_pred=out.get('uv_pred'), uv_gt=data.get('uv_gt'),
                            uv_prior=out.get('uv_prior'), uv_prior2=out.get('uv_prior2'))
                for key, value in loss.items():
                    loss_dict[key] += value.item()
                # total_loss += loss['loss'].item()
                step_duration = time.time() - t
                cur_error = self.get_2d_error(data['uv_gt'], out['uv_pred'])
                error_2d += cur_error

                self.total_step += 1
                # self.board_scalar('eval', self.total_step, self.optimizer.param_groups[0]['lr'], **loss)
                bar.suffix = (
                    '({epoch}/{max_epoch}:{batch}/{size}) '
                    'time: {time:.3f} | '
                    'loss: {loss:.4f} | '
                    'cur_error: {cur_error:.4f} | '
                    'lr: {lr:.6f} | '
                ).format(epoch=self.epoch, max_epoch=self.max_epochs, batch=step, size=len(self.test_loader),
                        loss=loss['loss'], cur_error=cur_error.item(), time=step_duration,
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


            bar.finish()
            self.board_img('test', self.epoch, data['img'][0], uv_gt=data.get('uv_gt'), uv_pred=out.get('uv_pred'), uv_prior=out.get('uv_prior'))

        for key, value in loss_dict.items():
            loss_dict[key] = value/len(self.test_loader)

        return loss_dict, error_2d/len(self.test_loader)


    def evaluation(self):
        raise 'not implemented yet'
        self.model.eval()
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
