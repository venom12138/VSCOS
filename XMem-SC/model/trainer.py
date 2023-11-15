"""
trainer.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import git
import datetime

sys.path.append('/home/venom/projects/XMem/')


from copy import deepcopy
from util.log_integrator import Integrator
from util.image_saver import pool_pairs
from util.configuration import Configuration
from util.logger import TensorboardLogger
import model.resnet as resnet
from model.network import XMem
from model.losses import LossComputer
import matplotlib.pyplot as plt
import wandb
import clip
from model.modules import RandomWalkHead
class EMA():
    def __init__(self, beta, iterations):
        super().__init__()
        self.beta = beta
        self.count = 0
        self.beta_base = beta
        self.count = 0
        self.total_counts = iterations
        
    def update_average(self, old, new):
        if old is None:
            return deepcopy(new)
        return old * self.beta + (1 - self.beta) * new
    
    def step(self,):
        self.count += 1
        self.beta = 1 - (1-self.beta_base)*(np.cos(np.pi*self.count/self.total_counts) + 1)/2
        
        
def update_moving_average(ema_updater, ma_model, current_model):
    
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

class XMemTrainer:
    def __init__(self, config, logger=None, save_path=None, local_rank=0, world_size=1):
        self.config = config
        self.num_frames = config['num_frames']
        self.num_ref_frames = config['num_ref_frames']
        self.deep_update_prob = config['deep_update_prob']
        self.local_rank = local_rank
        try:
            network = XMem(config)
            self.XMem = nn.parallel.DistributedDataParallel(
                network.cuda(), 
                device_ids=[local_rank], output_device=local_rank, 
                broadcast_buffers=False, find_unused_parameters=True)
            if config['use_teacher_model']:
                self.teacher_model = nn.parallel.DistributedDataParallel(
                        deepcopy(network).cuda(), 
                        device_ids=[local_rank], output_device=local_rank, 
                        broadcast_buffers=False, find_unused_parameters=True)
        except:
            network = XMem(config)
            self.XMem = nn.parallel.DataParallel(
                network.cuda())
            if config['use_teacher_model']:
                self.teacher_model = nn.parallel.DataParallel(
                        deepcopy(network).cuda())
        
        if config['use_randn_walk_loss']:
            self.randn_walk_head = RandomWalkHead(key_dim = config['key_dim'], 
                                                downsample_mode = config['randn_walk_downsample'],
                                                use_head = config['randn_walk_head'],
                                                dropout_rate = config['randn_walk_droprate'],
                                                temperature = config['randn_walk_temperature'],
                                                pooling_stride=config['randn_walk_pooling_stride']).to(self.XMem.device)
        
        
        self.logger = logger
        self.save_path = save_path
        if logger is not None:
            self.last_time = time.time()
            self.logger.log(f'model_size:{str(sum([param.nelement() for param in self.XMem.parameters()]))}')
        
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        self.loss_computer = LossComputer(config)

        
        self.train()
        if self.config['use_text']:
            
            for param in self.XMem.module.clip_text_encoder.parameters():
                param.requires_grad = False
        
        if config['use_teacher_model']:
            self.ema_updater = EMA(config['moving_average_decay'], config['iterations']+config['finetune'])
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        else:
            self.ema_updater = None
        
        
        if self.config['freeze'] == 1:
            
            for param in self.XMem.module.key_encoder.parameters():
                param.requires_grad = False
            for param in self.XMem.module.value_encoder.parameters():
                param.requires_grad = False
            
            
            
            
            
            
            
        else:
            print('not freeze!!!')

        model_param_group = [{'params': filter(lambda p: p.requires_grad, self.XMem.parameters())}]
        if config['use_randn_walk_loss']:
            model_param_group += [{'params': filter(lambda p: p.requires_grad, self.randn_walk_head.parameters())}]
        
        self.optimizer = optim.AdamW(model_param_group, lr=config['lr'], weight_decay=config['weight_decay'])
        if config['cos_lr']:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, config['iterations']+config['finetune'])
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, config['steps'], config['gamma'])
        if config['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        
        self.log_text_interval = config['log_text_interval']
        self.log_image_interval = config['log_image_interval']
        self.save_network_interval = config['save_network_interval']
        self.save_checkpoint_interval = config['save_checkpoint_interval']
        
        
        
        

    def do_pass(self, data, it=0):
        
        if it == self.config['teacher_warmup']:
            print('teacher start')
            self.teacher_model = deepcopy(self.XMem)
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        
        
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)
    
        out = {}
        
        frames = data['rgb']
        
        forward_flows = data['forward_flow']
        backward_flows = data['backward_flow']
        text = data['text']
        text = [f"a photo of {t}" for t in text]
        hand_mask = data['hand_mask'] 
        
        first_frame_gt = data['first_last_frame_gt'][:,0].unsqueeze(1).float()
        
        last_frame_gt = data['first_last_frame_gt'][:,1].unsqueeze(1).float()
        b = frames.shape[0]
        
        
        num_filled_objects = [o.item() for o in data['info']['num_objects']]
        
        num_objects = first_frame_gt.shape[2]
        selector = data['selector'].unsqueeze(2).unsqueeze(2) 

        with torch.cuda.amp.autocast(enabled=self.config['amp']):
            
            
            
            
            
            
            
            
            
            if self.config['use_text']:
                text = clip.tokenize(text).cuda()
                
                text_feat = self.XMem.module.encode_text(text)
                if self.config['use_teacher_model']:
                    t_text_feat = self.teacher_model.module.encode_text(text)

            
            
            key, shrinkage, selection, f16, f8, f4 = self.XMem('encode_key', frames)
            
            if self.config['use_randn_walk_loss']:
                randn_walk_loss, rand_walk_loss_dict = self.randn_walk_head(key) 
                
                
            if self.config['use_teacher_model']:
                t_key, t_shrinkage, t_selection, t_f16, t_f8, t_f4 = self.teacher_model('encode_key', frames)
            
            if self.config['use_flow']:
                forward_flow_feats = self.XMem('encode_flow', forward_flows) 
                if self.config['use_teacher_model']:
                    t_forward_flow_feats = self.teacher_model('encode_flow', forward_flows)
                backward_flow_feats = self.XMem('encode_flow', backward_flows) 
                if self.config['use_teacher_model']:
                    t_backward_flow_feats = self.teacher_model('encode_flow', backward_flows)
            
            if self.config['use_handmsk']:
                
                handkey = self.XMem.module.hand_encoder(hand_mask)
                
                if self.config['use_teacher_model']:
                    t_handkey = self.teacher_model.module.hand_encoder(hand_mask)
            
            filler_one = torch.zeros(1, dtype=torch.int64)
            hidden = torch.zeros((b, num_objects, self.config['hidden_dim'], *key.shape[-2:]))
            
            if self.config['use_teacher_model']:
                t_hidden = torch.zeros((b, num_objects, self.config['hidden_dim'], *t_key.shape[-2:]))
            
            
            
            
            
            
            
            
            v16, hidden = self.XMem('encode_value', frames[:,0], f16[:,0], hidden, first_frame_gt[:,0])
            
            
            
            if self.config['use_teacher_model']:
                t_v16, t_hidden = self.teacher_model('encode_value', frames[:,0], t_f16[:,0], t_hidden, first_frame_gt[:,0])
                t_values = t_v16.unsqueeze(3)
            
            values = v16.unsqueeze(3) 
            
            
            
            for ti in range(1, self.num_frames):
                
                if ti <= self.num_ref_frames:
                    ref_values = values
                    
                    ref_keys = key[:,:,:ti]
                    ref_shrinkage = shrinkage[:,:,:ti] if shrinkage is not None else None
                    if self.config['use_teacher_model']:
                        t_ref_values = t_values
                        
                        t_ref_keys = t_key[:,:,:ti]
                        t_ref_shrinkage = t_shrinkage[:,:,:ti] if t_shrinkage is not None else None
                else:
                    
                    
                    
                    
                    
                    
                    indices = [
                        torch.cat([filler_one, torch.randperm(ti-1)[:self.num_ref_frames-1]+1])
                    for _ in range(b)]
                    ref_values = torch.stack([
                        values[bi, :, :, indices[bi]] for bi in range(b)
                    ], 0)
                    ref_keys = torch.stack([
                        key[bi, :, indices[bi]] for bi in range(b)
                    ], 0)
                    ref_shrinkage = torch.stack([
                        shrinkage[bi, :, indices[bi]] for bi in range(b)
                    ], 0) if shrinkage is not None else None

                    if self.config['use_teacher_model']:
                        t_ref_values = torch.stack([
                            t_values[bi, :, :, indices[bi]] for bi in range(b)
                        ], 0)
                        t_ref_keys = torch.stack([
                            t_key[bi, :, indices[bi]] for bi in range(b)
                        ], 0)
                        t_ref_shrinkage = torch.stack([
                            t_shrinkage[bi, :, indices[bi]] for bi in range(b)
                        ], 0) if t_shrinkage is not None else None
                        
                
                
                
                memory_readout = self.XMem('read_memory', key[:,:,ti], selection[:,:,ti] if selection is not None else None, 
                                        ref_keys, ref_shrinkage, ref_values)
                
                if self.config['use_text'] or self.config['use_flow'] or self.config['use_handmsk']:
                    memory_readout = self.XMem('fuse_value', mv=memory_readout, \
                                            flow_feat=forward_flow_feats[:,ti] if self.config['use_flow'] else None, \
                                            text_feat=text_feat if self.config['use_text'] else None, \
                                            hand_feat=handkey[:, ti] if self.config['use_handmsk'] else None, \
                                            ) 
                
                if self.config['use_teacher_model']:
                    t_memory_readout = self.teacher_model('read_memory', t_key[:,:,ti], t_selection[:,:,ti] if t_selection is not None else None, 
                                        t_ref_keys, t_ref_shrinkage, t_ref_values)
                    
                    if self.config['use_text'] or self.config['use_flow'] or self.config['use_handmsk']:
                        t_memory_readout = self.teacher_model('fuse_value', mv=t_memory_readout, \
                            flow_feat=t_forward_flow_feats[:,ti] if self.config['use_flow'] else None, \
                            text_feat=t_text_feat if self.config['use_text'] else None, \
                            hand_feat=t_handkey[:, ti] if self.config['use_handmsk'] else None, \
                            ) 
                    
                
                
                
                
                
                
                hidden, logits, masks = self.XMem('segment', (f16[:,ti], f8[:,ti], f4[:,ti]), 
                                                memory_readout, hidden, selector, (ti < (self.num_frames-1))) 

                if self.config['use_teacher_model']:
                    t_hidden, t_logits, t_masks = self.teacher_model('segment', (t_f16[:,ti], t_f8[:,ti], t_f4[:,ti]), 
                                                t_memory_readout, t_hidden, selector, (ti < (self.num_frames-1))) 
                
                if ti < (self.num_frames-1):
                    is_deep_update = np.random.rand() < self.deep_update_prob 
                    v16, hidden = self.XMem('encode_value', frames[:,ti], f16[:,ti], hidden, masks, is_deep_update=is_deep_update)
                    values = torch.cat([values, v16.unsqueeze(3)], 3) 
                    if self.config['use_teacher_model']:
                        t_v16, t_hidden = self.teacher_model('encode_value', frames[:,ti], t_f16[:,ti], t_hidden, t_masks, is_deep_update=is_deep_update)
                        t_values = torch.cat([t_values, t_v16.unsqueeze(3)], 3) 

                out[f'fmasks_{ti}'] = masks
                out[f'flogits_{ti}'] = logits
                if self.config['use_teacher_model']:
                    out[f't_fmasks_{ti}'] = t_masks
                    out[f't_flogits_{ti}'] = t_logits
            

            
            
            
            
            
            
            
            
            
            
            frames = torch.flip(frames, [1]) 
            key = torch.flip(key, [2])
            shrinkage = torch.flip(shrinkage, [2]) if shrinkage is not None else None
            selection = torch.flip(selection, [2]) if selection is not None else None
            f16 = torch.flip(f16, [1])
            f8 = torch.flip(f8, [1])
            f4 = torch.flip(f4, [1])
            if self.config['use_handmsk']:
                handkey = torch.flip(handkey, [1])
            if self.config['use_teacher_model']:
                t_key = torch.flip(t_key, [2])
                t_shrinkage = torch.flip(t_shrinkage, [2]) if t_shrinkage is not None else None
                t_selection = torch.flip(t_selection, [2]) if t_selection is not None else None
                t_f16 = torch.flip(t_f16, [1])
                t_f8 = torch.flip(t_f8, [1])
                t_f4 = torch.flip(t_f4, [1])
                if self.config['use_handmsk']:
                    t_handkey = torch.flip(t_handkey, [1])
            
            if self.config['use_flow']:
                backward_flow_feats = torch.flip(backward_flow_feats, [1])
                if self.config['use_teacher_model']:
                    t_backward_flow_feats = torch.flip(t_backward_flow_feats, [1])

            filler_one = torch.zeros(1, dtype=torch.int64)
            hidden = torch.zeros((b, num_objects, self.config['hidden_dim'], *key.shape[-2:]))
            if self.config['use_teacher_model']:
                t_hidden = torch.zeros((b, num_objects, self.config['hidden_dim'], *t_key.shape[-2:]))
            
            
            
            
            
            
            
            
            v16, hidden = self.XMem('encode_value', frames[:,0], f16[:,0], hidden, last_frame_gt[:,0])
            if self.config['use_teacher_model']:
                t_v16, t_hidden = self.teacher_model('encode_value', frames[:,0], t_f16[:,0], t_hidden, last_frame_gt[:,0])
                t_values = t_v16.unsqueeze(3) 
            
            values = v16.unsqueeze(3) 
            
            
            for ti in range(1, self.num_frames):
                
                if ti <= self.num_ref_frames:
                    ref_values = values
                    
                    ref_keys = key[:,:,:ti]
                    ref_shrinkage = shrinkage[:,:,:ti] if shrinkage is not None else None
                    if self.config['use_teacher_model']:
                        t_ref_values = t_values
                        
                        t_ref_keys = t_key[:,:,:ti]
                        t_ref_shrinkage = t_shrinkage[:,:,:ti] if t_shrinkage is not None else None
                else:
                    
                    
                    
                    
                    
                    
                    indices = [
                        torch.cat([filler_one, torch.randperm(ti-1)[:self.num_ref_frames-1]+1])
                    for _ in range(b)]
                    ref_values = torch.stack([
                        values[bi, :, :, indices[bi]] for bi in range(b)
                    ], 0)
                    ref_keys = torch.stack([
                        key[bi, :, indices[bi]] for bi in range(b)
                    ], 0)
                    ref_shrinkage = torch.stack([
                        shrinkage[bi, :, indices[bi]] for bi in range(b)
                    ], 0) if shrinkage is not None else None
                    
                    if self.config['use_teacher_model']:
                        t_ref_values = torch.stack([
                            t_values[bi, :, :, indices[bi]] for bi in range(b)
                        ], 0)
                        t_ref_keys = torch.stack([
                            t_key[bi, :, indices[bi]] for bi in range(b)
                        ], 0)
                        t_ref_shrinkage = torch.stack([
                            t_shrinkage[bi, :, indices[bi]] for bi in range(b)
                        ], 0) if t_shrinkage is not None else None

                
                memory_readout = self.XMem('read_memory', key[:,:,ti], selection[:,:,ti] if selection is not None else None, 
                                        ref_keys, ref_shrinkage, ref_values)
                if self.config['use_text'] or self.config['use_flow'] or self.config['use_handmsk']:
                    memory_readout = self.XMem('fuse_value', mv=memory_readout, \
                                            flow_feat=backward_flow_feats[:,ti] if self.config['use_flow'] else None, \
                                            text_feat=text_feat if self.config['use_text'] else None, \
                                            hand_feat=handkey[:, ti] if self.config['use_handmsk'] else None, \
                                            ) 
                
                if self.config['use_teacher_model']:
                    t_memory_readout = self.teacher_model('read_memory', t_key[:,:,ti], t_selection[:,:,ti] if t_selection is not None else None, 
                                        t_ref_keys, t_ref_shrinkage, t_ref_values)
                    if self.config['use_text'] or self.config['use_flow'] or self.config['use_handmsk']:
                        t_memory_readout = self.teacher_model('fuse_value', mv=t_memory_readout, \
                            flow_feat=t_backward_flow_feats[:,ti] if self.config['use_flow'] else None, \
                            text_feat=t_text_feat if self.config['use_text'] else None, \
                            hand_feat=t_handkey[:, ti] if self.config['use_handmsk'] else None, \
                            ) 
                
                
                
                
                
                
                
                
                hidden, logits, masks = self.XMem('segment', (f16[:,ti], f8[:,ti], f4[:,ti]), 
                                                memory_readout, hidden, selector, (ti < (self.num_frames-1))) 

                if self.config['use_teacher_model']:
                    t_hidden, t_logits, t_masks = self.teacher_model('segment', (t_f16[:,ti], t_f8[:,ti], t_f4[:,ti]), 
                                                t_memory_readout, t_hidden, selector, (ti < (self.num_frames-1))) 
                    
                
                if ti < (self.num_frames-1):
                    is_deep_update = np.random.rand() < self.deep_update_prob 
                    v16, hidden = self.XMem('encode_value', frames[:,ti], f16[:,ti], hidden, masks, is_deep_update=is_deep_update)
                    values = torch.cat([values, v16.unsqueeze(3)], 3) 
                    if self.config['use_teacher_model']:
                        t_v16, t_hidden = self.teacher_model('encode_value', frames[:,ti], t_f16[:,ti], t_hidden, t_masks, is_deep_update=is_deep_update)
                        t_values = torch.cat([t_values, t_v16.unsqueeze(3)], 3) 
                out[f'bmasks_{self.num_frames-ti-1}'] = masks 
                out[f'blogits_{self.num_frames-ti-1}'] = logits 
                if self.config['use_teacher_model']:
                    out[f't_bmasks_{self.num_frames-ti-1}'] = t_masks
                    out[f't_blogits_{self.num_frames-ti-1}'] = t_logits
                
            
            
            
            
            
            
            
            
            if self._do_log or self._is_train:
                
                
                
                losses = self.loss_computer.compute({**data, **out}, num_filled_objects, it)
                
                
                if self.config['use_randn_walk_loss']:
                    losses['total_loss'] += self.config['randn_walk_loss_rate']*randn_walk_loss.to(losses['total_loss'].device)
                    losses.update(rand_walk_loss_dict)
                
                if self._do_log:
                    self.integrator.add_dict(losses)
                    if self._is_train:
                        if it % self.log_image_interval == 0 and it != 0:
                            if self.logger is not None:
                                images = {**data, **out}
                                size = (384, 384)
                                masks = pool_pairs(images, size, num_filled_objects, use_teacher=self.config['use_teacher_model'])
                                
                                self.logger.log_image(masks)
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                
                                

            if self._is_train:
                if (it) % self.log_text_interval == 0 and it != 0:
                    
                    train_metrics = self.train_integrator.finalize()
                    if self.logger is not None:
                        self.logger.write(prefix='train', train_metrics=train_metrics, **{'lr':self.scheduler.get_last_lr()[0],
                                        'time':(time.time()-self.last_time)/self.log_text_interval, 'beta':self.ema_updater.beta if self.ema_updater is not None else 0.0})
                        all_dicts = {**train_metrics, **{'lr':self.scheduler.get_last_lr()[0],
                                            'time':(time.time()-self.last_time)/self.log_text_interval}}
                        self.last_time = time.time()
                        for k, v in all_dicts.items():
                            msg = 'It {:6d} [{:5s}] [{:13}]: {:s}'.format(it, 'TRAIN', k, '{:.9s}'.format('{:0.9f}'.format(v)))
                            if self.logger is not None:
                                self.logger.log(msg)
                    self.train_integrator.reset_except_hooks()

                if it % self.save_network_interval == 0 and it != 0:
                    if self.logger is not None:
                        self.save_network(it)

                
                
                
        
        self.optimizer.zero_grad(set_to_none=True)
        if self.config['amp']:
            self.scaler.scale(losses['total_loss']).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total_loss'].backward() 
            self.optimizer.step()
        
        self.scheduler.step()
        if it >= self.config['teacher_warmup']:
            if self.config['use_teacher_model']:
                update_moving_average(self.ema_updater, self.teacher_model, self.XMem)
                self.ema_updater.step()
        
    def save_network(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = f'{self.save_path}/network_{it}.pth'
        torch.save(self.XMem.module.state_dict(), model_path)
        torch.save(self.XMem.module.state_dict(), f'{self.save_path}/latest_network.pth')
        print(f'Network saved to {model_path}.')

    def save_checkpoint(self, it):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = f'{self.save_path}/checkpoint_{it}.pth'
        checkpoint = { 
            'it': it,
            'network': self.XMem.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}.')

    def load_checkpoint(self, path):
        
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.XMem.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Network weights, optimizer states, and scheduler states loaded.')

        return it

    def load_network_in_memory(self, src_dict):
        self.XMem.module.load_weights(src_dict)
        print('Network weight loaded from memory.')

    def load_network(self, path):
        
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        self.load_network_in_memory(src_dict)
        print(f'Network weight loaded from {path}')

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        
        self.XMem.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.XMem.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.XMem.eval()
        return self

if __name__ == '__main__':
    raw_config = Configuration()
    raw_config.parse()
    stage_config = raw_config.get_stage_parameters('0')
    config = dict(**raw_config.args, **stage_config)
    config['num_frames'] = 4
    
    
    if config['exp_id'].lower() != 'null':
        print('I will take the role of logging!')
        long_id = '%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), config['exp_id'])
    else:
        long_id = None
    logger = TensorboardLogger(config['exp_id'], long_id, '123456')
    logger.log_string('hyperpara', str(config))
    data = {
            'rgb': torch.rand(2,4,3,384,384), 
            'flow':torch.rand(2,4,2,384,384), 
            'first_last_frame_gt': torch.randint(0, 1, (2,2,5,384,384)), 
            'cls_gt': torch.randint(0,2,(2,2,1,384,384)), 
            'selector': torch.tensor([[1, 1, 1, 0, 0],[1, 1, 1, 0, 0]]), 
            'info': {'num_frames': torch.tensor([4,4]), 'num_objects': torch.tensor([3,3])},
        }
    model = XMemTrainer(config).train()
    network = resnet.resnet50(pretrained=False)
    
    
    
    model.do_pass(data, 1)