import datetime
from os import path
import math


import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import torch.distributed as distributed

from model.trainer import XMemTrainer


from dataset.EPIC_dataset import EPICDataset
from dataset.EPIC_usetest_to_train import EPICTestToTrainDataset
from util.logger import TensorboardLogger
from util.configuration import Configuration
from util.load_subset import load_sub_davis, load_sub_yv
from argparse import ArgumentParser
from util.exp_handler import *
import pathlib
import sys
sys.path.append('./visualize')
from visualize.visualize_eval_result_eps import visualize_eval_result
import wandb
from glob import glob
import shutil


def get_eval_pics(yaml_root, output_path, val_data_path, iterations):
    with open(yaml_root, 'r') as f:
        info = yaml.safe_load(f)
    selected_pics = {} 
    for key, value in info.items():
        partition = key.split('_')[0]
        video_id = '_'.join(key.split('_')[:2])
        anno_path = f'{val_data_path}/{partition}/anno_masks/{video_id}/{key}'
        anno_pics = [pic.split('/')[-1] for pic in sorted(glob(f'{anno_path}/*.png'))[1:]] 
        try:
            selected_pic = random.choice(anno_pics)
        except:
            print(f'video {anno_path} has no mask:{anno_pics}')
            raise NotImplementedError
        pred_masks = []
        for it in iterations:
            eval_it_path = f'{output_path}/eval_{it}'
            pred_mask_path = f'{eval_it_path}/{partition}/{video_id}/{key}/{selected_pic}'
            pred_masks.append(pred_mask_path)
        rgb_path = [f'{val_data_path}/{partition}/rgb_frames/{video_id}/{key}/{selected_pic.replace("png", "jpg")}']*len(pred_masks)
        gt_path = [f'{val_data_path}/{partition}/anno_masks/{video_id}/{key}/{selected_pic}']*len(pred_masks)
        selected_pics.update({key: {'rgb_path': rgb_path, 'gt_path': gt_path, 'pred_path': pred_masks}})
    return selected_pics

def get_EPIC_parser():
    parser = ArgumentParser()

    
    parser.add_argument('--benchmark', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_true')

    
    parser.add_argument('--epic_root', help='EPIC data root', default='./EPIC_train') 
    parser.add_argument('--val_data_root', help='EPIC val data root', default='./val_data') 
    parser.add_argument('--yaml_root', help='yaml root', default='./EPIC_train/EPIC100_state_positive_train.yaml')
    parser.add_argument('--val_yaml_root', help='yaml root', default='./val_data/EPIC100_state_positive_val.yaml')
    parser.add_argument('--num_workers', help='Total number of dataloader workers across all GPUs processes', type=int, default=16)

    parser.add_argument('--key_dim', default=64, type=int)
    parser.add_argument('--value_dim', default=512, type=int)
    parser.add_argument('--hidden_dim', default=64, help='Set to =0 to disable', type=int)

    parser.add_argument('--deep_update_prob', default=0.2, type=float)

    """
    Stage-specific learning parameters
    Batch sizes are effective -- you don't have to scale them when you scale the number processes
    """
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--iterations', default=10000, type=int)
    parser.add_argument('--finetune', default=0, type=int)
    parser.add_argument('--steps', nargs="*", default=[1000,], type=int)
    parser.add_argument('--lr', help='Initial learning rate', default=1e-5, type=float)
    parser.add_argument('--num_ref_frames', default=3, type=int)
    parser.add_argument('--num_frames', default=8, type=int)
    parser.add_argument('--start_warm', default=500, type=int)
    parser.add_argument('--end_warm', default=3000, type=int)

    parser.add_argument('--gamma', help='LR := LR*gamma at every decay step', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=0.05, type=float)

    
    parser.add_argument('--load_network', help='Path to pretrained network weight only')
    parser.add_argument('--load_checkpoint', help='Path to the checkpoint file, including network, optimizer and such')

    
    parser.add_argument('--log_text_interval', default=100, type=int)
    parser.add_argument('--log_image_interval', default=2500, type=int)
    parser.add_argument('--save_network_interval', default=500, type=int)
    parser.add_argument('--save_checkpoint_interval', default=15000, type=int)

    parser.add_argument('--debug', help='Debug mode which logs information more often', action='store_true')

    
    parser.add_argument('--local_rank', default=0, type=int, help='Local rank of this process')
    parser.add_argument('--en_wandb', action='store_true')
    
    
    parser.add_argument('--use_dice_align', action='store_true')
    parser.add_argument('--cos_lr', action='store_true')
    parser.add_argument('--only_eval', action='store_true')
    parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
    parser.add_argument('--use_text', action='store_true')
    
    parser.add_argument('--moving_average_decay', default=0.99, type=float)
    parser.add_argument('--use_teacher_model', action='store_true')
    
    parser.add_argument('--teacher_warmup', default=100, type=int)
    
    parser.add_argument('--ts_all_align_loss', action='store_true')
    parser.add_argument('--teacher_loss_weight', default=0.1, type=float)
    parser.add_argument('--use_handmsk', default=0, type=int, choices=[0,1])
    parser.add_argument('--use_flow', default=0, type=int, choices=[0,1])
    parser.add_argument('--freeze', default=1, type=int, choices=[0,1])
    
    parser.add_argument('--openword_test', default=0, type=int, choices=[0,1])
    
    parser.add_argument('--fuser_type', default='cbam', type=str, choices=['cbam','cross_attention'])
    
    parser.add_argument('--use_randn_walk_loss', default=0, type=int, choices=[0,1])
    parser.add_argument('--randn_walk_pooling_stride', default=2, type=int, choices=[2,4])
    parser.add_argument('--randn_walk_head', default=1, type=int, choices=[0,1])
    parser.add_argument('--randn_walk_loss_rate', default=1.0, type=float) 
    parser.add_argument('--randn_walk_downsample', default='none', type=str, choices=['none', 'conv', 'pooling'])
    parser.add_argument('--randn_walk_droprate', default=0.0, type=float)
    parser.add_argument('--randn_walk_temperature', default=0.07, type=float)
    parser.add_argument('--seed', default=14159265, type=int)
    
    args = parser.parse_args()
    return {**vars(args), **{'amp': not args.no_amp}, **{'use_flow': args.use_flow}}

"""
Initial setup
"""

distributed.init_process_group(backend="nccl")
print(f'CUDA Device count: {torch.cuda.device_count()}')



config = get_EPIC_parser()


torch.backends.cudnn.benchmark = True

local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

print(f'I am rank {local_rank} in this world of size {world_size}!')


torch.manual_seed(config['seed'])
np.random.seed(config['seed'])
random.seed(config['seed'])

config['single_object'] = False

config['num_gpus'] = world_size
if config['batch_size']//config['num_gpus']*config['num_gpus'] != config['batch_size']:
    raise ValueError('Batch size must be divisible by the number of GPUs.')

config['batch_size'] //= config['num_gpus']
config['num_workers'] //= config['num_gpus']
print(f'We are assuming {config["num_gpus"]} GPUs.')

print(f'We are now starting stage EPIC')

if config['debug']:
    config['batch_size'] = 2
    config['num_frames'] = 3
    config['iterations'] = 5
    config['finetune'] = 0
    config['log_text_interval'] = config['log_image_interval'] = 1
    config['save_network_interval'] = config['save_checkpoint_interval'] = 2
    config['log_image_interval'] = 1

"""
Model related
"""
if local_rank == 0:    
    
    exp = ExpHandler(config=config, en_wandb=config['en_wandb'], resume=config['resume'])
    
    
    
    model = XMemTrainer(config, logger=exp, 
                    save_path=exp._save_dir, 
                    local_rank=local_rank, world_size=world_size).train()
else:
    
    exp = None
    model = XMemTrainer(config, local_rank=local_rank, world_size=world_size).train()


if config['load_checkpoint'] is not None:
    total_iter = model.load_checkpoint(config['load_checkpoint'])
    config['load_checkpoint'] = None
    print('Previously trained model loaded!')
else:
    total_iter = 0

if config['load_network'] is not None:
    print('I am loading network from a disk, as listed in configuration')
    model.load_network(config['load_network'])
    config['load_network'] = None

    
"""
Dataloader related
"""

def worker_init_fn(worker_id): 
    worker_seed = torch.initial_seed()%(2**31) + worker_id + local_rank*100
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def construct_loader(dataset):
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=local_rank, shuffle=True)
    train_loader = DataLoader(dataset, config['batch_size'], sampler=train_sampler, num_workers=config['num_workers'],
                            worker_init_fn=worker_init_fn, drop_last=True)
    return train_sampler, train_loader


def renew_epic_loader(max_skip, finetune=False):
    
    train_dataset = EPICDataset(config['epic_root'], config['yaml_root'], max_skip, openword_test=config['openword_test'], 
                            num_frames=config['num_frames'], finetune=finetune)
    
    if config['openword_test']:
        print('only use seen verb seen noun')
    print(f'EPIC dataset size: {len(train_dataset)}')
    print(f'Renewed with max_skip = {max_skip}')

    return construct_loader(train_dataset)

"""
Dataset related
"""

"""
These define the training schedule of the distance between frames
We will switch to max_skip_values[i] once we pass the percentage specified by increase_skip_fraction[i]
Not effective for stage 0 training
The initial value is not listed here but in renew_vos_loader(X)
"""


max_skip_values = [60, 90, 40, 40]


increase_skip_fraction = [0.1, 0.3, 0.8, 100]

train_sampler, train_loader = renew_epic_loader(max_skip_values[0])
renew_loader = renew_epic_loader


"""
Determine max epoch
"""
total_epoch = math.ceil(config['iterations']/len(train_loader))
current_epoch = total_iter // len(train_loader)
print(f'We approximately use {total_epoch} epochs.')

change_skip_iter = [round(config['iterations']*f) for f in increase_skip_fraction]

print(f'The skip value will change approximately at the following iterations: {change_skip_iter[:-1]}')

"""
Starts training
"""
finetuning = False

np.random.seed(np.random.randint(2**30-1) + local_rank*100)
if not config["only_eval"]:
    try:
        while total_iter < config['iterations'] + config['finetune']:
            
            
            train_sampler.set_epoch(current_epoch)
            current_epoch += 1
            print(f'Current epoch: {current_epoch}')

            
            model.train()
            for data in train_loader:
                
                
                
                if total_iter >= change_skip_iter[0]:
                    while total_iter >= change_skip_iter[0]:
                        cur_skip = max_skip_values[0]
                        max_skip_values = max_skip_values[1:]
                        change_skip_iter = change_skip_iter[1:]
                    print(f'Changing skip to {cur_skip}')
                    train_sampler, train_loader = renew_loader(cur_skip)
                    break

                
                
                if config['finetune'] > 0 and not finetuning and total_iter >= config['iterations']:
                    train_sampler, train_loader = renew_loader(cur_skip, finetune=True)
                    finetuning = True
                    model.save_network_interval = 1000
                    break

                model.do_pass(data, total_iter)
                total_iter += 1

                if total_iter >= config['iterations'] + config['finetune']:
                    break
    finally:
        
        if model.logger is not None:
            model.save_network(total_iter)
            model.save_checkpoint(total_iter)

del model
distributed.destroy_process_group()
