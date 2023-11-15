import os
from os import path, replace
import math
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
import sys
sys.path.append('/cluster/home2/yjw/venom/XMem')
from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed
import yaml
import matplotlib.pyplot as plt
from glob import glob

class EPICTestToTrainDataset(Dataset):
    """
    Works for EPIC training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, data_root, yaml_root, max_jump, num_frames=3, max_num_obj=3, finetune=False):
        print('We are using EPIC TestToTrainDataset !!!!!')
        self.data_root = data_root
        self.max_jump = max_jump
        self.num_frames = num_frames
        self.max_num_obj = max_num_obj
        with open(os.path.join(yaml_root), 'r') as f:
            self.data_info = yaml.safe_load(f)
        self.vids = [] 
        for key in list(self.data_info.keys()):
            PART = key.split('_')[0]
            VIDEO_ID = '_'.join(key.split('_')[:2])
            vid_gt_path = os.path.join(self.data_root, PART, 'anno_masks', VIDEO_ID, key)
            
            
            
            if len(glob(f"{vid_gt_path}/*.png")) >= 2:
                self.vids.append(key)
        assert num_frames >= 3
        
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])
        
        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune else 15, shear=0 if finetune else 10, interpolation=InterpolationMode.BILINEAR, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune else 15, shear=0 if finetune else 10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        
        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.BILINEAR)
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
        ])

        
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

    def __getitem__(self, idx):
        video_value = self.data_info[self.vids[idx]] 
        
        info = {}
        info['name'] = self.vids[idx]

        vid_im_path = path.join(self.data_root, video_value['participant_id'], 'rgb_frames', video_value['video_id'], self.vids[idx])
        
        vid_gt_path = path.join(self.data_root, video_value['participant_id'], 'anno_masks', video_value['video_id'], self.vids[idx])
        vid_flow_path = path.join(self.data_root, video_value['participant_id'], 'flow_frames', video_value['video_id'], self.vids[idx])
        gt_frames = sorted(glob(f"{vid_gt_path}/*.png"))
        start_frame = int(gt_frames[0].split('/')[-1].split('.')[0][6:])
        stop_frame = int(gt_frames[-1].split('/')[-1].split('.')[0][6:])+1
        frames = list(range(start_frame, stop_frame))

        trials = 0
        while trials < 5:
            info['frames'] = [] 

            num_frames = self.num_frames
            
            length = len(frames)
            this_max_jump = min(len(frames), self.max_jump)

            
            
            
            
            frames_idx = [0, len(frames)-1, np.random.randint(1,length-1)] 
            acceptable_set = set(range(max(1, frames_idx[-1]-this_max_jump), min(length-1, frames_idx[-1]+this_max_jump+1))).difference(set(frames_idx))
            while(len(frames_idx) < num_frames):
                idx = np.random.choice(list(acceptable_set))
                frames_idx.append(idx)
                new_set = set(range(max(1, frames_idx[-1]-this_max_jump), min(length-1, frames_idx[-1]+this_max_jump+1)))
                acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))

            frames_idx = sorted(frames_idx)
            
            
            
            
            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            flows = []
            target_objects = []
            for f_idx in frames_idx:
                jpg_name = 'frame_' + str(frames[f_idx]).zfill(10)+ '.jpg'
                png_name = 'frame_' + str(frames[f_idx]).zfill(10)+ '.png'
                if len(video_value['video_id'].split('_')[-1]) == 2:
                    flow_idx = int(np.ceil((float(frames[f_idx]) - 3) / 2))
                    flow_name = 'frame_' + str(int(np.ceil((float(frames[f_idx]) - 3) / 2))).zfill(10)+ '.jpg'
                    while True:
                        if os.path.exists(path.join(vid_flow_path, 'u', flow_name)):
                            break
                        else:
                            for i in range(100):
                                flow_name = 'frame_' + str(flow_idx - i).zfill(10)+ '.jpg'
                                if os.path.exists(path.join(vid_flow_path, 'u', flow_name)):
                                    left_flow_idx = flow_idx - i
                                    break
                                else:
                                    left_flow_idx = flow_idx - i
                            for i in range(100):
                                flow_name = 'frame_' + str(flow_idx + i).zfill(10)+ '.jpg'
                                if os.path.exists(path.join(vid_flow_path, 'u', flow_name)):
                                    right_flow_idx = flow_idx + i
                                    break
                                else:
                                    right_flow_idx = flow_idx + i
                            
                            if np.minimum(np.abs(right_flow_idx - flow_idx), np.abs(left_flow_idx - flow_idx)) > 20:
                                print('Warning: flow frame too large')
                            if np.abs(right_flow_idx - flow_idx) > np.abs(left_flow_idx - flow_idx):
                                flow_idx = left_flow_idx
                            else:
                                flow_idx = right_flow_idx
                            
                            flow_name = 'frame_' + str(flow_idx).zfill(10)+ '.jpg'
                            break 
                        
                else:
                    flow_name = 'frame_' + str(frames[f_idx]).zfill(10)+ '.jpg'
                info['frames'].append(jpg_name)

                
                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)

                reseed(sequence_seed)
                this_flowu = Image.open(path.join(vid_flow_path, 'u', flow_name)).convert('P')
                this_flowu = self.all_gt_dual_transform(this_flowu)

                reseed(sequence_seed)
                this_flowv = Image.open(path.join(vid_flow_path, 'v', flow_name)).convert('P')
                this_flowv = self.all_gt_dual_transform(this_flowv)

                if f_idx == frames_idx[0] or f_idx == frames_idx[-1]:
                    reseed(sequence_seed)
                    this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('1')
                    this_gt = self.all_gt_dual_transform(this_gt)
                    

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                
                reseed(pairwise_seed)
                this_flowu = self.pair_gt_dual_transform(this_flowu)

                reseed(pairwise_seed)
                this_flowv = self.pair_gt_dual_transform(this_flowv)
                
                if f_idx == frames_idx[0] or f_idx == frames_idx[-1]:
                    reseed(pairwise_seed)
                    this_gt = self.pair_gt_dual_transform(this_gt)
                    
                    this_gt = np.array(this_gt)
                    masks.append(this_gt)

                this_im = self.final_im_transform(this_im)
                
                
                this_flowu = transforms.ToTensor()(this_flowu)
                this_flowv = transforms.ToTensor()(this_flowv)
                this_flowu = this_flowu - torch.mean(this_flowu)
                this_flowv = this_flowv - torch.mean(this_flowv)
                this_flow = torch.cat([this_flowu, this_flowv], dim=0)
                
                
                images.append(this_im)
                flows.append(this_flow)

            images = torch.stack(images, 0)
            flows = torch.stack(flows, 0).float()

            labels = np.unique(masks[0])
            
            
            
            labels = labels[labels!=0]
            
            if len(labels) == 0:
                target_objects = []
                trials += 1
            else:
                target_objects = labels.tolist()
                break
        
        
        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)
        
        info['num_objects'] = max(1, len(target_objects))
        
        
        masks = np.stack(masks, 0)

        
        cls_gt = np.zeros((2, 384, 384), dtype=np.int) 
        first_last_frame_gt = np.zeros((2, self.max_num_obj, 384, 384), dtype=np.int)
        
        
        for i, l in enumerate(target_objects):
            
            
            this_mask = (masks==l)
            
            try:
                cls_gt[this_mask] = i+1
            except:
                print(frames_idx)
                print(cls_gt.shape)
                print(this_mask.shape)
                print(masks.shape)
                print(l)
                print(i)
                print(self.vids[idx])
                raise Exception('error')
            
            
            first_last_frame_gt[:,i] = this_mask
        
        cls_gt = np.expand_dims(cls_gt, 1)

        
        
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        selector = torch.FloatTensor(selector)
        
        data = {
            'rgb': images, 
            'flow': flows, 
            'first_last_frame_gt': first_last_frame_gt, 
            'cls_gt': cls_gt, 
            'selector': selector, 
            'text':video_value['narration'],
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.vids)

if __name__ == '__main__':
    dataset = EPICDataset(data_root='../data', yaml_root='../data/EPIC55_cut_subset_200.yaml', max_jump=20, num_frames=3, max_num_obj=3, finetune=False)
    images = dataset[2]
    print(f"name={images['info']['name']}")
    
    for obj in range(images['first_last_frame_gt'].shape[1]):
        plt.imsave(f"../visuals/gt_{obj}.jpg", images['first_last_frame_gt'][0,obj],cmap='gray')