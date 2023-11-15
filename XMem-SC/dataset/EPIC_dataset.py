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

class EPICDataset(Dataset):
    """
    Works for EPIC training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, data_root, yaml_root, max_jump, openword_test=False, num_frames=3, max_num_obj=3, finetune=False):
        print('We are using EPIC Dataset !!!!!')
        self.data_root = data_root
        self.max_jump = max_jump
        self.num_frames = num_frames
        self.max_num_obj = max_num_obj
        
        with open(os.path.join(self.data_root, 'train_open_word.yaml'), 'r') as f:
            self.open_word_info = yaml.safe_load(f)
        f.close()
        
        with open(os.path.join(yaml_root), 'r') as f:
            self.data_info = yaml.safe_load(f)
        f.close()
        
        self.vids = [] 
        for key in list(self.data_info.keys()):
            if openword_test:
                if self.open_word_info[key] != 'svsn':
                    continue
            
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
        vid_hand_path = path.join(self.data_root, video_value['participant_id'], 'hand_masks', video_value['video_id'], self.vids[idx])
        frames = list(range(video_value['start_frame'], video_value['stop_frame']))

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
            hands = []
            forward_flows = []
            backward_flows = []
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
                    flow_idx = f_idx
                    flow_name = 'frame_' + str(frames[flow_idx]).zfill(10)+ '.jpg'
                info['frames'].append(jpg_name)

                
                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)

                this_hand = Image.open(path.join(vid_hand_path, png_name)).convert('P')
                reseed(sequence_seed)
                this_hand = self.all_gt_dual_transform(this_hand)
                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_hand = self.pair_gt_dual_transform(this_hand)
                
                
                agg_u_frames = []
                agg_v_frames = []
                u_path = path.join(vid_flow_path, 'u')
                v_path = path.join(vid_flow_path, 'v')
                all_u_jpgs = sorted(glob(f'{u_path}/*.jpg'))
                all_v_jpgs = sorted(glob(f'{v_path}/*.jpg'))
                assert len(all_u_jpgs) > 5 and len(all_v_jpgs) > 5
                u_idx = all_u_jpgs.index(path.join(vid_flow_path, 'u', flow_name))
                v_idx = all_v_jpgs.index(path.join(vid_flow_path, 'v', flow_name))
                if u_idx == 0 or u_idx == 1:
                    agg_u_frames = all_u_jpgs[:5]
                elif u_idx == len(all_u_jpgs) - 1 or u_idx == len(all_u_jpgs) - 2:
                    agg_u_frames = all_u_jpgs[-5:]
                else:
                    agg_u_frames = all_u_jpgs[u_idx-2:u_idx+3]
                
                if v_idx == 0 or v_idx == 1:
                    agg_v_frames = all_v_jpgs[:5]
                elif v_idx == len(all_v_jpgs) - 1 or v_idx == len(all_v_jpgs) - 2:
                    agg_v_frames = all_v_jpgs[-5:]
                else:
                    agg_v_frames = all_v_jpgs[v_idx-2:v_idx+3]
                    
                this_flow = None
                this_backward_flow = None
                
                
                for tmp_idx in range(len(agg_u_frames)):
                    this_flowu = Image.open(agg_u_frames[tmp_idx]).convert('P')
                    this_backward_u = Image.fromarray(255 - np.array(this_flowu), mode='P')
                    
                    reseed(sequence_seed)
                    this_flowu = self.all_gt_dual_transform(this_flowu)
                    reseed(sequence_seed)
                    this_backward_u = self.all_gt_dual_transform(this_backward_u)

                    this_flowv = Image.open(agg_v_frames[tmp_idx]).convert('P')
                    this_backward_v = Image.fromarray(255 - np.array(this_flowv), mode='P')
                    
                    reseed(sequence_seed)
                    this_flowv = self.all_gt_dual_transform(this_flowv)
                    reseed(sequence_seed)
                    this_backward_v = self.all_gt_dual_transform(this_backward_v)
                    
                    reseed(pairwise_seed)
                    this_flowu = self.pair_gt_dual_transform(this_flowu)
                    reseed(pairwise_seed)
                    this_backward_u = self.pair_gt_dual_transform(this_backward_u)

                    reseed(pairwise_seed)
                    this_flowv = self.pair_gt_dual_transform(this_flowv)
                    reseed(pairwise_seed)
                    this_backward_v = self.pair_gt_dual_transform(this_backward_v)
                    
                    
                    this_flowu = transforms.ToTensor()(this_flowu)
                    this_flowv = transforms.ToTensor()(this_flowv)
                    this_flowu = this_flowu - torch.mean(this_flowu)
                    this_flowv = this_flowv - torch.mean(this_flowv)
                    
                    
                    this_backward_u = transforms.ToTensor()(this_backward_u)
                    this_backward_v = transforms.ToTensor()(this_backward_v)
                    this_backward_u = this_backward_u - torch.mean(this_backward_u)
                    this_backward_v = this_backward_v - torch.mean(this_backward_v)
                    
                    
                    if this_flow == None:
                        this_flow = torch.cat([this_flowu, this_flowv], dim=0)
                    else:
                        this_flow = torch.cat([this_flow, this_flowu, this_flowv], dim=0)
                    
                    if this_backward_flow == None:
                        this_backward_flow = torch.cat([this_backward_u, this_backward_v], dim=0)
                    else:
                        this_backward_flow = torch.cat([this_backward_flow, this_backward_u, this_backward_v], dim=0)
                    
                if f_idx == frames_idx[0] or f_idx == frames_idx[-1]:
                    reseed(sequence_seed)
                    this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('1')
                    this_gt = self.all_gt_dual_transform(this_gt)

                
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                
                if f_idx == frames_idx[0] or f_idx == frames_idx[-1]:
                    reseed(pairwise_seed)
                    this_gt = self.pair_gt_dual_transform(this_gt)
                    
                    this_gt = np.array(this_gt)
                    masks.append(this_gt)

                this_im = self.final_im_transform(this_im)
                
                hands.append(this_hand)
                images.append(this_im)
                forward_flows.append(this_flow)
                backward_flows.append(this_backward_flow)

            images = torch.stack(images, 0)
            forward_flows = torch.stack(forward_flows, 0).float()
            backward_flows = torch.stack(backward_flows, 0).float()
            
            hands = np.stack(hands, 0)
            
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
        
        hands_gt = np.zeros((self.num_frames, 2, 384, 384), dtype=np.int)
        
        for hand_idx in range(2):
            this_hand = (hands==hand_idx+1)
            hands_gt[:, hand_idx] = this_hand
        
        
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
            'forward_flow': forward_flows, 
            'backward_flow': backward_flows, 
            'first_last_frame_gt': first_last_frame_gt, 
            'cls_gt': cls_gt, 
            'selector': selector, 
            'text':video_value['narration'],
            'info': info,
            'hand_mask': hands_gt, 
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