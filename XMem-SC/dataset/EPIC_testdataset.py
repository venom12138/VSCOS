import os
from os import path, replace

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
from torch.utils.data import DataLoader
from glob import glob
import skimage.measure as measure

class VideoReader(Dataset):
    
    def __init__(self, data_root, video_info, max_num_obj=3, ):
        self.data_root = data_root
        self.video_info = video_info
        self.max_num_obj = max_num_obj
        
        
        self.im_transform = transforms.Compose([
            transforms.Resize((384,384), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((384,384), interpolation=InterpolationMode.NEAREST),
        ])
        
        video_value = video_info[list(video_info.keys())[0]]
        
        vid_im_path = path.join(self.data_root, video_value['participant_id'], 'rgb_frames', video_value['video_id'], list(video_info.keys())[0])
        frames = list(range(video_value['start_frame'], video_value['stop_frame']))
        jpg_name = 'frame_' + str(frames[1]).zfill(10)+ '.jpg'
        this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
        self.img_size = [this_im.size[1], this_im.size[0]] 
        
        
        vid_gt_path = path.join(self.data_root, video_value['participant_id'], 'anno_masks', video_value['video_id'], list(video_info.keys())[0])
        png_name = 'frame_' + str(frames[0]).zfill(10)+ '.png'
        this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('1')
        labels = np.unique(this_gt)
        labels = labels[labels!=0]
        if len(labels) == 0:
            target_objects = []
        else:
            target_objects = labels.tolist()
        
        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)
        self.target_objects = target_objects
        
        
        self.frames = []
        for idx in range(len(frames)):
            png_name = 'frame_' + str(frames[idx]).zfill(10)+ '.png'
            if not os.path.isfile(path.join(vid_gt_path, png_name)):
                if idx % 5 == 0:
                    self.frames.append(frames[idx])
            else:
                self.frames.append(frames[idx])
        
        
        
        
    def __getitem__(self, idx):
        info = {}
        info['name'] = list(self.video_info.keys())[0]
        video_value = self.video_info[list(self.video_info.keys())[0]]
        info['frames'] = []
        vid_im_path = path.join(self.data_root, video_value['participant_id'], 'rgb_frames', video_value['video_id'], info['name'])
        info['rgb_dir'] = vid_im_path
        
        vid_gt_path = path.join(self.data_root, video_value['participant_id'], 'anno_masks', video_value['video_id'], info['name'])
        vid_flow_path = path.join(self.data_root, video_value['participant_id'], 'flow_frames', video_value['video_id'], info['name'])
        vid_hand_path = path.join(self.data_root, video_value['participant_id'], 'hand_masks', video_value['video_id'], info['name'])
        
        sequence_seed = np.random.randint(2147483647)
        images = []
        masks = []
        hands = []
        masks_count = [] 
        forward_flows = []
        backward_flows = []
        
        info['num_objects'] = max(1, len(self.target_objects))
        f_idx = idx
        
        jpg_name = 'frame_' + str(self.frames[f_idx]).zfill(10)+ '.jpg'
        png_name = 'frame_' + str(self.frames[f_idx]).zfill(10)+ '.png'
        if len(video_value['video_id'].split('_')[-1]) == 2:
            flow_name = 'frame_' + str(int(np.ceil((float(self.frames[f_idx]) - 3) / 2))).zfill(10)+ '.jpg'
        else:
            flow_name = 'frame_' + str(self.frames[f_idx]).zfill(10)+ '.jpg'
        info['frames'].append(jpg_name)

        
        reseed(sequence_seed)
        this_im = Image.open(path.join(vid_im_path, jpg_name))
        this_im = self.im_transform(this_im)

        this_hand = Image.open(path.join(vid_hand_path, png_name)).convert('P')
        reseed(sequence_seed)
        this_hand = self.gt_transform(this_hand)
        
        
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
            this_flowu = Image.open(agg_u_frames[tmp_idx]).convert('P').resize((384,384))
            this_backward_u = Image.fromarray(255 - np.array(this_flowu), mode='P')
            this_flowv = Image.open(agg_v_frames[tmp_idx]).convert('P').resize((384,384))
            this_backward_v = Image.fromarray(255 - np.array(this_flowv), mode='P')
            
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
            
        if os.path.isfile(path.join(vid_gt_path, png_name)):
            masks_count.append(1)
            if f_idx == 0:
                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('1')
                this_gt = self.gt_transform(this_gt)
                this_gt = np.array(this_gt)          
                this_gt = this_gt.squeeze()
                masks.append(this_gt)
        else:
            masks_count.append(0)
        
        images.append(this_im)
        forward_flows.append(this_flow)
        backward_flows.append(this_backward_flow)
        hands.append(this_hand)
        
        images = torch.stack(images, 0)
        forward_flows = torch.stack(forward_flows, 0).float()
        backward_flows = torch.stack(backward_flows, 0).float()
        hands = np.stack(hands, 0)
        
        masks_count = torch.tensor(masks_count, dtype=torch.int)
        
        hands_gt = np.zeros((1, 2, 384, 384), dtype=np.int)
        
        for hand_idx in range(2):
            this_hand = (hands==hand_idx+1)
            hands_gt[:, hand_idx] = this_hand
        hands_gt = torch.tensor(hands_gt, dtype=torch.int)
        
        
        
        
        
        
        
        
        
        
        if f_idx == 0:
            
            
            
            
            masks = np.stack(masks, 0)
            assert masks.shape[0] == 1

            
            cls_gt = np.zeros((1, 384, 384), dtype=np.int32) 
            first_frame_gt = np.zeros((1, len(self.target_objects), 384, 384), dtype=np.int32)
        
            
            for i, l in enumerate(self.target_objects):
                
                
                this_mask = (masks==l)
                
                try:
                    cls_gt[this_mask] = i+1
                except:
                    print(cls_gt.shape)
                    print(this_mask.shape)
                    print(masks.shape)
                    print(l)
                    print(i)
                    
                    raise Exception('error')
                
                
                first_frame_gt[:,i] = this_mask
            
            cls_gt = np.expand_dims(cls_gt, 1)

        
        
        selector = [1 if i < info['num_objects'] else 0 for i in range(len(self.target_objects))]
        selector = torch.FloatTensor(selector)
        
        
        if f_idx == 0:
            data = {
                'rgb': images, 
                'forward_flow': forward_flows, 
                'backward_flow': backward_flows, 
                'first_frame_gt': torch.tensor(first_frame_gt), 
                'cls_gt': torch.tensor(cls_gt), 
                'selector': selector, 
                'info': info,
                
                'whether_save_mask': masks_count, 
                'hand_mask': hands_gt, 
            }
        else:
            data = {
                'rgb': images, 
                'forward_flow': forward_flows, 
                'backward_flow': backward_flows, 
                'selector': selector, 
                'info': info,
                
                'whether_save_mask': masks_count, 
                'hand_mask': hands_gt, 
            }

        return data
        
    def __len__(self):
        return len(self.frames)

class EPICtestDataset(Dataset):
    """
    Works for EPIC training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, data_root, yaml_root, max_num_obj=3, ):
        print('We are using EPIC testDataset !!!!!')
        self.data_root = data_root
        self.max_num_obj = max_num_obj
        with open(os.path.join(yaml_root), 'r') as f:
            self.data_info = yaml.safe_load(f)
        self.vids = list(self.data_info.keys())
        
        for k in list(self.data_info.keys()):
            video_value = self.data_info[k]
            vid_gt_path = path.join(self.data_root, video_value['participant_id'], 'anno_masks', video_value['video_id'], k)
            frame_name = video_value['start_frame']
            jpg_name = 'frame_' + str(frame_name).zfill(10)+ '.jpg'
            png_name = 'frame_' + str(frame_name).zfill(10)+ '.png'
            if not os.path.isfile(os.path.join(vid_gt_path, jpg_name)) and not os.path.isfile(os.path.join(vid_gt_path, png_name)):
                self.vids.remove(k)

    def __getitem__(self, idx):
        video_value = self.data_info[self.vids[idx]] 

        return {self.vids[idx]: video_value}

    def __len__(self):
        return len(self.vids)







































        










        









        






        







































            






                






                





                





                











            

            


        








        




            
        














        


























        
















if __name__ == '__main__':
    dataset = EPICtestDataset(data_root='../data', yaml_root='../data/EPIC55_cut_subset_200.yaml', max_num_obj=3, finetune=False)
    val_loader = DataLoader(dataset, 1,  shuffle=False, num_workers=4)
    for i, data in enumerate(val_loader):
        print(data['info']['name'][0])
        print(data['rgb'][0].shape)
        print(data['first_frame_gt'][0][0].shape)
        print(data['whether_save_mask'][0][1])
        
        dd
    
    
    
    
    
    