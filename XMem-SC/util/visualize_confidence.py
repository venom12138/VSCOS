import torch
import glob
if torch.cuda.is_available():
    print('Using GPU')
    device = 'cuda'
else:
    print('CUDA not available. Please connect to a GPU instance if possible.')
    device = 'cpu'

import os
from os import path
from argparse import ArgumentParser
import shutil

import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image

from inference.data.test_datasets import LongTestDataset, DAVISTestDataset, YouTubeVOSTestDataset
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore

from progressbar import progressbar
import cv2
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis
import matplotlib.pyplot as plt
import seaborn


torch.set_grad_enabled(False)


config = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 128,
    'min_mid_term_frames': 5,
    'max_mid_term_frames': 10,
    'max_long_term_elements': 10000,
}

network = XMem(config, './saves/XMem.pth').eval().to(device)

conf_save_path = './data/P04/conf_masks/P04_01'

video_path = '/cluster/home2/yjw/venom/XMem/data/P04/positive_frames/P04_01/7967'

mask_name = '/cluster/home2/yjw/venom/EPIC-data/data/P04/first_last_masks/P04_01/7967/frame_0000012756.jpg'
uid = video_path.split('/')[-1]

if not os.path.isdir(f"{conf_save_path}/{uid}"):
    os.makedirs(f"{conf_save_path}/{uid}")




mask = np.array(Image.open(mask_name).convert('1'),dtype=np.int32)
print(np.unique(mask))
print(mask.shape)
num_objects = len(np.unique(np.round(mask))) - 1

"""

torch.cuda.empty_cache()

processor = InferenceCore(network, config=config)
processor.set_all_labels(range(1, num_objects+1)) 



frames_to_propagate = 1000
visualize_every = 20

current_frame_index = 0

with torch.cuda.amp.autocast(enabled=True):
    for frame_path in sorted(glob.glob(f'{video_path}/*.jpg')):
        
        frame = np.array(Image.open(frame_path))
        
        print(frame_path)
        if frame is None or current_frame_index > frames_to_propagate:
            break

        
        frame_torch, _ = image_to_torch(frame, device=device)
        if current_frame_index == 0:
            
            mask_torch = index_numpy_to_one_hot_torch(mask, num_objects+1).to(device)
            
            
            prediction = processor.step(frame_torch, mask_torch[1:])
        else:
            
            prediction = processor.step(frame_torch)

        
        
        
        prediction = torch.abs(prediction[0] - prediction[1]).cpu().numpy()
        
        
        
        
        plt.figure()
        ax = seaborn.heatmap(prediction, cmap='coolwarm', vmin=0, vmax=1, )
        ax.get_figure().savefig(f"{conf_save_path}/{uid}/{frame_path.split('/')[-1]}")
        plt.clf()
        
        
        
        
            
            
        

        current_frame_index += 1






