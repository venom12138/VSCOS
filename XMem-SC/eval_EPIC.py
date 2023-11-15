import os
from os import path
from argparse import ArgumentParser
import shutil
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from dataset.EPIC_testdataset import EPICtestDataset, VideoReader
from inference.data.mask_mapper import MaskMapper
from model.network import XMem
from inference.inference_core import InferenceCore

from progressbar import progressbar
from tqdm import tqdm
from inference.interact.interactive_utils import image_to_torch, index_numpy_to_one_hot_torch, torch_prob_to_numpy_mask, overlay_davis
import clip
import yaml





def colorize_mask(mask):
    
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    
    palette = [0,0,0,255,255,255,128,0,0,0,128,0,0,0,128,255,0,0,255,255,0]
    others = list(np.random.randint(0,255,size=256*3-len(palette)))
    palette.extend(others)
    new_mask.putpalette(palette)

    return new_mask
"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--model', default='./saves/XMem.pth')



parser.add_argument('--EPIC_path', default='./val_data')
parser.add_argument('--yaml_path', default='./val_data/EPIC100_state_positive_val.yaml')
parser.add_argument('--dataset', help='D16/D17/Y18/Y19/LV1/LV3/G', default='EPIC')
parser.add_argument('--split', help='val/test', default='val')
parser.add_argument('--output', default=None)
parser.add_argument('--save_all', action='store_true', 
            help='Save all frames. Useful only in YouTubeVOS/long-time video', )

parser.add_argument('--benchmark', action='store_true', help='enable to disable amp for FPS benchmarking')
        

parser.add_argument('--disable_long_term', action='store_true')

parser.add_argument('--max_mid_term_frames', help='T_max in paper, decrease to save memory', type=int, default=10)
parser.add_argument('--min_mid_term_frames', help='T_min in paper, decrease to save memory', type=int, default=5)
parser.add_argument('--max_long_term_elements', help='LT_max in paper, increase if objects disappear for a long time', 
                                                type=int, default=10000)
parser.add_argument('--num_prototypes', help='P in paper', type=int, default=128)

parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--mem_every', help='r in paper. Increase to improve running speed.', type=int, default=5)
parser.add_argument('--deep_update_every', help='Leave -1 normally to synchronize with mem_every', type=int, default=-1)
parser.add_argument('--fuser_type', default='cross_attention', type=str, choices=['cbam','cross_attention'])

parser.add_argument('--use_flow', type=int, default=0)
parser.add_argument('--use_text', help='whether to use text', type=int, default=0)
parser.add_argument('--use_handmsk', default=0, type=int, choices=[0,1])
parser.add_argument('--openword_test', default=0, type=int, choices=[0,1])


parser.add_argument('--save_scores', action='store_true')


args = parser.parse_args()

config = vars(args)
config['enable_long_term'] = not config['disable_long_term']
config['enable_long_term_count_usage'] = True

if args.output is None:
    
    args.output = f"./output/{args.model.split('/')[-1][:-4]}"
    print(f'Output path not provided. Defaulting to {args.output}')

"""
Data preparation
"""
out_path = args.output


print(out_path)
use_flow = args.use_flow
if 'noflow' in args.model:
    use_flow = False
if use_flow == False:    
    print('not use flow !!!!!!!!!!!')
if args.use_text == 0:
    print('not use text !!!!!!')

val_dataset = EPICtestDataset(args.EPIC_path, args.yaml_path)

torch.autograd.set_grad_enabled(False)
print('load model from {args.model}')

network = XMem(config, args.model).cuda().eval()

total_process_time = 0
total_frames = 0

with open(os.path.join(args.EPIC_path,'val_open_word.yaml'), 'r') as f:
    open_word_info = yaml.safe_load(f)
    

for this_vid in tqdm(val_dataset):
    vid_reader = VideoReader(args.EPIC_path, this_vid)
    vid_name = list(this_vid.keys())[0]
    
    vid_open_word_type = open_word_info[vid_name]
    
    vid_value = this_vid[vid_name]
    vid_length = len(vid_reader)
    
    
    
    
    
    
    
    

    
    processor = InferenceCore(network, config=config)
    first_mask_loaded = False
    
    
    
    
    text = vid_value['narration']
    text = [f"a photo of {text}"]
    
    
    
    if config['use_text']:
        text = clip.tokenize(text).cuda()
        
        text_feat = network.encode_text(text)
        
    for ti, data in enumerate(vid_reader):
        with torch.cuda.amp.autocast(enabled=not args.benchmark):
            
            whether_to_save_mask = int(data['whether_save_mask'][0].cpu())
            
            
            
            
            
                    
            rgb = data['rgb'][0].cuda() 
            flow = data['forward_flow'][0].cuda() 
            hand_mask = data['hand_mask'][0].cuda() 
            
            if ti == 0:
                msk = data['first_frame_gt'][0].cuda() 
                num_objects = msk.shape[0]
                processor.set_all_labels(range(1, num_objects+1))
            else:
                msk = None
            
            frame = data['info']['frames'][0]
            shape = vid_reader.img_size 
            
            need_resize = True
            raw_rgb_path = data['info']['rgb_dir'] + '/' + frame
            raw_frame = np.array(Image.open(raw_rgb_path))
            
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            if not first_mask_loaded:
                if msk is not None:
                    first_mask_loaded = True
                else:
                    
                    continue

            
            
            

            
            
                
                
                
                
            
            
            
            
            
            
            
            prob = processor.step(rgb, flow=flow if use_flow else None, 
                            text=text_feat if config['use_text'] else None, 
                            hand_mask=hand_mask if config['use_handmsk'] else None,
                            mask=msk if (msk is not None) else None, 
                            end=(ti==vid_length-1))         

            
            if need_resize:
                prob = F.interpolate(prob.unsqueeze(1), shape, mode='bilinear', align_corners=False)[:,0]
                
            end.record()
            torch.cuda.synchronize()
            total_process_time += (start.elapsed_time(end)/1000)
            total_frames += 1

            
            

            
            out_mask = torch.argmax(prob, dim=0)
            out_mask = (out_mask.detach().cpu().numpy()).astype(np.uint8)
            
            
            

            
            
            if (args.save_all or whether_to_save_mask) and msk is None:
                
                partition = vid_name.split('_')[0]
                video_part = '_'.join(vid_name.split('_')[:2])
                this_out_path = path.join(out_path, 'all', partition, video_part, vid_name)
                os.makedirs(this_out_path, exist_ok=True)
                
                visualization = overlay_davis(raw_frame, out_mask)
                visual_outpath = path.join(out_path, 'draw', partition, video_part, vid_name)
                if not os.path.isdir(visual_outpath):
                    os.makedirs(visual_outpath)
                plt.imsave(os.path.join(visual_outpath, frame), visualization)
                
                out_mask = colorize_mask(out_mask)
                out_mask.save(os.path.join(this_out_path, frame.replace('jpg','png')))
                if args.openword_test:
                    open_word_path = path.join(out_path, vid_open_word_type, partition, video_part, vid_name)
                    os.makedirs(open_word_path, exist_ok=True)
                    out_mask.save(os.path.join(open_word_path, frame.replace('jpg','png')))
                
                
                
                



print(f'Total processing time: {total_process_time}')
print(f'Total processed frames: {total_frames}')
print(f'FPS: {total_frames / total_process_time}')
print(f'Max allocated memory (MB): {torch.cuda.max_memory_allocated() / (2**20)}')








