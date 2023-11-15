import os
import numpy as np
from PIL import Image
import sys


class Results(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir

    def _read_mask(self, sequence, frame_id):
        try:
            partition = sequence.split('_')[0]
            video_id = '_'.join(sequence.split('_')[:-1])
            
            mask_path = os.path.join(self.root_dir, partition, video_id, sequence, f'{frame_id}.png')
            
            
            if os.path.exists(mask_path):
                return np.array(Image.open(mask_path).convert('1'))
            else:
                mask_path = os.path.join(self.root_dir, partition, video_id, sequence, f'{frame_id}.jpg')
                return np.array(Image.open(mask_path).convert('1'))
                
        except:
            print(sequence + " frame %s not found!\n" % frame_id)
            return None
            
            
            
            

    def read_masks(self, sequence, masks_id):
        mask_0 = self._read_mask(sequence, masks_id[0])
        try:
            if mask_0 == None:
                return None
        except:
            pass
        masks = np.zeros((len(masks_id), *mask_0.shape))
        for ii, m in enumerate(masks_id):
            masks[ii, ...] = self._read_mask(sequence, m)
        num_objects = int(np.max(masks))
        tmp = np.ones((num_objects, *masks.shape))
        tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
        masks = (tmp == masks[None, ...]) > 0
        return masks
