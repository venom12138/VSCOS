import os
from glob import glob
from collections import defaultdict
import numpy as np
from PIL import Image
import yaml

class DAVIS(object):
    SUBSET_OPTIONS = ['train', 'val', 'test-dev', 'test-challenge']
    TASKS = ['semi-supervised', 'unsupervised']
    DATASET_WEB = 'https://davischallenge.org/davis2017/code.html'
    VOID_LABEL = 255

    def __init__(self, root, yaml_root, sequences='all', resolution='480p', codalab=False):
        """
        Class to read the DAVIS dataset
        :param root: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to load the annotations, choose between semi-supervised or unsupervised.
        :param subset: Set to load the annotations
        :param sequences: Sequences to consider, 'all' to use all the sequences in a set.
        :param resolution: Specify the resolution to use the dataset, choose between '480' and 'Full-Resolution'
        """
        self.root = root
        self.yaml_root = yaml_root
        with open(self.yaml_root, 'r') as f:
            self.data_info = yaml.safe_load(f)
            
        self.vids = list(self.data_info.keys())
        for k in self.vids:
            partition = k.split('_')[0]
            video_id = '_'.join(k.split('_')[:-1])
            masks_path = os.path.join(self.root, partition, 'anno_masks', video_id, k)
            if not os.path.isdir(masks_path):
                
                self.vids.remove(k)

        self.sequences = defaultdict(dict)

        for seq in self.vids:
            partition = seq.split('_')[0]
            video_id = '_'.join(seq.split('_')[:-1])
            masks_path = os.path.join(self.root, partition, 'anno_masks', video_id, seq)
            masks = np.sort(glob(f'{masks_path}/*.png')).tolist()
            if sequences == 'all':
                if len(masks) > 2:
                    self.sequences[seq]['masks'] = masks
                
                
            elif sequences == 'second_half':
                mask_len = len(masks)
                if len(masks) > 2:
                    self.sequences[seq]['masks'] = masks[max(int(mask_len//2)-1,0):]
                else:
                    self.sequences[seq]['masks'] = masks

    def _get_all_elements(self, sequence, obj_type):
        obj = np.array(Image.open(self.sequences[sequence][obj_type][0]).convert('1'))
        all_objs = np.zeros((len(self.sequences[sequence][obj_type]), *obj.shape))
        obj_id = []
        for i, obj in enumerate(self.sequences[sequence][obj_type]):
            all_objs[i, ...] = np.array(Image.open(obj).convert('1'))
            obj_id.append(''.join(obj.split('/')[-1].split('.')[:-1]))
        return all_objs, obj_id

    def get_all_masks(self, sequence, separate_objects_masks=False):
        masks, masks_id = self._get_all_elements(sequence, 'masks')
        masks_void = np.zeros_like(masks)

        
        for i in range(masks.shape[0]):
            masks_void[i, ...] = masks[i, ...] == 255
            masks[i, masks[i, ...] == 255] = 0

        if separate_objects_masks:
            num_objects = int(np.max(masks[0, ...])) 
            tmp = np.ones((num_objects, *masks.shape)) 
            tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
            masks = (tmp == masks[None, ...])
            masks = masks > 0
        return masks, masks_void, masks_id

    def get_sequences(self):
        for seq in self.sequences:
            yield seq


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    only_first_frame = True
    subsets = ['train', 'val']

    for s in subsets:
        dataset = DAVIS(root='/home/csergi/scratch2/Databases/DAVIS2017_private', subset=s)
        for seq in dataset.get_sequences():
            g = dataset.get_frames(seq)
            img, mask = next(g)
            plt.subplot(2, 1, 1)
            plt.title(seq)
            plt.imshow(img)
            plt.subplot(2, 1, 2)
            plt.imshow(mask)
            plt.show(block=True)

