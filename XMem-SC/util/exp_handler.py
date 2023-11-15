import csv
import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import wandb
import yaml
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class ExpHandler:
    _home = Path.home()

    def __init__(self, config=None, en_wandb=False, resume=''):
        exp_name = os.getenv('exp_name', default='default_group')
        run_name = os.getenv('run_name', default='default_name')
        self._exp_id = f'{run_name}'
        self._exp_name = exp_name
        self._run_name = run_name
        self._en_wandb = en_wandb
        if resume != '' and (Path(resume) / 'config.yaml').exists():
            print('----------resuming-----------')
            self._save_dir = Path(resume)
            with open(self._save_dir / 'config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            self._exp_id = config['exp_id']
            self._exp_name = config['exp_name']
            self._run_name = config['run_name']
            if en_wandb:
                self.wandb_run = wandb.init(group=self._exp_name, name=self._run_name, save_code=True,
                                            id=config['wandb_id'], resume='allow')
        else:
            self._save_dir = os.path.join('{}/.exp/{}'.format(self._home, os.getenv('WANDB_PROJECT', default='default_project')),
                                    exp_name, self._exp_id)
            if not os.path.exists(self._save_dir):
                os.makedirs(self._save_dir)
            if en_wandb:
                self.wandb_run = wandb.init(group=exp_name, name=run_name, settings=wandb.Settings(start_method="fork"), save_code=True)
                test_step = wandb.define_metric('test_step')
                wandb.define_metric(name='eval/all/JF_mean', step_metric=test_step)
                wandb.define_metric(name='eval/all/J_mean', step_metric=test_step)
                wandb.define_metric(name='eval/all/F_mean', step_metric=test_step)
                wandb.define_metric(name='eval/half/JF_mean', step_metric=test_step)
                wandb.define_metric(name='eval/half/J_mean', step_metric=test_step)
                wandb.define_metric(name='eval/half/F_mean', step_metric=test_step)
            self.save_config(config)
        sym_dest = self._get_sym_path('N')
        

        self._logger = self._init_logger()
        

    def log_eval_acc(self, all_JF_mean, all_J_mean, all_F_mean, 
            half_JF_mean, half_J_mean, half_F_mean, step):
        if self._en_wandb:
            wandb.log({'test_step': step, 'eval/all/JF_mean': all_JF_mean, \
                'eval/all/J_mean': all_J_mean, 'eval/all/F_mean': all_F_mean, \
                'eval/half/JF_mean': half_JF_mean, 'eval/half/J_mean': half_J_mean, \
                'eval/half/F_mean': half_F_mean})
            
    def log_image(self, images):
        image = wandb.Image(images, )
        wandb.log({"images": image})
        
    @staticmethod
    def resume_sanity(new_conf, old_conf):
        print('-' * 10, 'Resume sanity check', '-' * 10)
        old_config_hashable = {k: tuple(v) if isinstance(v, list) else v for k, v in old_conf.items()
                                if k not in new_conf['resume_check_exclude_keys']}
        new_config_hashable = {k: tuple(v) if isinstance(v, list) else v for k, v in new_conf.items()
                                if k not in new_conf['resume_check_exclude_keys']}
        print(f'Diff config: {set(old_config_hashable.items()) ^ set(new_config_hashable.items())}')
        assert old_config_hashable == new_config_hashable, 'Resume sanity check failed'

    def _get_sym_path(self, state):
        sym_dir = f'{self._home}/.exp/syms'
        if not os.path.exists(sym_dir):
            os.makedirs(sym_dir)

        sym_dest = os.path.join(sym_dir, '--'.join([self._exp_id, state, self._exp_name]))
        return sym_dest

    @property
    def save_dir(self):
        return self._save_dir

    def _init_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        fh = logging.FileHandler(os.path.join(self._save_dir, f'{self._exp_id}_log.txt'))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)

        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

    def save_config(self, conf):
        
        conf['exp_id'] = self._exp_id
        conf['exp_name'] = self._exp_name
        conf['run_name'] = self._run_name
        conf['commit'] = os.getenv('commit', default='not_set')
        if hasattr(self, 'wandb_run'):
            conf['wandb_id'] = self.wandb_run.id
        with open(f'{self._save_dir}/config.yaml', 'w') as f:
            yaml.dump(conf, f)

        if self._en_wandb:
            wandb.config.update(conf,allow_val_change=True)

    def write(self, prefix, eval_metrics=None, train_metrics=None, **kwargs):
        rowd = OrderedDict([(f'{prefix}/{k}', v) for k, v in kwargs.items() ])
        if eval_metrics:
            rowd.update([(f'{prefix}/eval_' + k, v) for k, v in eval_metrics.items()])
        if train_metrics:
            rowd.update([(f'{prefix}/train_' + k, v) for k, v in train_metrics.items()])

        path = os.path.join(self._save_dir, f'{self._exp_id}_{prefix}_summary.csv')
        initial = not os.path.exists(path)
        with open(path, mode='a') as cf:
            dw = csv.DictWriter(cf, fieldnames=rowd.keys())
            if initial:
                dw.writeheader()
            dw.writerow(rowd)

        if self._en_wandb:
            wandb.log(rowd)

    def log(self, msg):
        self._logger.info(msg)

    def finish(self):
        Path(f'{self._save_dir}/finished').touch()
        os.rename(self._get_sym_path('N'), self._get_sym_path('Y'))


def pair_pics_together(selected_pics_info):
    h, w = [456, 256]
    
    cate_counts = len(selected_pics_info[list(selected_pics_info.keys())[0]].keys()) 
    rows_counts = len(selected_pics_info[list(selected_pics_info.keys())[0]]['pred_path']) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    output_imgs = []
    for key, value in selected_pics_info.items():
        output_image = np.zeros([w*cate_counts, h*(rows_counts+1), 3], dtype=np.uint8)
        col_cnt = 0 
        skip_this_video = False
        
        
        for k, v in value.items():
            
            caption = key + k.split('_')[0]

            
            dy = 40
            for i, line in enumerate(caption.split('\n')):
                cv2.putText(output_image, line, (10, col_cnt*w+100+i*dy),
                        font, 0.8, (255,255,255), 2, cv2.LINE_AA)
            for row_cnt, img_path in enumerate(v):
                try:
                    if 'rgb' in k:
                        img = np.array(Image.open(img_path))
                    else:
                        img = np.array(Image.open(img_path))
                        img = (img * 255).astype('uint8')
                except:
                    print(f'img not found:{img_path}')
                    skip_this_video = True
                    break
                im_shape = img.shape
                if len(im_shape) == 2:
                    img = img[..., np.newaxis]

                output_image[(col_cnt+0)*w:(col_cnt+1)*w,
                            (row_cnt+1)*h:(row_cnt+2)*h, :] = img
            col_cnt += 1
        if not skip_this_video:
            
            
            
            output_imgs.append(output_image)
    return output_imgs

def consume_prefix_in_state_dict_if_present(
        state_dict, prefix
):
    r"""Strip the prefix in state_dict in place, if any.
    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

    
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            
            
            
            

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)