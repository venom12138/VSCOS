import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
import numpy as np



def dice_loss(input_mask, cls_gt): 
    num_objects = input_mask.shape[1] 
    losses = []
    for i in range(num_objects):
        mask = input_mask[:,i].flatten(start_dim=1) 
        
        gt = (cls_gt==(i+1)).float().flatten(start_dim=1) 
        numerator = 2 * (mask * gt).sum(-1)
        denominator = mask.sum(-1) + gt.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        losses.append(loss)
    return torch.cat(losses).mean()




def dice_loss_between_mask(mask1, tgt_mask): 
    num_classes = mask1.shape[1]
    tgt_mask = torch.argmax(tgt_mask, dim=1)
    
    tgt_mask = F.one_hot(tgt_mask, num_classes=num_classes+1).permute(0, 3, 1, 2).float()
    
    mask1 = mask1.flatten(start_dim=2) 
    tgt_mask = tgt_mask.flatten(start_dim=2) 
    tgt_mask = tgt_mask[:,1:,:] 
    
    
    numerator = 2 * (mask1 * tgt_mask).sum(-1)
    
    
    
    denominator = mask1.sum(-1) + tgt_mask.sum(-1)
    
    loss = 1 - (numerator + 1) / (denominator + 1)
    
    
    return loss.mean()



    




    

    

    



class BootstrappedCE(nn.Module):
    def __init__(self, start_warm, end_warm, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            
            
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1) 
        num_pixels = raw_loss.numel() 

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p

class BootstrappedKL(nn.Module):
    def __init__(self, start_warm, end_warm, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p
    
    def forward(self, input, target, it):
        if it < self.start_warm:
            
            
            return F.kl_div(input.softmax(dim=1).log(), target.detach().softmax(dim=1), reduction='sum')/input.shape[-1]/input.shape[-2], 1.0
        
        
        raw_loss = F.kl_div(input.softmax(dim=1).log(), target.detach().softmax(dim=1), reduction='none').sum(dim=1).view(-1) 
        num_pixels = raw_loss.numel() 

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p

class LossComputer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bce = BootstrappedCE(config['start_warm'], config['end_warm'])
        
        
        self.start_w = 1
        self.end_w = 0.0
        
    def compute(self, data, num_objects, it):
        losses = defaultdict(int)

        b, t = data['rgb'].shape[:2]
        
        losses['total_loss'] = 0
        for ti in range(0, t):
            
            
            
            weight = self.end_w + (self.start_w - self.end_w) * (ti / (t-1))
            for bi in range(b):
                if ti == t-1:
                    loss, p = self.bce(data[f'flogits_{ti}'][bi:bi+1, :num_objects[bi]+1], data['cls_gt'][bi:bi+1,1,0], it)
                    losses['p'] += p / b / 2 
                    losses[f'ce_loss_{ti}'] += loss / b
                elif ti == 0:
                    loss, p = self.bce(data[f'blogits_{ti}'][bi:bi+1, :num_objects[bi]+1], data['cls_gt'][bi:bi+1,0,0], it)
                    losses['p'] += p / b / 2 
                    losses[f'ce_loss_{ti}'] += loss / b
                
                
                
                
                
                
                
                

            losses['total_loss'] += losses['ce_loss_%d'%ti]
            
            if ti == 0:
                losses[f'dice_loss_{ti}'] = dice_loss(data[f'bmasks_{ti}'], data['cls_gt'][:,0,0]) 
                losses['total_loss'] += losses[f'dice_loss_{ti}']
            elif ti == t - 1:
                losses[f'dice_loss_{ti}'] = dice_loss(data[f'fmasks_{ti}'], data['cls_gt'][:,1,0]) 
                losses['total_loss'] += losses[f'dice_loss_{ti}']
            else:
                if self.config['use_dice_align']:
                    if np.random.rand() < weight:
                        losses[f'dice_loss_{ti}'] = dice_loss_between_mask(data[f'fmasks_{ti}'], data[f'blogits_{ti}'].detach())
                    else:
                        losses[f'dice_loss_{ti}'] = dice_loss_between_mask(data[f'bmasks_{ti}'], data[f'flogits_{ti}'].detach())
                    losses['total_loss'] += losses[f'dice_loss_{ti}']
                if it >= self.config['teacher_warmup']:
                    if self.config['use_teacher_model']:
                        losses[f'sftf_dice_loss_{ti}'] = self.config['teacher_loss_weight']*dice_loss_between_mask(data[f'fmasks_{ti}'], data[f't_flogits_{ti}'].detach())
                        losses[f'sbtb_dice_loss_{ti}'] = self.config['teacher_loss_weight']*dice_loss_between_mask(data[f'bmasks_{ti}'], data[f't_blogits_{ti}'].detach())
                        losses['total_loss'] += losses[f'sftf_dice_loss_{ti}'] + losses[f'sbtb_dice_loss_{ti}']
                        if self.config['ts_all_align_loss']:
                            losses[f'sftb_dice_loss_{ti}'] = self.config['teacher_loss_weight']*dice_loss_between_mask(data[f'fmasks_{ti}'], data[f't_blogits_{ti}'].detach())
                            losses[f'sbtf_dice_loss_{ti}'] = self.config['teacher_loss_weight']*dice_loss_between_mask(data[f'bmasks_{ti}'], data[f't_flogits_{ti}'].detach())
                            losses['total_loss'] += losses[f'sftb_dice_loss_{ti}'] + losses[f'sbtf_dice_loss_{ti}']
        return losses








        


























        








        











        












        


if __name__ == '__main__':
    keys = torch.rand(2, 256, 8, 24, 24)
    rand_loss = Random_Walk_Loss()
    loss, diags = rand_loss.compute_random_walk_loss(keys)
    print(loss)
    print(diags)