import math
import numpy as np
import torch
from typing import Optional


def get_similarity(mk, ms, qk, qe):
    
    
    
    
    
    
    
    
    
    CK = mk.shape[1]
    mk = mk.flatten(start_dim=2)
    
    ms = ms.flatten(start_dim=1).unsqueeze(2) if ms is not None else None
    qk = qk.flatten(start_dim=2)
    qe = qe.flatten(start_dim=2) if qe is not None else None

    if qe is not None:
        
        
        mk = mk.transpose(1, 2)
        
        
        
        
        a_sq = (mk.pow(2) @ qe)
        
        
        
        
        two_ab = 2 * (mk @ (qk * qe))
        
        
        
        b_sq = (qe * qk.pow(2)).sum(1, keepdim=True)
        
        
        similarity = (-a_sq+two_ab-b_sq) 
    else:
        
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        two_ab = 2 * (mk.transpose(1, 2) @ qk)
        similarity = (-a_sq+two_ab)
    
    if ms is not None:
        similarity = similarity * ms / math.sqrt(CK)   
    else:
        similarity = similarity / math.sqrt(CK)   
    
    return similarity

def do_softmax(similarity, top_k: Optional[int]=None, inplace=False, return_usage=False):
    
    
    
    if top_k is not None:
        values, indices = torch.topk(similarity, k=top_k, dim=1)

        x_exp = values.exp_()
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
        if inplace:
            similarity.zero_().scatter_(1, indices, x_exp) 
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp) 
    else:
        
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        
        x_exp = torch.exp(similarity - maxes)
        
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        
        affinity = x_exp / x_exp_sum 
        indices = None

    if return_usage:
        return affinity, affinity.sum(dim=2)

    return affinity

def get_affinity(mk, ms, qk, qe):
    
    similarity = get_similarity(mk, ms, qk, qe)
    affinity = do_softmax(similarity)
    return affinity

def readout(affinity, mv):
    B, CV, T, H, W = mv.shape 

    mo = mv.view(B, CV, T*H*W) 
    mem = torch.bmm(mo, affinity) 
    mem = mem.view(B, CV, H, W)

    return mem
