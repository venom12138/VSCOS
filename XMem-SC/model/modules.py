"""
modules.py - This file stores the rather boring network blocks.

x - usually means features that only depends on the image
g - usually means features that also depends on the mask. 
    They might have an extra "group" or "num_objects" dimension, hence
    batch_size * num_objects * num_channels * H * W

The trailing number of a variable usually denote the stride

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.group_modules import *
from model import resnet
from model.cbam import CBAM



from torchsummary import summary
from einops import rearrange

EPS = 1e-20

class FeatureFusionBlock(nn.Module):
    def __init__(self, x_in_dim, g_in_dim, g_mid_dim, g_out_dim):
        super().__init__()

        self.distributor = MainToGroupDistributor()
        self.block1 = GroupResBlock(x_in_dim+g_in_dim, g_mid_dim)
        self.attention = CBAM(g_mid_dim)
        self.block2 = GroupResBlock(g_mid_dim, g_out_dim)

    def forward(self, x, g):
        batch_size, num_objects = g.shape[:2]
        
        g = self.distributor(x, g)
        
        g = self.block1(g)
        
        r = self.attention(g.flatten(start_dim=0, end_dim=1))
        r = r.view(batch_size, num_objects, *r.shape[1:])

        g = self.block2(g+r)
        
        return g

class ValueFuser(nn.Module):
    
    def __init__(self, x_in_dim, f_in_dim, t_in_dim, h_in_dim, out_dim):
        super().__init__()

        self.block1 = GroupResBlock(x_in_dim + f_in_dim + t_in_dim + h_in_dim, out_dim)
        self.attention = CBAM(out_dim)
        self.block2 = GroupResBlock(out_dim, out_dim)
    
    
    
    
    
    
    def forward(self, memory_value, flow_feat_16=None, text_feat=None, hand_feat=None):
        batch_size, num_objects = memory_value.shape[:2]
        HbyP, WbyP = memory_value.shape[-2:]
        if flow_feat_16 != None:
            flow_feat_16 = flow_feat_16.unsqueeze(1).repeat(1, num_objects, 1, 1, 1) 
            memory_value = torch.cat([memory_value, flow_feat_16], dim=2) 
        if text_feat != None:
            text_feat = text_feat.unsqueeze(1).unsqueeze(3).unsqueeze(3).repeat(1, num_objects, 1, HbyP, WbyP) 
            memory_value = torch.cat([memory_value, text_feat], dim=2) 
        if hand_feat != None:
            hand_feat = hand_feat.unsqueeze(1).repeat(1, num_objects, 1, 1, 1) 
            memory_value = torch.cat([memory_value, hand_feat], dim=2) 
        
        
        memory_value = self.block1(memory_value)
        
        r = self.attention(memory_value.flatten(start_dim=0, end_dim=1))
        r = r.view(batch_size, num_objects, *r.shape[1:])

        memory_value = self.block2(memory_value+r)
        
        
        return memory_value

class CrossAttentionValueFuser(nn.Module):
    def __init__(self, x_in_dim, f_in_dim, out_dim, hidden_channels=256):
        super().__init__()
        self.attention_layer1 = CrossAttentionLayer(query_dim=x_in_dim, refer_dim=f_in_dim, hidden_channels=hidden_channels)
        self.attention_layer2 = CrossAttentionLayer(query_dim=f_in_dim, refer_dim=x_in_dim, hidden_channels=hidden_channels)
        self.dimension_reduction = nn.Conv2d(
            2*(x_in_dim+f_in_dim), out_dim, kernel_size=1, stride=1, padding=0)
    
    def forward(self, memory_value, flow_feat_16, text_feat=None):
        assert text_feat == None
        batch_size, num_objects = memory_value.shape[:2]
        HbyP, WbyP = memory_value.shape[-2:]
        flow_feat_16 = flow_feat_16.unsqueeze(1).repeat(1, num_objects, 1, 1, 1) 
        flow_feat_16 = flow_feat_16.flatten(start_dim=0, end_dim=1) 
        memory_value = memory_value.flatten(start_dim=0, end_dim=1) 
        
        weighted_r = self.attention_layer1(query=memory_value, reference=flow_feat_16)
        weighted_l = self.attention_layer2(query=flow_feat_16, reference=memory_value)
        
        left_attended_features = rearrange(
            [memory_value, weighted_l], "two b c h w -> b (two c) h w"
        )                               
        right_attended_features = rearrange(
            [flow_feat_16, weighted_r], "two b c h w -> b (two c) h w"
        )
        out_value = torch.cat([left_attended_features, right_attended_features], dim=1)
        out_value = self.dimension_reduction(out_value)
        return out_value.view(batch_size, num_objects, *out_value.shape[1:])


class CrossAttentionLayer(nn.Module):
    def __init__(self, query_dim, refer_dim, hidden_channels=256):
        super().__init__()
        self.query_dimensionality_reduction = nn.Conv2d(
            query_dim, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        
        self.reference_dimensionality_reduction = nn.Conv2d(
            refer_dim, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
    
    
    
    
    def forward(self, query, reference):
        Q = self.query_dimensionality_reduction(query) 
        K = self.reference_dimensionality_reduction(reference) 
        V = rearrange(reference, "b c h w -> b c (h w)") 
        attention_map = torch.einsum("bcij,bckl->bijkl", Q, K)
        attention_map = rearrange(attention_map, "b h1 w1 h2 w2 -> b h1 w1 (h2 w2)")
        attention_map = nn.Softmax(dim=3)(attention_map)
        attended_features = torch.einsum("bijp,bcp->bcij", attention_map, V)
        
        return attended_features 
class HiddenUpdater(nn.Module):
    
    def __init__(self, g_dims, mid_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.g16_conv = GConv2D(g_dims[0], mid_dim, kernel_size=1)
        self.g8_conv = GConv2D(g_dims[1], mid_dim, kernel_size=1)
        self.g4_conv = GConv2D(g_dims[2], mid_dim, kernel_size=1)

        self.transform = GConv2D(mid_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = self.g16_conv(g[0]) + self.g8_conv(downsample_groups(g[1], ratio=1/2)) + \
            self.g4_conv(downsample_groups(g[2], ratio=1/4))

        g = torch.cat([g, h], 2)

        
        
        
        
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class HiddenReinforcer(nn.Module):
    
    def __init__(self, g_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.transform = GConv2D(g_dim+hidden_dim, hidden_dim*3, kernel_size=3, padding=1)

        nn.init.xavier_normal_(self.transform.weight)

    def forward(self, g, h):
        g = torch.cat([g, h], 2)

        
        
        
        
        values = self.transform(g)
        forget_gate = torch.sigmoid(values[:,:,:self.hidden_dim])
        update_gate = torch.sigmoid(values[:,:,self.hidden_dim:self.hidden_dim*2])
        new_value = torch.tanh(values[:,:,self.hidden_dim*2:])
        new_h = forget_gate*h*(1-update_gate) + update_gate*new_value

        return new_h


class ValueEncoder(nn.Module):
    def __init__(self, value_dim, hidden_dim, single_object=False):
        super().__init__()
        
        self.single_object = single_object
        network = resnet.resnet18(pretrained=True, extra_dim=1 if single_object else 2)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  
        self.maxpool = network.maxpool

        self.layer1 = network.layer1 
        self.layer2 = network.layer2 
        self.layer3 = network.layer3 

        self.distributor = MainToGroupDistributor()
        self.fuser = FeatureFusionBlock(1024, 256, value_dim, value_dim)
        if hidden_dim > 0:
            self.hidden_reinforce = HiddenReinforcer(value_dim, hidden_dim)
        else:
            self.hidden_reinforce = None

    def forward(self, image, image_feat_f16, h, masks, others, is_deep_update=True):
        
        if not self.single_object:
            
            
            g = torch.stack([masks, others], 2)
        else:
            g = masks.unsqueeze(2)
        
        
        g = self.distributor(image, g)

        batch_size, num_objects = g.shape[:2]
        
        g = g.flatten(start_dim=0, end_dim=1)

        g = self.conv1(g)
        g = self.bn1(g) 
        g = self.maxpool(g)  
        g = self.relu(g) 

        g = self.layer1(g) 
        g = self.layer2(g) 
        g = self.layer3(g) 

        g = g.view(batch_size, num_objects, *g.shape[1:])
        
        
        
        g = self.fuser(image_feat_f16, g)
        
        
        if is_deep_update and self.hidden_reinforce is not None:
            
            h = self.hidden_reinforce(g, h)

        return g, h

class KeyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        network = resnet.resnet50(pretrained=True)
        self.conv1 = network.conv1
        self.bn1 = network.bn1
        self.relu = network.relu  
        self.maxpool = network.maxpool

        self.res2 = network.layer1 
        self.layer2 = network.layer2 
        self.layer3 = network.layer3 

    def forward(self, f):
        x = self.conv1(f) 
        x = self.bn1(x)
        x = self.relu(x)   
        x = self.maxpool(x)  
        f4 = self.res2(x)   
        f8 = self.layer2(f4) 
        f16 = self.layer3(f8) 

        return f16, f8, f4

class FlowEncoder(nn.Module):
    def __init__(self, ):
        super(FlowEncoder, self).__init__()
        network = resnet.resnet18(pretrained=True)
        
        self.conv1 = nn.Conv2d(10, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = network.bn1
        self.relu = network.relu  
        self.maxpool = network.maxpool

        self.layer1 = network.layer1 
        self.layer2 = network.layer2 
        self.layer3 = network.layer3 
    
    
    def forward(self, flow):
        if len(flow.shape) == 5:
            batch_size, num_frames = flow.shape[:2]
            
            flow = flow.flatten(start_dim=0, end_dim=1) 
            need_reshape = True
        else:
            need_reshape = False 
        flow = self.conv1(flow)
        flow = self.bn1(flow)
        flow = self.relu(flow) 
        flow = self.maxpool(flow)  

        flow_4 = self.layer1(flow) 
        flow_8 = self.layer2(flow_4) 
        flow_16 = self.layer3(flow_8) 
        if need_reshape:
            flow_16 = flow_16.view(batch_size, num_frames, *flow_16.shape[1:])
        
        return flow_16

class HandEncoder(nn.Module):
    def __init__(self, ):
        super(HandEncoder, self).__init__()
        network = resnet.resnet18(pretrained=True)
        
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = network.bn1
        self.relu = network.relu  
        self.maxpool = network.maxpool

        self.layer1 = network.layer1 
        self.layer2 = network.layer2 
        self.layer3 = network.layer3 
        self.final_conv = nn.Conv2d(256, 1, kernel_size=1, bias=False) 
    
    
    
    def forward(self, hand_msk):
        hand_msk = hand_msk.to(torch.float32)
        if len(hand_msk.shape) == 5:
            batch_size, num_frames = hand_msk.shape[:2]
            
            hand_msk = hand_msk.flatten(start_dim=0, end_dim=1) 
            need_reshape = True
        else:
            need_reshape = False 
        hand_msk = self.conv1(hand_msk)
        hand_msk = self.bn1(hand_msk)
        hand_msk = self.relu(hand_msk) 
        hand_msk = self.maxpool(hand_msk)  

        hand_msk_4 = self.layer1(hand_msk) 
        hand_msk_8 = self.layer2(hand_msk_4) 
        hand_msk_16 = self.layer3(hand_msk_8) 
        hand_msk_16 = self.final_conv(hand_msk_16)
        
        if need_reshape:
            hand_msk_16 = hand_msk_16.view(batch_size, num_frames, *hand_msk_16.shape[1:])
        
        return hand_msk_16

class UpsampleBlock(nn.Module):
    def __init__(self, skip_dim, g_up_dim, g_out_dim, scale_factor=2):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_dim, g_up_dim, kernel_size=3, padding=1)
        self.distributor = MainToGroupDistributor(method='add')
        self.out_conv = GroupResBlock(g_up_dim, g_out_dim)
        self.scale_factor = float(scale_factor)

    def forward(self, skip_f, up_g):
        skip_f = self.skip_conv(skip_f) 
        g = upsample_groups(up_g, ratio=self.scale_factor) 
        g = self.distributor(skip_f, g) 
        g = self.out_conv(g) 
        return g


class KeyProjection(nn.Module):
    def __init__(self, in_dim, keydim):
        super().__init__()

        self.key_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)
        
        self.d_proj = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)
        
        self.e_proj = nn.Conv2d(in_dim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
    
    def forward(self, x, need_s, need_e):
        shrinkage = self.d_proj(x)**2 + 1 if (need_s) else None
        selection = torch.sigmoid(self.e_proj(x)) if (need_e) else None

        return self.key_proj(x), shrinkage, selection


class Decoder(nn.Module):
    def __init__(self, val_dim, hidden_dim):
        super().__init__()

        self.fuser = FeatureFusionBlock(1024, val_dim+hidden_dim, 512, 512)
        if hidden_dim > 0:
            self.hidden_update = HiddenUpdater([512, 256, 256+1], 256, hidden_dim)
        else:
            self.hidden_update = None
        
        self.up_16_8 = UpsampleBlock(512, 512, 256) 
        self.up_8_4 = UpsampleBlock(256, 256, 256) 

        self.pred = nn.Conv2d(256, 1, kernel_size=3, padding=1, stride=1)

    def forward(self, f16, f8, f4, hidden_state, memory_readout, h_out=True):
        batch_size, num_objects = memory_readout.shape[:2] 

        if self.hidden_update is not None:
            g16 = self.fuser(f16, torch.cat([memory_readout, hidden_state], 2))
        else:
            g16 = self.fuser(f16, memory_readout)

        g8 = self.up_16_8(f8, g16) 
        g4 = self.up_8_4(f4, g8) 
        logits = self.pred(F.relu(g4.flatten(start_dim=0, end_dim=1))) 

        if h_out and self.hidden_update is not None:
            g4 = torch.cat([g4, logits.view(batch_size, num_objects, 1, *logits.shape[-2:])], 2)
            hidden_state = self.hidden_update([g16, g8, g4], hidden_state)
        else:
            hidden_state = None
        
        logits = F.interpolate(logits, scale_factor=4, mode='bilinear', align_corners=False)
        logits = logits.view(batch_size, num_objects, *logits.shape[-2:])
        
        
        return hidden_state, logits

class RandomWalkHead(nn.Module):
    def __init__(self, key_dim, downsample_mode='none', use_head=1, dropout_rate=0.0, temperature=0.07, **kwargs):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.temperature = temperature
        self._xent_targets = dict()
        self.xent = nn.CrossEntropyLoss(reduction="none")
        self.key_dim = key_dim
        self.downsample_mode = downsample_mode
        self.use_head = use_head
        if downsample_mode == 'conv':
            self.downsample_head = nn.Sequential(*[nn.Conv2d(self.key_dim, self.key_dim,\
                            kernel_size=1, stride=2, bias=False),
                                nn.BatchNorm2d(self.key_dim)])
        elif downsample_mode == 'pooling':
            self.downsample_head = nn.AvgPool2d(kernel_size=4, stride=kwargs['pooling_stride'], padding=2)
        else:
            assert downsample_mode == 'none'
        if use_head:
            self.linear_head = nn.Linear(self.key_dim, 128)
        
    def cal_affinity(self, x1, x2):
        in_t_dim = x1.ndim
        if in_t_dim < 4:  
            x1, x2 = x1.unsqueeze(-2), x2.unsqueeze(-2)

        A = torch.einsum('bctn,bctm->btnm', x1, x2)

        return A.squeeze(1) if in_t_dim < 4 else A

    def stoch_mat(self, A, do_dropout=True):
        ''' Affinity -> Stochastic Matrix '''

        if do_dropout and self.dropout_rate > 0:
            
            
            
            
            A[torch.rand_like(A) < self.dropout_rate] = torch.tensor(-1e20).to(torch.float16)
            
        return F.softmax(A/self.temperature, dim=-1)

    def xent_targets(self, A):
        B, N = A.shape[:2]
        key = '%s:%sx%s' % (str(A.device), B,N)

        if key not in self._xent_targets:
            I = torch.arange(A.shape[-1])[None].repeat(B, 1)
            self._xent_targets[key] = I.view(-1).to(A.device)

        return self._xent_targets[key]
    
    def forward(self, x):
        
        assert len(x.shape) == 5
        x = x.transpose(1, 2) 
        b, t = x.shape[:2]
        
        
        x = x.flatten(start_dim=0, end_dim=1)
        
        if self.downsample_mode == 'conv' or self.downsample_mode == 'pooling':
            x = self.downsample_head(x) 
        if self.use_head:
            x = self.linear_head(x.permute(0,2,3,1)).permute(0,3,1,2)  
        
        
        keys = x.view(b, t, *x.shape[-3:]).transpose(1, 2) 
        
        keys = keys.view(keys.shape[0],keys.shape[1],keys.shape[2],-1) 
        keys = F.normalize(keys, p=2, dim=1)
        B, C, T, N = keys.shape
        walks = dict()
        As = self.cal_affinity(keys[:, :, :-1], keys[:, :, 1:])
        A12s = [self.stoch_mat(As[:, i], do_dropout=True) for i in range(T-1)]
        
        A21s = [self.stoch_mat(As[:, i].transpose(-1, -2), do_dropout=True) for i in range(T-1)]
        AAs = []
        for i in list(range(1, len(A12s))): 
            g = A12s[:i+1] + A21s[:i+1][::-1]
            aar = g[0]
            for _a in g[1:]:
                aar = aar @ _a
            AAs.append((f"r{i}", aar))

        for i, aa in AAs:
            walks[f"cyc {i}"] = [aa, self.xent_targets(aa)]
        
        
        
        xents = 0
        diags = dict()

        for name, (A, target) in walks.items():
            logits = torch.log(A+EPS).flatten(0, -2)
            loss = self.xent(logits, target).mean()
            acc = (torch.argmax(logits, dim=-1) == target).float().mean()
            diags.update({f"rand_walk/xent_{name}": loss.detach(),
                            f"rand_walk/acc_{name}": acc})
            xents += loss
        
        return loss, diags

if __name__ == '__main__':
    dummy_mv = torch.randn(1, 5, 3, 16, 16)
    dummy_flow = torch.randn(1, 2, 16, 16)
    cross_atten = CrossAttentionValueFuser(3, 2, 8)
    out_value = cross_atten(dummy_mv, dummy_flow)
    print(out_value.shape)