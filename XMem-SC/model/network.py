"""
This file defines XMem, the highest level nn.Module interface
During training, it is used by trainer.py
During evaluation, it is used by inference_core.py

It further depends on modules.py which gives more detailed implementations of sub-modules
"""

import torch
import torch.nn as nn
from copy import deepcopy
from model.aggregate import aggregate
from model.modules import *
from model.memory_util import *



from torchsummary import summary
import clip
    
class XMem(nn.Module):
    def __init__(self, config, model_path=None, map_location=None):
        """
        model_path/map_location are used in evaluation only
        map_location is for converting models saved in cuda to cpu
        """
        super().__init__()
        model_weights = self.init_hyperparameters(config, model_path, map_location)
        self.config = config
        self.single_object = config.get('single_object', False)
        print(f'Single object mode: {self.single_object}')

        self.key_encoder = KeyEncoder() 
        self.value_encoder = ValueEncoder(self.value_dim, self.hidden_dim, self.single_object)            
        
        if config['use_text']:
            clip_model,_ = clip.load("ViT-L/14@336px") 
            self.clip_text_encoder = nn.Module()
            self.clip_text_encoder.token_embedding = clip_model.token_embedding
            self.clip_text_encoder.positional_embedding = clip_model.positional_embedding
            self.clip_text_encoder.transformer = clip_model.transformer
            self.clip_text_encoder.ln_final= clip_model.ln_final
            self.clip_text_encoder.text_projection = clip_model.text_projection
            del clip_model
            self.text_dim = 256
            self.text_proj = nn.Linear(self.clip_text_encoder.text_projection.shape[1], self.text_dim)
        
        self.use_text = config['use_text']
        self.use_flow = config['use_flow']
        self.use_handmsk = config['use_handmsk']
        
        t_in_dim = 0
        f_in_dim = 0
        h_in_dim = 0
        
        if self.use_text:
            t_in_dim = 256
            assert config['fuser_type'] == 'cbam'
        if self.use_flow:
            self.flow_encoder = FlowEncoder()
            f_in_dim = 256
        if self.use_handmsk:
            self.hand_encoder = HandEncoder()
            h_in_dim = 1
            assert config['fuser_type'] == 'cbam'
        if self.use_text or self.use_flow or self.use_handmsk:
            if config['fuser_type'] == 'cbam':
                self.value_fuser = ValueFuser(x_in_dim=self.value_dim, f_in_dim=f_in_dim, t_in_dim=t_in_dim, h_in_dim=h_in_dim, out_dim=self.value_dim)
            elif config['fuser_type'] == 'cross_attention':
                self.value_fuser = CrossAttentionValueFuser(x_in_dim=self.value_dim, f_in_dim=f_in_dim, out_dim=self.value_dim)
        
        
        
        self.key_proj = KeyProjection(1024, self.key_dim)

        self.decoder = Decoder(self.value_dim, self.hidden_dim)

        if model_weights is not None:
            self.load_weights(model_weights, init_as_zero_if_needed=True)
    
    @property
    def dtype(self):
        return self.key_encoder.conv1.weight.dtype
    
    
    def encode_text(self, text):
        x = self.clip_text_encoder.token_embedding(text).type(torch.float16)  

        x = x + self.clip_text_encoder.positional_embedding.type(torch.float16)
        x = x.permute(1, 0, 2)  
        x = self.clip_text_encoder.transformer(x)
        x = x.permute(1, 0, 2)  
        x = self.clip_text_encoder.ln_final(x).type(torch.float16)

        
        
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip_text_encoder.text_projection
        x = self.text_proj(x.type(self.dtype)) 
        return x
    
    
    
    
    
    def fuse_value(self, mv=None, flow_feat=None, text_feat=None, hand_feat=None):
        assert mv != None
        if self.use_text:
            assert text_feat != None
        
        if self.use_flow:
            assert flow_feat != None
        
        if self.use_handmsk:
            assert hand_feat != None

        return self.value_fuser(memory_value=mv, flow_feat_16=flow_feat, text_feat=text_feat, hand_feat=hand_feat)
        
    def encode_key(self, frame, need_sk=True, need_ek=True): 
        
        
        
        if len(frame.shape) == 5:
            
            
            need_reshape = True
            b, t = frame.shape[:2]
            
            
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            
            need_reshape = False
        else:
            raise NotImplementedError
        
        f16, f8, f4 = self.key_encoder(frame)
        
        
        
        key, shrinkage, selection = self.key_proj(f16, need_sk, need_ek)
        
        if need_reshape:
            
            key = key.view(b, t, *key.shape[-3:]).transpose(1, 2).contiguous()
            if shrinkage is not None:
                shrinkage = shrinkage.view(b, t, *shrinkage.shape[-3:]).transpose(1, 2).contiguous()
            if selection is not None:
                selection = selection.view(b, t, *selection.shape[-3:]).transpose(1, 2).contiguous()

            
            f16 = f16.view(b, t, *f16.shape[-3:])
            f8 = f8.view(b, t, *f8.shape[-3:])
            f4 = f4.view(b, t, *f4.shape[-3:])

        return key, shrinkage, selection, f16, f8, f4

    def encode_value(self, frame, image_feat_f16, h16, masks, is_deep_update=True): 
        
        
        
        num_objects = masks.shape[1]
        if num_objects != 1:
            
            
            others = torch.cat([
                torch.sum(
                    masks[:, [j for j in range(num_objects) if i!=j]]
                , dim=1, keepdim=True)
            for i in range(num_objects)], 1)
        else:
            others = torch.zeros_like(masks)

        g16, h16 = self.value_encoder(frame, image_feat_f16, h16, masks, others, is_deep_update)

        return g16, h16
    
    
    
    def encode_flow(self, flow):
        return self.flow_encoder(flow)
    
    
    
    def read_memory(self, query_key, query_selection, memory_key, 
                    memory_shrinkage, memory_value):
        """
        query_key       : B * CK * H * W
        query_selection : B * CK * H * W
        memory_key      : B * CK * T * H * W
        memory_shrinkage: B * 1  * T * H * W
        memory_value    : B * num_objects * CV * T * H * W
        """
        
        
        batch_size, num_objects = memory_value.shape[:2]
        memory_value = memory_value.flatten(start_dim=1, end_dim=2) 
        
        
        affinity = get_affinity(memory_key, memory_shrinkage, query_key, query_selection)
        
        memory = readout(affinity, memory_value)
        
        memory = memory.view(batch_size, num_objects, self.value_dim, *memory.shape[-2:])

        return memory

    def segment(self, multi_scale_features, memory_readout,
                    hidden_state, selector=None, h_out=True, strip_bg=True): 
        
        
        hidden_state, logits = self.decoder(*multi_scale_features, hidden_state, memory_readout, h_out=h_out)
        prob = torch.sigmoid(logits) 
        if selector is not None:
            prob = prob * selector 
            
        logits, prob = aggregate(prob, dim=1, return_logits=True)
        if strip_bg:
            
            prob = prob[:, 1:]
        
        return hidden_state, logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'read_memory':
            return self.read_memory(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        elif mode == 'encode_flow':
            return self.encode_flow(*args, **kwargs)
        elif mode == 'fuse_value':
            return self.fuse_value(*args, **kwargs)
        else:
            raise NotImplementedError

    def init_hyperparameters(self, config, model_path=None, map_location=None):
        """
        Init three hyperparameters: key_dim, value_dim, and hidden_dim
        If model_path is provided, we load these from the model weights
        The actual parameters are then updated to the config in-place

        Otherwise we load it either from the config or default
        """
        if model_path is not None:
            
            
            model_weights = torch.load(model_path, map_location=map_location)
            self.key_dim = model_weights['key_proj.key_proj.weight'].shape[0]
            self.value_dim = model_weights['value_encoder.fuser.block2.conv2.weight'].shape[0]
            self.disable_hidden = 'decoder.hidden_update.transform.weight' not in model_weights
            if self.disable_hidden:
                self.hidden_dim = 0
            else:
                self.hidden_dim = model_weights['decoder.hidden_update.transform.weight'].shape[0]//3
            print(f'Hyperparameters read from the model weights: '
                    f'C^k={self.key_dim}, C^v={self.value_dim}, C^h={self.hidden_dim}')
        else:
            model_weights = None
            
            if 'key_dim' not in config:
                self.key_dim = 64
                print(f'key_dim not found in config. Set to default {self.key_dim}')
            else:
                self.key_dim = config['key_dim']

            if 'value_dim' not in config:
                self.value_dim = 512
                print(f'value_dim not found in config. Set to default {self.value_dim}')
            else:
                self.value_dim = config['value_dim']

            if 'hidden_dim' not in config:
                self.hidden_dim = 64
                print(f'hidden_dim not found in config. Set to default {self.hidden_dim}')
            else:
                self.hidden_dim = config['hidden_dim']

            self.disable_hidden = (self.hidden_dim <= 0)

        config['key_dim'] = self.key_dim
        config['value_dim'] = self.value_dim
        config['hidden_dim'] = self.hidden_dim

        return model_weights

    def load_weights(self, src_dict, init_as_zero_if_needed=False):
        
        load_strict = False
        flow_check = False
        text_check = False
        for k in list(src_dict.keys()):
            if (('flow_encoder' in k) or ('value_fuser' in k)):
                flow_check = True
            if ('text_proj' in k):
                text_check = True
            if (('flow_encoder' in k) or ('value_fuser' in k)) and (not self.use_text) and self.use_flow:
                flow_check = True
                load_strict = True
            if ('text_proj' in k) and (not self.use_flow) and self.use_text:
                text_check = True
                load_strict = True
            if self.use_flow and self.use_text:
                load_strict = flow_check and text_check
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    print('Converting weights from single object to multiple objects.')
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    if not init_as_zero_if_needed:
                        print('Randomly initialized padding.')
                        nn.init.orthogonal_(pads)
                    else:
                        print('Zero-initialized padding.')
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)
        print(f'----------load_strict:{load_strict}--------')
        sd_before_load = deepcopy(self.state_dict())
        msg = self.load_state_dict(src_dict, strict=load_strict)
        
        sd_after_load = deepcopy(self.state_dict())
        same_keys = [k for k in sd_before_load if torch.equal(sd_before_load[k], sd_after_load[k])]
        new_keys = []
        for key in same_keys:
            
            
            if key.startswith('key_encoder') or key.startswith('value_encoder'):
                if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key: 
                    continue
            if key.startswith('clip_text_encoder'):
                continue
            
            new_keys.append(key)
        print('-------------------- Loaded weights --------------------')
        print(f'Weights unloaded:{new_keys}')
        print('----------------------------')
        
        if load_strict == False:
            if self.use_text:
                text_new_weights = len(self.text_proj.state_dict().keys())
            else:
                text_new_weights = 0
            
            if self.use_flow:
                flow_new_weights = len(self.flow_encoder.state_dict().keys())
            else:
                flow_new_weights = 0
                
            if self.use_text or self.use_flow or self.use_handmsk:
                value_fuser_new_weights = len(self.value_fuser.state_dict().keys())
            else:
                value_fuser_new_weights = 0
                
            
            
            
            
            
            if self.use_handmsk:
                hand_new_weights = len(self.hand_encoder.state_dict().keys())
            else:
                hand_new_weights = 0
            
            
            
            
            assert ((len(new_keys) == text_new_weights + flow_new_weights + value_fuser_new_weights \
                    + hand_new_weights ) \
                    or len(new_keys) == 0 )