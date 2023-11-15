from inference.memory_manager import MemoryManager
from model.network import XMem
from model.aggregate import aggregate

from util.tensor_util import pad_divide_by, unpad


class InferenceCore:
    def __init__(self, network:XMem, config):
        self.config = config
        self.network = network
        self.mem_every = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']

        
        self.deep_update_sync = (self.deep_update_every < 0)

        self.clear_memory()
        self.all_labels = None

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        if not self.deep_update_sync:
            self.last_deep_update_ti = -self.deep_update_every
        self.memory = MemoryManager(config=self.config)

    def update_config(self, config):
        self.mem_every = config['mem_every']
        self.deep_update_every = config['deep_update_every']
        self.enable_long_term = config['enable_long_term']

        
        self.deep_update_sync = (self.deep_update_every < 0)
        self.memory.update_config(config)

    def set_all_labels(self, all_labels):
        
        self.all_labels = all_labels

    def step(self, image, flow=None, text=None, hand_mask=None, mask=None, valid_labels=None, end=False):
        
        
        
        
        self.curr_ti += 1
        image, self.pad = pad_divide_by(image, 16)
        if flow != None:
            flow, _ = pad_divide_by(flow, 16)
            flow = flow.unsqueeze(0) 
        image = image.unsqueeze(0) 
        if hand_mask is not None:
            hand_mask, _ = pad_divide_by(hand_mask, 16)
            hand_mask = hand_mask.unsqueeze(0) 
        
        

        is_mem_frame = ((self.curr_ti-self.last_mem_ti >= self.mem_every) or (mask is not None)) and (not end)
        need_segment = (self.curr_ti > 0) and ((valid_labels is None) or (len(self.all_labels) != len(valid_labels)))
        is_deep_update = (
            (self.deep_update_sync and is_mem_frame) or  
            (not self.deep_update_sync and self.curr_ti-self.last_deep_update_ti >= self.deep_update_every) 
        ) and (not end)
        is_normal_update = (not self.deep_update_sync or not is_deep_update) and (not end)

        key, shrinkage, selection, f16, f8, f4 = self.network.encode_key(image, 
                                                    need_ek=(self.enable_long_term or need_segment), 
                                                    need_sk=is_mem_frame)
        multi_scale_features = (f16, f8, f4)
        if flow != None:
            flow_feat = self.network.encode_flow(flow)
        else:
            flow_feat = None
        
        if hand_mask is not None:
            hand_mask = self.network.hand_encoder(hand_mask)
        else:
            hand_mask = None
            
        
        if need_segment:
            memory_readout = self.memory.match_memory(key, selection).unsqueeze(0)
            if flow != None or text != None or hand_mask != None:
                memory_readout = self.network.fuse_value(mv=memory_readout, flow_feat=flow_feat, text_feat=text, hand_feat=hand_mask)
            
            hidden, _, pred_prob_with_bg = self.network.segment(multi_scale_features, memory_readout, 
                                    self.memory.get_hidden(), h_out=is_normal_update, strip_bg=False)
            
            
            
            pred_prob_with_bg = pred_prob_with_bg[0]
            pred_prob_no_bg = pred_prob_with_bg[1:]
            if is_normal_update:
                self.memory.set_hidden(hidden)
        else:
            pred_prob_no_bg = pred_prob_with_bg = None

        
        if mask is not None:
            mask, _ = pad_divide_by(mask, 16)

            if pred_prob_no_bg is not None:
                
                
                mask_regions = (mask.sum(0) > 0.5)
                pred_prob_no_bg[:, mask_regions] = 0
                
                mask = mask.type_as(pred_prob_no_bg)
                if valid_labels is not None:
                    shift_by_one_non_labels = [i for i in range(pred_prob_no_bg.shape[0]) if (i+1) not in valid_labels]
                    
                    mask[shift_by_one_non_labels] = pred_prob_no_bg[shift_by_one_non_labels]
            pred_prob_with_bg = aggregate(mask, dim=0)

            
            self.memory.create_hidden_state(len(self.all_labels), key)

        
        if is_mem_frame:
            value, hidden = self.network.encode_value(image, f16, self.memory.get_hidden(), 
                                    pred_prob_with_bg[1:].unsqueeze(0), is_deep_update=is_deep_update)
            self.memory.add_memory(key, shrinkage, value, self.all_labels, 
                                    selection=selection if self.enable_long_term else None)
            self.last_mem_ti = self.curr_ti

            if is_deep_update:
                self.memory.set_hidden(hidden)
                self.last_deep_update_ti = self.curr_ti
                
        return unpad(pred_prob_with_bg, self.pad)
