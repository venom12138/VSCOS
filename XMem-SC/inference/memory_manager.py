import torch
import warnings

from inference.kv_memory_store import KeyValueMemoryStore
from model.memory_util import *


class MemoryManager:
    """
    Manages all three memory stores and the transition between working/long-term memory
    """
    def __init__(self, config):
        self.hidden_dim = config['hidden_dim']
        self.top_k = config['top_k']

        self.enable_long_term = config['enable_long_term']
        self.enable_long_term_usage = config['enable_long_term_count_usage']
        if self.enable_long_term:
            self.max_mt_frames = config['max_mid_term_frames']
            self.min_mt_frames = config['min_mid_term_frames']
            self.num_prototypes = config['num_prototypes']
            self.max_long_elements = config['max_long_term_elements']

        
        self.CK = self.CV = None
        self.H = self.W = None

        
        
        self.hidden = None

        self.work_mem = KeyValueMemoryStore(count_usage=self.enable_long_term)
        if self.enable_long_term:
            self.long_mem = KeyValueMemoryStore(count_usage=self.enable_long_term_usage)

        self.reset_config = True

    def update_config(self, config):
        self.reset_config = True
        self.hidden_dim = config['hidden_dim']
        self.top_k = config['top_k']

        assert self.enable_long_term == config['enable_long_term'], 'cannot update this'
        assert self.enable_long_term_usage == config['enable_long_term_count_usage'], 'cannot update this'

        self.enable_long_term_usage = config['enable_long_term_count_usage']
        if self.enable_long_term:
            self.max_mt_frames = config['max_mid_term_frames']
            self.min_mt_frames = config['min_mid_term_frames']
            self.num_prototypes = config['num_prototypes']
            self.max_long_elements = config['max_long_term_elements']

    def _readout(self, affinity, v):
        
        return v @ affinity

    def match_memory(self, query_key, selection):
        
        
        num_groups = self.work_mem.num_groups
        h, w = query_key.shape[-2:]

        query_key = query_key.flatten(start_dim=2)
        selection = selection.flatten(start_dim=2) if selection is not None else None

        """
        Memory readout using keys
        """

        if self.enable_long_term and self.long_mem.engaged():
            
            long_mem_size = self.long_mem.size
            memory_key = torch.cat([self.long_mem.key, self.work_mem.key], -1)
            shrinkage = torch.cat([self.long_mem.shrinkage, self.work_mem.shrinkage], -1) 

            similarity = get_similarity(memory_key, shrinkage, query_key, selection)
            work_mem_similarity = similarity[:, long_mem_size:]
            long_mem_similarity = similarity[:, :long_mem_size]

            
            
            affinity, usage = do_softmax(
                    torch.cat([long_mem_similarity[:, -self.long_mem.get_v_size(0):], work_mem_similarity], 1), 
                    top_k=self.top_k, inplace=True, return_usage=True)
            affinity = [affinity]

            
            for gi in range(1, num_groups):
                if gi < self.long_mem.num_groups:
                    
                    affinity_one_group = do_softmax(
                        torch.cat([long_mem_similarity[:, -self.long_mem.get_v_size(gi):], 
                                    work_mem_similarity[:, -self.work_mem.get_v_size(gi):]], 1), 
                        top_k=self.top_k, inplace=True)
                else:
                    
                    affinity_one_group = do_softmax(work_mem_similarity[:, -self.work_mem.get_v_size(gi):], 
                        top_k=self.top_k, inplace=(gi==num_groups-1))
                affinity.append(affinity_one_group)

            all_memory_value = []
            for gi, gv in enumerate(self.work_mem.value):
                
                if gi < self.long_mem.num_groups:
                    all_memory_value.append(torch.cat([self.long_mem.value[gi], self.work_mem.value[gi]], -1))
                else:
                    all_memory_value.append(gv)

            """
            Record memory usage for working and long-term memory
            """
            
            work_usage = usage[:, long_mem_size:]
            self.work_mem.update_usage(work_usage.flatten())

            if self.enable_long_term_usage:
                
                long_usage = usage[:, :long_mem_size]
                self.long_mem.update_usage(long_usage.flatten())
        else:
            
            similarity = get_similarity(self.work_mem.key, self.work_mem.shrinkage, query_key, selection)

            if self.enable_long_term:
                affinity, usage = do_softmax(similarity, inplace=(num_groups==1), 
                    top_k=self.top_k, return_usage=True)

                
                self.work_mem.update_usage(usage.flatten())
            else:
                affinity = do_softmax(similarity, inplace=(num_groups==1), 
                    top_k=self.top_k, return_usage=False)

            affinity = [affinity]

            
            for gi in range(1, num_groups):
                affinity_one_group = do_softmax(similarity[:, -self.work_mem.get_v_size(gi):], 
                    top_k=self.top_k, inplace=(gi==num_groups-1))
                affinity.append(affinity_one_group)
                
            all_memory_value = self.work_mem.value

        
        all_readout_mem = torch.cat([
            self._readout(affinity[gi], gv)
            for gi, gv in enumerate(all_memory_value)
        ], 0)

        return all_readout_mem.view(all_readout_mem.shape[0], self.CV, h, w)

    def add_memory(self, key, shrinkage, value, objects, selection=None):
        
        
        
        if self.H is None or self.reset_config:
            self.reset_config = False
            self.H, self.W = key.shape[-2:]
            self.HW = self.H*self.W
            if self.enable_long_term:
                
                self.min_work_elements = self.min_mt_frames*self.HW
                self.max_work_elements = self.max_mt_frames*self.HW

        
        
        key = key.flatten(start_dim=2)
        shrinkage = shrinkage.flatten(start_dim=2) 
        value = value[0].flatten(start_dim=2)

        self.CK = key.shape[1]
        self.CV = value.shape[1]

        if selection is not None:
            if not self.enable_long_term:
                warnings.warn('the selection factor is only needed in long-term mode', UserWarning)
            selection = selection.flatten(start_dim=2)

        self.work_mem.add(key, value, shrinkage, selection, objects)

        
        if self.enable_long_term:
            
            if self.work_mem.size >= self.max_work_elements:
                
                if self.long_mem.size >= (self.max_long_elements-self.num_prototypes):
                    self.long_mem.remove_obsolete_features(self.max_long_elements-self.num_prototypes)
                    
                self.compress_features()


    def create_hidden_state(self, n, sample_key):
        
        h, w = sample_key.shape[-2:]
        if self.hidden is None:
            self.hidden = torch.zeros((1, n, self.hidden_dim, h, w), device=sample_key.device)
        elif self.hidden.shape[1] != n:
            self.hidden = torch.cat([
                self.hidden, 
                torch.zeros((1, n-self.hidden.shape[1], self.hidden_dim, h, w), device=sample_key.device)
            ], 1)

        assert(self.hidden.shape[1] == n)

    def set_hidden(self, hidden):
        self.hidden = hidden

    def get_hidden(self):
        return self.hidden

    def compress_features(self):
        HW = self.HW
        candidate_value = []
        total_work_mem_size = self.work_mem.size
        for gv in self.work_mem.value:
            
            
            
            mem_size_in_this_group = gv.shape[-1]
            if mem_size_in_this_group == total_work_mem_size:
                
                candidate_value.append(gv[:,:,HW:-self.min_work_elements+HW])
            else:
                
                assert HW <= mem_size_in_this_group < total_work_mem_size
                if mem_size_in_this_group > self.min_work_elements+HW:
                    
                    candidate_value.append(gv[:,:,HW:-self.min_work_elements+HW])
                else:
                    
                    candidate_value.append(None)

        
        prototype_key, prototype_value, prototype_shrinkage = self.consolidation(
            *self.work_mem.get_all_sliced(HW, -self.min_work_elements+HW), candidate_value)

        
        self.work_mem.sieve_by_range(HW, -self.min_work_elements+HW, min_size=self.min_work_elements+HW)

        
        self.long_mem.add(prototype_key, prototype_value, prototype_shrinkage, selection=None, objects=None)

    def consolidation(self, candidate_key, candidate_shrinkage, candidate_selection, usage, candidate_value):
        
        
        N = candidate_key.shape[-1]

        
        _, max_usage_indices = torch.topk(usage, k=self.num_prototypes, dim=-1, sorted=True)
        prototype_indices = max_usage_indices.flatten()

        
        validity = [prototype_indices >= (N-gv.shape[2]) if gv is not None else None for gv in candidate_value]

        prototype_key = candidate_key[:, :, prototype_indices]
        prototype_selection = candidate_selection[:, :, prototype_indices] if candidate_selection is not None else None

        """
        Potentiation step
        """
        similarity = get_similarity(candidate_key, candidate_shrinkage, prototype_key, prototype_selection)

        
        
        affinity = [
            do_softmax(similarity[:, -gv.shape[2]:, validity[gi]]) if gv is not None else None
            for gi, gv in enumerate(candidate_value)
        ]

        
        affinity = [
            aff if aff is None or aff.shape[-1] > 0 else None for aff in affinity
        ]

        
        prototype_value = [
            self._readout(affinity[gi], gv) if affinity[gi] is not None else None
            for gi, gv in enumerate(candidate_value)
        ]

        
        prototype_shrinkage = self._readout(affinity[0], candidate_shrinkage) if candidate_shrinkage is not None else None

        return prototype_key, prototype_value, prototype_shrinkage
