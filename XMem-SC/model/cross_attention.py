import torch
import torch.nn as nn
from einops import rearrange

class CrossAttentionValueFuser(nn.Module):
    def __init__(self, x_in_dim, f_in_dim, out_dim, hidden_channels=256, attention_type="coam"):
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

if __name__ == '__main__':
    dummy_mv = torch.randn(1, 5, 3, 16, 16)
    dummy_flow = torch.randn(1, 2, 16, 16)
    cross_atten = CrossAttentionValueFuser(3, 2, 8)
    out_value = cross_atten(dummy_mv, dummy_flow)
    print(out_value.shape)