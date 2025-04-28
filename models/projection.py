import torch
from ..model import init_weight
from .vision_transformer import Aggregation_Block 
import torch.nn as nn
from functools import partial
import torch.nn.functional as F

def modify_grad_v2(x, factor):
    factor = factor.expand_as(x)
    x *= factor
    return x

class Projection(torch.nn.Module):
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc", torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                if layer_type > 1:
                    self.layers.add_module(f"{i}act", torch.nn.SiLU())
        self.apply(init_weight)

    def forward(self, x):

        x = self.layers(x)
        return x



class ProjectionV2(torch.nn.Module):
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0,target_embed_dimension=512):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes+16 if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc", torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                if layer_type > 1:
                    self.layers.add_module(f"{i}act", torch.nn.SiLU())
        self.apply(init_weight)

        num_heads = 12
        inp_num = 6
        embed_dim = target_embed_dimension
        aggregation = []
        for i in range(1):
            blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            aggregation.append(blk)
        self.aggregation = nn.ModuleList(aggregation)
        self.prototype_token = nn.ParameterList([nn.Parameter(torch.randn(inp_num, embed_dim))])

    def forward(self, x,shape=None,mask=None,return_loss=True):
        #x:[B*H*W,C]

        B,C,H,W = shape
        x0 = torch.reshape(x0,[B,H,W,x.shape[-1]])
        x0 = torch.transpose(x0,[0,3,1,2])
        x0 = F.interpolate(x0,size=(H//4,W//4),mode="bilinear")
        x0 = torch.transpose(x0,[0,2,3,1])
        x0 = torch.reshape(x0,[B,-1,C])


        en = x0
        agg_prototype = self.prototype_token[0]
        for i, blk in enumerate(self.aggregation):
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), x)
        if self.training and return_loss:
            g_loss = self.gather_loss(x0, agg_prototype)
        else:
            g_loss = None

        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, agg_prototype)
            de_list.append(x)
        de_list = de_list[::-1]

        de = self.fuse_feature(de_list)


        en = en.permute(0, 2, 1).reshape([x.shape[0], -1, H, W]).contiguous()
        de = de.permute(0, 2, 1).reshape([x.shape[0], -1, H, W]).contiguous()
        #return en, de, g_loss
        if self.training and return_loss:
            sm_loss = self.global_cosine_hm_adaptive([en],[de])
        else:
            sm_loss = None

        map = F.cosine_similarity(en, de)
        map = F.interpolate(map,size=(H,W),mode="bilinear")
        map = torch.reshape(map,[-1,1])
        map = torch.tile(map,[1,16])
        x = torch.cat([x,map],dim=-1)
        x = self.layers(x)

        res = dict(x=x,g_loss=g_loss*0.2,lsm=sm_loss)
        return res


    def gather_loss(self, query, keys,mask=None):
        self.distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        self.distance, self.cluster_index = torch.min(self.distribution, dim=2)
        gather_loss = self.distance.mean()
        return gather_loss

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

    @staticmethod
    def global_cosine_hm_adaptive(a, b, y=3):
        cos_loss = torch.nn.CosineSimilarity()
        loss = 0
        for item in range(len(a)):
            a_ = a[item].detach()
            b_ = b[item]
            with torch.no_grad():
                point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach()
            mean_dist = point_dist.mean()
            # std_dist = point_dist.reshape(-1).std()
            # thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
            factor = (point_dist/mean_dist)**(y)
            # factor = factor/torch.max(factor)
            # factor = torch.clip(factor, min=min_grad)
            # print(thresh)
            loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                            b_.reshape(b_.shape[0], -1)))
            partial_func = partial(modify_grad_v2, factor=factor)
            b_.register_hook(partial_func)
    
        loss = loss / len(a)
        return loss

