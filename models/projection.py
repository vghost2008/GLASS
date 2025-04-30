import torch
from model import init_weight
from .vision_transformer import Aggregation_Block,Prototype_Block, Mlp
import wml.semantic.mask_utils as smu
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import wml.wtorch.train_toolkit as wtt

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
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0,target_embed_dimension=None):
        super().__init__()

        if target_embed_dimension is None:
            target_embed_dimension = in_planes

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        in_planes = 256
        for i in range(n_layers):
            _in = in_planes+64 if i == 0 else _out
            _out = out_planes
            self.layers.add_module(f"{i}fc", torch.nn.Linear(_in, _out))
            if i < n_layers - 1:
                if layer_type > 1:
                    self.layers.add_module(f"{i}act", torch.nn.SiLU())
        self.apply(init_weight)

        num_heads = target_embed_dimension//64
        num_heads = 8
        inp_num = 6
        embed_dim = target_embed_dimension
        aggregation = []
        for i in range(1):
            blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            aggregation.append(blk)
        self.aggregation = nn.ModuleList(aggregation)
        self.prototype_token = nn.ParameterList([nn.Parameter(torch.randn(inp_num, embed_dim))])

        bottleneck = []
        bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
        self.bottleneck = nn.ModuleList(bottleneck)

        trans_convs = []
        for i in range(4):
            trans_convs.append(nn.Conv2d(512,256,kernel_size=1))
        self.trans_convs = nn.ModuleList(trans_convs)

        decoder = []
        for i in range(4):
            blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                              qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            decoder.append(blk)
        self.decoder = nn.ModuleList(decoder)
        wtt.set_bn_momentum(self,0.01)


    @staticmethod
    def trans2lc(x,shape):
        B,_,H,W = shape
        if H%4== 0 and W%4==0:
            target_shape = (H//4,W//4)
        else:
            target_shape = (H*2//7,W*2//7)

        if len(x.shape) == 4:
            C = x.shape[1]
            x0 = x
        else:
            C = x.shape[-1]
            x0 = torch.reshape(x,[B,H,W,x.shape[-1]])
            x0 = torch.permute(x0,[0,3,1,2])
        x0 = F.interpolate(x0,size=target_shape,mode="bilinear")
        x0 = torch.permute(x0,[0,2,3,1])
        x0 = torch.reshape(x0,[B,-1,C])
        return x0, target_shape

    @staticmethod
    def trans2chw(x,shape):
        if isinstance(x,(list,tuple)):
            res = []
            ts = None
            for d in x:
                cx,ts = ProjectionV2.trans2chw(d,shape)
                res.append(cx)
            return res,ts
        B,_,H,W = shape
        if H%4== 0 and W%4==0:
            target_shape = (H//4,W//4)
        else:
            target_shape = (H*2//7,W*2//7)

        if len(x.shape) == 4:
            C = x.shape[1]
            x0 = x
        else:
            C = x.shape[-1]
            x0 = torch.reshape(x,[B,H,W,x.shape[-1]])
            x0 = torch.permute(x0,[0,3,1,2])
        x0 = F.interpolate(x0,size=target_shape,mode="bilinear")
        return x0, target_shape

    def forward(self, x,raw_x,shape=None,mask=None,return_loss=True):
        #x:[B*H*W,C]

        B,C,H,W = shape
        x,target_shape = self.trans2lc(x,shape)
        en,_ = self.trans2chw(raw_x,shape)
        if mask is not None:
            mask = smu.resize_mask(mask,[target_shape[1],target_shape[0]])
            mask = mask.to(torch.bool).to(x.device)
        agg_prototype = self.prototype_token[0]
        for i, blk in enumerate(self.aggregation):
            agg_prototype = blk(agg_prototype.unsqueeze(0).repeat((B, 1, 1)), x)
        if self.training and return_loss:
            g_loss = self.gather_loss(x, agg_prototype,mask=mask)
        else:
            g_loss = None

        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, agg_prototype)
            de_list.append(x)
        de_list = de_list[::-1]

        #de = self.fuse_feature(de_list)


        de_list = [cde.permute(0, 2, 1).reshape([x.shape[0], -1, target_shape[0], target_shape[1]]).contiguous() for cde in de_list]
        de_list = [m(cde) for m,cde in zip(self.trans_convs,de_list)]
        #return en, de, g_loss
        if self.training and return_loss:
            if mask is None:
                sm_loss = self.global_cosine_hm_adaptive(en,de_list)
            else:
                sm_loss = self.global_cosine_hm_adaptive_with_mask(en,de_list,mask)
        else:
            sm_loss = None

        map = [F.cosine_similarity(cen, cde) for cen,cde in zip(en,de_list)]
        map = torch.stack(map,dim=1)
        map = F.interpolate(map,size=(H,W),mode="bilinear")
        map = torch.permute(map,[0,2,3,1])
        map = torch.reshape(map,[-1,map.shape[-1]])
        map = torch.tile(map,[1,16])
        de = self.fuse_feature(de_list)
        de = F.interpolate(de,size=(H,W),mode="bilinear")
        de = torch.permute(de,[0,2,3,1])
        de = torch.reshape(de,[-1,de.shape[-1]])
        de = torch.cat([de,map],dim=-1)
        de = self.layers(de)

        if return_loss and self.training:
            res = dict(g_loss=g_loss*0.2,lsm=sm_loss*0.1)
            return de,res
        else:
            return de


    def gather_loss(self, query, keys,mask=None):
        self.distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        distance, cluster_index = torch.min(self.distribution, dim=2)
        if mask is not None:
            distance = torch.reshape(distance,[-1])
            mask = torch.logical_not(torch.reshape(mask,[-1]))
            distance = distance[mask]
        gather_loss = distance.mean()
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
            #loss += torch.mean(1 - cos_loss(a_,b_))
            partial_func = partial(modify_grad_v2, factor=factor)
            b_.register_hook(partial_func)
    
        loss = loss / len(a)
        return loss


    @staticmethod
    def global_cosine_hm_adaptive_with_mask(a, b, m,y=3):
        cos_loss = torch.nn.CosineSimilarity()
        loss = 0
        m_ = m
        m_ = m_.unsqueeze(1)
        m_ = m_.to(b[0].dtype)
        m_ = m_*2-1
        for item in range(len(a)):
            a_ = a[item].detach()
            b_ = b[item]
            with torch.no_grad():
                point_dist = (1 +m_*cos_loss(a_, b_).unsqueeze(1).detach())/2
            mean_dist = point_dist.mean()
            # std_dist = point_dist.reshape(-1).std()
            # thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
            factor = (point_dist/mean_dist)**(y)
            # factor = factor/torch.max(factor)
            # factor = torch.clip(factor, min=min_grad)
            # print(thresh)
            #loss += torch.mean(1 +m_*cos_loss(a_.reshape(a_.shape[0], -1),
                                            #b_.reshape(b_.shape[0], -1)))
            loss += torch.mean(1 +m_*cos_loss(a_,b_))
            partial_func = partial(modify_grad_v2, factor=factor)
            b_.register_hook(partial_func)
    
        loss = loss / len(a)
        return loss