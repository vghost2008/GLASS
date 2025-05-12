import torch
from wml.wtorch import ConvModule


def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    if isinstance(m, torch.nn.BatchNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, torch.nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)


class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=2, hidden=None):
        super(Discriminator, self).__init__()
        self.img_shape = None
        _hidden = 512
        self.body = torch.nn.Sequential()
        for i in range(3):
            _in = in_planes if i == 0 else _hidden
            self.body.add_module('block%d' % (i + 1),
                                     ConvModule(_in,_hidden,3,1,1,act_cfg=dict(type="Swish"),norm_cfg=dict(type='BN',momentum=0.01))
                                 )
        self.tail = torch.nn.Conv2d(_hidden, 1, 1,1,bias=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.apply(init_weight)

    def forward(self, x):
        B,_,H,W = self.img_shape
        C = x.shape[-1]
        with torch.cuda.amp.autocast():
            x = torch.reshape(x,[B,H//4,W//4,C])
            x = torch.permute(x,[0,3,1,2])
            x = self.body(x)
            x = self.tail(x)
            x = torch.permute(x,[0,2,3,1])
            x = torch.reshape(x,[-1,1])
        with torch.cuda.amp.autocast(False):
            score = self.sigmoid(x.float())
        return score,x


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


class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=None):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1)
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        x = x[:, :, 0]
        x = torch.max(x, dim=1).values
        return x
