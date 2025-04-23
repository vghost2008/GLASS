import copy
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
from wml.wtorch import ConvModule
import wml.wtorch.train_toolkit as wtt
import torch.nn as nn
from wml.wtorch.nn import MParent,WeakRefmodel

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for _ in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1) ##[B*RH*RW,1,C*P*P]
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class RescaleSegmentor:
    def __init__(self, device, target_size=288):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores,target_size=None):
        if target_size is None:
            target_size = self.target_size
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()
        return [ndimage.gaussian_filter(patch_score, sigma=self.smoothing) for patch_score in patch_scores]


class NetworkFeatureAggregator(torch.nn.Module):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device, train_backbone=False):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """
        self.outputs = {}
        self.backbone = backbone
        self.device = device
        self.train_backbone = train_backbone
        if hasattr(backbone,'out_dict'):
            out_dict = backbone.out_dict
            self.layers_to_extract_from = out_dict
        else:
            self.layers_to_extract_from = layers_to_extract_from

            if not hasattr(backbone, "hook_handles"):
                self.backbone.hook_handles = []
            for handle in self.backbone.hook_handles:
                handle.remove()
    
            for extract_layer in layers_to_extract_from:
                self.register_hook(extract_layer)


        self.to(self.device)

    def forward(self, images, eval=True):
        self.outputs.clear()
        if self.train_backbone and not eval:
            self.backbone(images)
        else:
            with torch.no_grad():
                try:
                    res = self.backbone(images)
                    if isinstance(res,dict) and len(self.outputs)==0:
                        self.outputs = res
                except LastLayerToExtractReachedException:
                    pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]

    def register_hook(self, layer_name):
        module = self.find_module(self.backbone, layer_name)
        if hasattr(module,"blocks"):
            module = module.blocks
        if module is not None:
            forward_hook = ForwardHook(self.outputs, layer_name, self.layers_to_extract_from[-1])
            if isinstance(module, torch.nn.Sequential):
                hook = module[-1].register_forward_hook(forward_hook)
            else:
                hook = module.register_forward_hook(forward_hook)
            self.backbone.hook_handles.append(hook)
        else:
            raise ValueError(f"Module {layer_name} not found in the model")
    
    def find_module(self, model, module_name):
        for name, module in model.named_modules():
            if name == module_name:
                return module
            elif '.' in module_name:
                father, child = module_name.split('.', 1)
                if name == father:
                    return self.find_module(module, child)
        print(f"ERROR: find {module_name} faild.")
        return None

class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        return None

class LastLayerToExtractReachedException(Exception):
    pass

class NetworkFeatureAggregatorV2(NetworkFeatureAggregator):
    """Efficient extraction of network features."""

    def __init__(self, backbone, layers_to_extract_from, device, train_backbone=False,in_channels=[256,512,1024,2048]):
        if hasattr(backbone,"out_info"):
            layers_to_extract_from,in_channels = backbone.out_info
        super().__init__(backbone=backbone,layers_to_extract_from=layers_to_extract_from,device=device,train_backbone=False)
        print(f"layers_to_extract_from: {self.layers_to_extract_from}")
        wtt.freeze_model(backbone)
        channels = 256
        self.lateral_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()
        for i,ic in enumerate(in_channels):
            self.lateral_convs.append(nn.Conv2d(ic,channels,1))
            if i<len(in_channels)-1:
                self.out_convs.append(ConvModule(channels,channels,3,1,1,act_cfg=dict(type="Swish")))
            else:
                self.out_convs.append(ConvModule(channels*2,channels*2,3,1,1,act_cfg=dict(type="Swish")))

        self.output = ConvModule(channels*2,channels*2,3,1,1,act_cfg=dict(type="Swish"))
        wtt.set_bn_eps(self,1e-3)

        

    def forward(self, images, eval=True):
        outputs = super().forward(images)
        features = [outputs[n] for n in self.layers_to_extract_from]
        new_features = []
        shapes = []
        for f,m in zip(features,self.lateral_convs):
            shapes.append(f.shape[-2:])
            new_features.append(m(f))
        last_feature = None
        for i,(f,m) in enumerate(zip(new_features[::-1],self.out_convs)):
            if i ==0:
                last_feature = m(f)
            elif i<len(new_features)-1:
                last_feature = F.interpolate(last_feature,scale_factor=2,mode='nearest')
                last_feature = m(last_feature+f)
            else:
                last_feature = F.interpolate(last_feature,scale_factor=2,mode='nearest')
                last_feature = m(torch.cat([f,last_feature],dim=1))
        
        feature = self.output(last_feature)

        feature = torch.permute(feature,[0,2,3,1])
        C = feature.shape[-1]
        feature = torch.reshape(feature,[-1,C])
        return feature,shapes


    def train(self,mode=True):
        [m.train(mode=mode) for m in self.lateral_convs]
        [m.train(mode=mode) for m in self.out_convs]
        self.output.train(mode=mode)
        if mode==False:
            self.backbone.eval()

    
    def train_parameters(self):
        res = []
        res.extend(list(self.lateral_convs))
        res.extend(list(self.out_convs))
        res.extend(list(self.output))
        return res




