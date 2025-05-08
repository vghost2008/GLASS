import torchvision.models as models
import math
import timm
import timm
from torchvision import transforms
from PIL import Image
import torch
import os
import numpy as np
import torch.nn as nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes,draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image
from models.vit_encoder import load as dino_load
from models.image_encoder import ImageEncoderViT
import wml.wtorch.utils as wtu

_BACKBONES = {
    "alexnet": "models.alexnet(pretrained=True)",
    "resnet18": "models.resnet18(pretrained=True)",
    "resnet50": "models.resnet50(pretrained=True)",
    "resnet101": "models.resnet101(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "resnetv2_50_bit": 'timm.create_model("resnetv2_50x3_bitm", pretrained=True)',
    "resnetv2_50_21k": 'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True)',
    "resnetv2_101_bit": 'timm.create_model("resnetv2_101x3_bitm", pretrained=True)',
    "resnetv2_101_21k": 'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)',
    "resnetv2_152_bit": 'timm.create_model("resnetv2_152x4_bitm", pretrained=True)',
    "resnetv2_152_21k": 'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True)',
    "resnetv2_152_384": 'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True)',
    "resnetv2_101": 'timm.create_model("resnetv2_101", pretrained=True)',
    "vgg11": "models.vgg11(pretrained=True)",
    "vgg19": "models.vgg19(pretrained=True)",
    "vgg19_bn": "models.vgg19_bn(pretrained=True)",
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
    "wideresnet101": "models.wide_resnet101_2(pretrained=True)",
    "mnasnet_100": 'timm.create_model("mnasnet_100", pretrained=True)',
    "mnasnet_a1": 'timm.create_model("mnasnet_a1", pretrained=True)',
    "mnasnet_b1": 'timm.create_model("mnasnet_b1", pretrained=True)',
    "densenet121": 'timm.create_model("densenet121", pretrained=True)',
    "densenet201": 'timm.create_model("densenet201", pretrained=True)',
    "inception_v4": 'timm.create_model("inception_v4", pretrained=True)',
    "vit_small": 'timm.create_model("vit_small_patch16_224", pretrained=True)',
    "vit_base": 'timm.create_model("vit_base_patch16_224", pretrained=True)',
    "vit_large": 'timm.create_model("vit_large_patch16_224", pretrained=True)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    "vit_deit_base": 'timm.create_model("deit_base_patch16_224", pretrained=True)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch16_224", pretrained=True)',
    "vit_swin_base": 'timm.create_model("swin_base_patch4_window7_224", pretrained=True)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b1": 'timm.create_model("tf_efficientnet_b1", pretrained=True)',
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=True)',
    "convnext": 'timm.create_model("convnextv2_base", pretrained=True)',
    "maskrcnn": 'create_maskrcnn()',
    "dino": 'create_dino()',
    "ensemble": 'create_ensemble()',
    "vit": 'create_vit()',
}

def create_vit():
    backbone = ImageEncoderViT(img_size=768)
    state_dict = torch.load("./backbones/weights/sam_vit_b_01ec64.pth")
    wtu.forgiving_state_restore(backbone,state_dict)
    return backbone

def create_maskrcnn():
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn_v2(weights=weights, progress=False)
    model = model.backbone
    model = model.eval()
    return model

class DINOBackbone(nn.Module):
    def __init__(
            self,
            target_layers =[2, 3, 4, 5, 6, 7, 8, 9],
            remove_class_token=True,
            encoder_require_grad_layer=[],
    ) -> None:
        super().__init__()
        self.encoder = dino_load("dinov2reg_vit_base_14")
        self.target_layers = target_layers #[2, 3, 4, 5, 6, 7, 8, 9]
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer #[]
        self.aggregator = "NetworkFeatureAggregatorV3"

        if not hasattr(self.encoder, 'num_register_tokens'): #4
            self.encoder.num_register_tokens = 0

    def forward(self, x):
        _,_,IH,IW = x.shape
        x = self.encoder.prepare_tokens(x)
        B, L, _ = x.shape #[16,789,768],[B,L,C]
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)

        if self.remove_class_token: # True
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list] #self.encoder.num_register_tokens==4
        
        res = {}
        B,_,C = en_list[0].shape

        for i,v in enumerate(en_list):
            v = torch.reshape(v,[B,IH//14,IW//14,C])
            v = torch.permute(v,[0,3,1,2])
            res[str(i)] = v
        
        return res

class Ensemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone0 = create_maskrcnn()
        self.backbone1 = DINOBackbone()
        self.aggregator = "NetworkFeatureAggregatorV4"
    
    def forward(self,x):
        with torch.no_grad():
            res0 = self.backbone0(x)
            res1 = self.backbone1(x)
            res1['A'] = res0['0']  #res1['A] -> mask rcnn
        return res1
    

def create_dino():
    return DINOBackbone()

def create_ensemble():
    return Ensemble()

def load(name):
    '''
    out_info: 存储模型输出的key及通道数
    out_dict与out_info有重复，可以不设置
    '''
    backbone = eval(_BACKBONES[name])
    if name == "wideresnet50":
        backbone.out_info = [["layer1","layer2","layer3","layer4"],[256,512,1024,2048]]
    elif name == "convnext":
        backbone.out_info = [["stages.0","stages.1","stages.2","stages.3"],[128,256,512,1024]]
    elif name == "maskrcnn":
        backbone.out_info = [['0','1','2','3'],[256,256,256,256]]
        backbone.out_dict = ['0','1','2','3']
    elif name == "dino":
        backbone.out_info = [['0','1','2','3','4','5','6','7'],[768]*8]
        backbone.out_dict = ['0','1','2','3','4','5','6','7']
    elif name == "ensemble":
        backbone.out_info = [['0','1','2','3','4','5','6','7'],[768]*8]
        backbone.out_dict = ['A','0','1','2','3','4','5','6','7']
    
    return backbone
