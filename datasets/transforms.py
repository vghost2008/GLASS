from torchvision import transforms
import torch.nn as nn
from torchvision.transforms import functional as F
import torch
import numpy as np
from PIL import Image


class Resize(transforms.Resize):
    def forward(self,results):
        img = results['img']
        img = super().forward(img)
        results['img'] = img
        if 'fg_mask' in results:
            fg_mask = results['fg_mask']
            fg_mask = super().forward(fg_mask)
            results['fg_mask'] = fg_mask
        return results

class RandomCut(nn.Module):
    def __init__(self,size,enable=True):
        super().__init__()
        self.size = size
        self.enable = enable

    def forward(self,results):
        if not self.enable:
            return results
        img = results['img']
        img = np.array(img)
        max_y = img.shape[0]-self.size[0]
        max_x = img.shape[1]-self.size[1]
        x_pos = np.random.randint(0,max_x)
        y_pos = np.random.randint(0,max_y)
        img = img[y_pos:y_pos+self.size[0],x_pos:x_pos+self.size[1]]
        img = Image.fromarray(img)
        results['img'] = img
        if 'fg_mask' in results:
            fg_mask = results['fg_mask']
            fg_mask = np.array(fg_mask)
            fg_mask = fg_mask[y_pos:y_pos+self.size[0],x_pos:x_pos+self.size[1]]
            fg_mask = Image.fromarray(fg_mask)
            results['fg_mask'] = fg_mask
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(size={self.size}, enable={self.enable})"

class ColorJitter(nn.Module):
    def __init__(self,brightness=[0.5,1.5],p=0.5,p2=0.7):
        super().__init__()
        self.brightness = brightness
        self.p = p
        self.p2 = p2

    def forward(self,results):
        if torch.rand(1) > self.p:
            return results
        factor = float(torch.empty(1).uniform_(self.brightness[0], self.brightness[1]))
        img = results['img']
        adj_img = F.adjust_brightness(img,factor)
        if torch.rand(1)<self.p2:
            img = np.array(img)
            adj_img = np.array(adj_img)
            alpha = self.get_alpha(img)
            old_type = img.dtype
            new_img = img.astype(np.float32)*alpha+adj_img.astype(np.float32)*(1-alpha)
            new_img = new_img.astype(old_type)
            new_img = Image.fromarray(new_img)
        else:
            new_img = adj_img
        results['img'] = new_img
        return results

    def get_alpha(self,img):
        if torch.rand(1)>0.5:
            nr = img.shape[0]
            a = np.linspace(0,1,nr)
            if torch.rand(1)>0.5:
                a = a[::-1]
            a = np.reshape(a,[-1,1,1])
        else:
            nr = img.shape[1]
            a = np.linspace(0,1,nr)
            if torch.rand(1)>0.5:
                a = a[::-1]
            a = np.reshape(a,[1,-1,1])
        
        return a

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, p2={self.p2})"

class RandomHorizontalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, results):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """

        if torch.rand(1) > self.p:
            return results
        img = results['img']
        img = F.hflip(img)
        results['img'] = img
        if 'fg_mask' in results:
            fg_mask = results['fg_mask']
            results['fg_mask'] = F.hflip(fg_mask)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

class RandomVerticalFlip(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, results):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """

        if torch.rand(1) > self.p:
            return results
        img = results['img']
        img = F.vflip(img)
        results['img'] = img
        if 'fg_mask' in results:
            fg_mask = results['fg_mask']
            results['fg_mask'] = F.vflip(fg_mask)
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"

class RandomGrayscale(transforms.RandomGrayscale):
    def forward(self,results):
        img = results['img']
        img = super().forward(img)
        results['img'] = img
        return results

class ToTensor(nn.Module):
    def forward(self,results):
        img = results['img']
        img = F.to_tensor(img)
        results['img'] = img
        if 'fg_mask' in results:
            fg_mask = results['fg_mask']
            fg_mask = F.to_tensor(fg_mask)
            fg_mask = torch.ceil(fg_mask)[0]
            results['fg_mask'] = fg_mask
        return results

class Normalize(transforms.Normalize):
    def forward(self,results):
        img = results['img']
        img = super().forward(img)
        results['img'] = img
        return results