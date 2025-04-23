from torchvision import transforms
import torch.nn as nn
from torchvision.transforms import functional as F
import torch


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

class ColorJitter(nn.Module):
    def __init__(self,brightness=[0.5,1.5],p=0.5):
        super().__init__()
        self.brightness = brightness
        self.p = p

    def forward(self,results):
        if torch.rand(1) > self.p:
            return results
        factor = float(torch.empty(1).uniform_(self.brightness[0], self.brightness[1]))
        img = results['img']
        img = F.adjust_brightness(img,factor)
        results['img'] = img
        return results

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