from torchvision import transforms
from perlin import perlin_mask
from enum import Enum
import wml.wml_utils as wmlu
import math
import numpy as np
import pandas as pd
import os.path as osp
import PIL
import torch
import os
import glob
import random
import wml.img_utils as wmli
import wml.semantic.mask_utils as sem
import wml.wtorch.utils as wtu
import wml.object_detection2.visualization as odv
from .transforms import *
import cv2

_CLASSNAMES = [
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def align(v,a):
    return int(math.ceil(v/a)*a)


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test_public"
    PREDICT="predict"


class MVTecDataset2(torch.utils.data.Dataset):
    """
    PyTorch Dataset for MVTec.
    """

    def __init__(
            self,
            source,
            anomaly_source_path='/root/dataset/dtd/images',
            dataset_name='mvtec',
            classname='leather',
            resize=288,
            imagesize=288,
            split=DatasetSplit.TRAIN,
            rotate_degrees=0,
            translate=0,
            brightness_factor=0,
            contrast_factor=0,
            saturation_factor=0,
            gray_p=0,
            h_flip_p=0,
            v_flip_p=0,
            distribution=0,
            mean=0.5,
            std=0.1,
            fg=0,
            rand_aug=1,
            downsampling=8,
            scale=0,
            batch_size=8,
            **kwargs,
    ):
        """
        Args:
            source: [str]. Path to the MVTec data folder.
            classname: [str or None]. Name of MVTec class that should be
                       provided in this dataset. If None, the datasets
                       iterates over all available images.
            resize: [int]. (Square) Size the loaded image initially gets
                    resized to.
            imagesize: [int]. (Square) Size the resized loaded image gets
                       (center-)cropped to.
            split: [enum-option]. Indicates if training or test split of the
                   data should be used. Has to be an option taken from
                   DatasetSplit, e.g. mvtec.DatasetSplit.TRAIN. Note that
                   mvtec.DatasetSplit.TEST will also load mask data.
        """
        size_dict = {"can":[2230,1020],  "fabric":[2440,2040], "fruit_jelly":[2100,1520], 
        "rice":[2440,2040], "sheet_metal":[4220,1050],"vial":[1400,1900], "wallplugs":[2440,2040], "walnuts":[2440,2040]}

        super().__init__()
        self.source = source
        self.split = split
        self.batch_size = batch_size
        self.distribution = distribution
        self.mean = mean
        self.std = std
        self.fg = fg
        self.rand_aug = rand_aug
        self.downsampling = downsampling
        #self.resize = resize if self.distribution != 1 else [resize, resize]
        s = size_dict[classname]
        down_stride = math.ceil(math.sqrt(s[0]*s[1])/768)
        self.resize = [align(s[1]//down_stride,32),align(s[0]//down_stride,32)]
        print(f"Use resize {self.resize} for {classname}, downsample stride {down_stride}")
        self.imgsize = imagesize
        self.imagesize = (3, self.imgsize, self.imgsize)
        self.classname = classname
        self.dataset_name = dataset_name

        #if self.distribution != 1 and (self.classname == 'toothbrush' or self.classname == 'wood'):
            #self.resize = round(self.imgsize * 329 / 288)

        xlsx_path = './datasets/excel/' + self.dataset_name + '_distribution.xlsx'
        if self.fg == 2:  # choose by file
            try:
                df = pd.read_excel(xlsx_path)
                self.class_fg = df.loc[df['Class'] == self.dataset_name + '_' + classname, 'Foreground'].values[0]
            except:
                self.class_fg = 1
        elif self.fg == 1:  # with foreground mask
            self.class_fg = 1
        else:  # without foreground mask
            self.class_fg = 0

        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        self.anomaly_source_paths = sorted(1 * glob.glob(os.path.join(anomaly_source_path,"**","*.jpg"),recursive=True) +
                                           0 * list(next(iter(self.imgpaths_per_class.values())).values())[0])

        if split == DatasetSplit.TRAIN:
            self.transform_img = [
                Resize(self.resize),
                ColorJitter(brightness_factor, contrast_factor, saturation_factor),
                RandomHorizontalFlip(h_flip_p),
                RandomVerticalFlip(v_flip_p),
                RandomGrayscale(gray_p),
                #transforms.RandomAffine(rotate_degrees,
                #                        translate=(translate, translate),
                #                        scale=(1.0 - scale, 1.0 + scale),
                #                        interpolation=transforms.InterpolationMode.BILINEAR),
                #transforms.CenterCrop(self.imgsize),
                ToTensor(),
                Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
            self.transform_mask = None
        else:
            self.transform_img = [
                Resize(self.resize),
                ToTensor(),
                Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
            self.transform_mask = [
                transforms.Resize(self.resize),
                transforms.ToTensor(),
            ]
            self.transform_mask = transforms.Compose(self.transform_mask)

        self.transform_img = transforms.Compose(self.transform_img)
        self.transform_aug_img = [
            transforms.Resize(self.resize),
            transforms.ColorJitter(brightness_factor, contrast_factor, saturation_factor),
            transforms.RandomHorizontalFlip(h_flip_p),
            transforms.RandomVerticalFlip(v_flip_p),
            transforms.RandomGrayscale(0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
        self.transform_aug_img = transforms.Compose(self.transform_aug_img)

        self.gt_mask_files = self.get_gt_mask_files()
        print(f"img_transform: {self.transform_img}")
        print(f"aug_img_transform: {self.transform_aug_img}")
        print(f"mask_transform : {self.transform_mask}")

    def get_gt_mask_files(self):
        path = osp.join(self.source,self.classname)
        files = wmlu.get_files(path,suffix=".png")
        def filter_func(path):
            return "ground_truth" in path

        files = list(filter(filter_func,files))

        return files
        
    def get_mask_by_files(self,img_shape, feat_size,mask_fg=None):
        max_try = 10
        while max_try>0:
            mask_s,mask_l = self._get_mask_by_files(img_shape,feat_size,mask_fg)
            if np.sum(mask_s)>0:
                break
            max_try -= 1
        return mask_s,mask_l

    def _get_mask_by_files(self,img_shape, feat_size,mask_fg=None):
        gt_mask = random.choice(self.gt_mask_files)
        mask = PIL.Image.open(gt_mask)
        mask = np.array(mask)
        mask = wmli.resize_img(mask,size=(img_shape[2],img_shape[1]),keep_aspect_ratio=False,interpolation=cv2.INTER_NEAREST)
        points = np.nonzero(mask)
        x_min,x_max = np.min(points[1]),np.max(points[1])
        y_min,y_max = np.min(points[0]),np.max(points[0])
        valid_mask = mask[y_min:y_max,x_min:x_max]

        if np.random.rand()<0.5:
            valid_mask = valid_mask[::-1]
        if np.random.rand()<0.5:
            valid_mask = valid_mask[::,::-1]

        if mask_fg is not None and mask_fg.numel()>100:
            if torch.is_tensor(mask_fg):
                mask_fg = mask_fg.cpu().numpy()
            bbox = sem.get_mask_bbox(mask_fg)
            wh = bbox[2:]-bbox[:2]
            if wh[0]>=valid_mask.shape[1] and wh[1]>=valid_mask.shape[0]:
                x = np.random.randint(bbox[0],bbox[2]-valid_mask.shape[1])
                y = np.random.randint(bbox[1],bbox[3]-valid_mask.shape[0])
            else:
                x = np.random.randint(0,img_shape[2]-valid_mask.shape[1])
                y = np.random.randint(0,img_shape[1]-valid_mask.shape[0])
        else:
            x = np.random.randint(0,img_shape[2]-valid_mask.shape[1])
            y = np.random.randint(0,img_shape[1]-valid_mask.shape[0])

        mask_l = np.zeros(img_shape[1:],dtype=np.uint8)
        mask_l[y:y+valid_mask.shape[0],x:x+valid_mask.shape[1]] = valid_mask

        if mask_fg is not None and len(mask_fg) > 10:
            mask_l = (mask_l*mask_fg).astype(np.uint8)

        mask_s = wmli.resize_img(mask_l,size=(feat_size[1],feat_size[0]),keep_aspect_ratio=False,interpolation=cv2.INTER_NEAREST)
        mask_l = (mask_l>0).astype(np.float32)
        mask_s = (mask_s>0).astype(np.float32)

        return mask_s,mask_l



    def rand_augmenter(self,size=None):
        list_aug = [
            transforms.ColorJitter(contrast=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomGrayscale(p=1),
            transforms.RandomAutocontrast(p=1),
            transforms.RandomEqualize(p=1),
            transforms.RandomAffine(degrees=(-45, 45)),
        ]
        aug_idx = np.random.choice(np.arange(len(list_aug)), 3, replace=False)

        transform_aug = [
            transforms.Resize(self.resize) if size is None else transforms.Resize(size),
            list_aug[aug_idx[0]],
            list_aug[aug_idx[1]],
            list_aug[aug_idx[2]],
            #transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]

        transform_aug = transforms.Compose(transform_aug)
        return transform_aug

    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        fgmask_path = image_path.replace(classname,"fg_mask/"+classname)
        #print(image_path)
        #print(fgmask_path)
        image = PIL.Image.open(image_path).convert("RGB")
        results = {}
        results['img'] = image
        if self.class_fg:
            if osp.exists(fgmask_path):
                mask_fg = PIL.Image.open(fgmask_path)
                results['fg_mask'] = mask_fg
        results = self.transform_img(results)
        if 'img' in results:
            image = results['img']
        if 'fg_mask' in results:
            mask_fg = results['fg_mask']
        else:
            mask_fg = torch.tensor([1])

        mask_s = aug_image = torch.tensor([1])

        if self.split == DatasetSplit.TRAIN:
            aug = PIL.Image.open(np.random.choice(self.anomaly_source_paths)).convert("RGB")
            if self.rand_aug:
                transform_aug = self.rand_augmenter(size=image.shape[-2:])
                aug = transform_aug(aug)
            else:
                aug = self.transform_aug_img(aug)

            s = image.shape[-2:]
            if np.random.rand()<0.9:
                pl_max = np.random.randint(3,6+1)
                mask_all = perlin_mask(image.shape, [s[0]//self.downsampling,s[1]//self.downsampling], 0, pl_max, mask_fg, 1)
            else:
                try:
                    mask_all = self.get_mask_by_files(image.shape,[s[0]//self.downsampling,s[1]//self.downsampling],mask_fg=mask_fg)
                except Exception as e:
                    print(f"Get mask by file faild: {e}.")
                    pl_max = np.random.randint(3,6+1)
                    mask_all = perlin_mask(image.shape, [s[0]//self.downsampling,s[1]//self.downsampling], 0, pl_max, mask_fg, 1)
            mask_s = torch.from_numpy(mask_all[0])
            mask_l = torch.from_numpy(mask_all[1])

            beta = np.random.normal(loc=self.mean, scale=self.std)
            beta = np.clip(beta, .2, .8)
            aug_image = image * (1 - mask_l) + (1 - beta) * aug * mask_l + beta * image * mask_l

        if self.split == DatasetSplit.TEST and mask_path is not None:
            mask_gt = PIL.Image.open(mask_path).convert('L')
            mask_gt = self.transform_mask(mask_gt)
        else:
            mask_gt = torch.zeros([1, *image.size()[1:]])

        #self.save_img(image,aug_image,mask_s,"a.jpg")
        #wmli.imwrite("y.jpg",(mask_fg.cpu().numpy()*255).astype(np.uint8))
        return {
            "image": image,
            "aug": aug_image,
            "mask_s": mask_s,
            "mask_gt": mask_gt,
            "is_anomaly": int(anomaly != "good"),
            "image_path": image_path,
        }

    def save_img(self,image,aug_image,mask_s,path):
        image = wtu.unnormalize(image,self.mean*255,self.std*255).cpu().numpy().astype(np.uint8)
        aug_image = wtu.unnormalize(aug_image,self.mean*255,self.std*255).cpu().numpy().astype(np.uint8)
        image = np.transpose(image,[1,2,0])
        aug_image = np.transpose(aug_image,[1,2,0])
        mask_s = (mask_s.cpu().numpy()*255).astype(np.uint8)
        mask_s = np.expand_dims(mask_s,axis=-1)
        mask_s = wmli.resize_img(mask_s,[image.shape[1],image.shape[0]])
        couns,_ = cv2.findContours(mask_s,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        aug_image = np.ascontiguousarray(aug_image)
        aug_image = cv2.drawContours(aug_image,couns,-1,color=(0,255,0),thickness=2)
        mask_s = np.tile(mask_s,[1,1,3])
        img = np.concatenate([image,aug_image,mask_s],axis=1)
        img = np.ascontiguousarray(img)
        wmli.imwrite(path,img)


    def __len__(self):
        return len(self.data_to_iterate)

    def get_image_data(self):
        imgpaths_per_class = {}
        maskpaths_per_class = {}

        if self.split == DatasetSplit.TRAIN:
            splits = ["train","validation"]
        elif self.split == DatasetSplit.PREDICT:
            splits = ["test_private",  "test_private_mixed"]
        else:
            splits = [self.split.value]

        imgpaths_per_class[self.classname] = {}
        maskpaths_per_class[self.classname] = {}
        for split in splits:
            classpath = os.path.join(self.source, self.classname, split)
            if self.split == DatasetSplit.TEST:
                maskpath = os.path.join(self.source, self.classname, split,"ground_truth")
            anomaly_types = os.listdir(classpath)
            if "ground_truth" in anomaly_types:
                idx = anomaly_types.index("ground_truth")
                del anomaly_types[idx]

            if self.split == DatasetSplit.PREDICT:
                anomaly = "NONE"
                anomaly_path = classpath
                anomaly_files = sorted(os.listdir(anomaly_path))

                imgpaths_per_class[self.classname][anomaly] = imgpaths_per_class[self.classname].get(anomaly,[]) + [os.path.join(anomaly_path, x) for x in anomaly_files]
                maskpaths_per_class[self.classname]["good"] = None
            
            else:
                for anomaly in anomaly_types:
                    anomaly_path = os.path.join(classpath, anomaly)
                    anomaly_files = sorted(os.listdir(anomaly_path))
    
                    imgpaths_per_class[self.classname][anomaly] = imgpaths_per_class[self.classname].get(anomaly,[]) + [os.path.join(anomaly_path, x) for x in anomaly_files]
                    
    
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        anomaly_mask_path = os.path.join(maskpath, anomaly)
                        anomaly_mask_files = sorted(os.listdir(anomaly_mask_path))
                        maskpaths_per_class[self.classname][anomaly] = maskpaths_per_class[self.classname].get(anomaly,[])+[os.path.join(anomaly_mask_path, x) for x in anomaly_mask_files]
                    else:
                        maskpaths_per_class[self.classname]["good"] = None


        data_to_iterate = []
        for classname in sorted(imgpaths_per_class.keys()):
            for anomaly in sorted(imgpaths_per_class[classname].keys()):
                for i, image_path in enumerate(imgpaths_per_class[classname][anomaly]):
                    data_tuple = [classname, anomaly, image_path]
                    if self.split == DatasetSplit.TEST and anomaly != "good":
                        data_tuple.append(maskpaths_per_class[classname][anomaly][i])
                    else:
                        data_tuple.append(None)
                    data_to_iterate.append(data_tuple)

        return imgpaths_per_class, data_to_iterate
