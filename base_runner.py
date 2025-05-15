from datetime import datetime
import pandas as pd
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = "python"
import logging
import sys
import click
import torch
import warnings
import backbones
import glass
import utils
import numpy as np
import time
#from wml.wtorch.data import DataLoader as torchDataLoader
from wml.env_utils import get_git_info
#[(0, 'can'), (1, 'fabric'), (2, 'fruit_jelly'), (3, 'rice'), (4, 'sheet_metal'), (5, 'vial'), (6, 'wallplugs'), (7, 'walnuts')]
#classname walnuts, split DatasetSplit.TRAIN, len 480, 7
#classname can, split DatasetSplit.TRAIN, len 458, 0
#classname fabric, split DatasetSplit.TRAIN, len 430, 1
#classname rice, split DatasetSplit.TRAIN, len 348, 3
#classname vial, split DatasetSplit.TRAIN, len 332, 5
#classname wallplugs, split DatasetSplit.TRAIN, len 326, 6
#classname fruit_jelly, split DatasetSplit.TRAIN, len 300, 2
#classname sheet_metal, split DatasetSplit.TRAIN, len 156, 4
DataLoader = torch.utils.data.DataLoader
#DataLoader = torchDataLoader


@click.group(chain=True)
@click.option("--results_path", type=str, default="results")
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--run_name", type=str, default="test")
@click.option("--test", type=str, default="ckpt")
@click.option("--ckpt_path", type=str, default=None)
@click.option("--lpid", type=int, default=-1, show_default=True)
@click.option("--gpu_mem", type=int, default=22, show_default=True)
def main(**kwargs):
    pass


@main.command("net")
@click.option("--dsc_margin", type=float, default=0.5)
@click.option("--train_backbone", is_flag=True)
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=["layer1","layer2","layer3","layer4"])
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--patchsize", type=int, default=3)
@click.option("--meta_epochs", type=int, default=640)
@click.option("--eval_epochs", type=int, default=20)
@click.option("--dsc_layers", type=int, default=2)
@click.option("--dsc_hidden", type=int, default=1024)
@click.option("--pre_proj", type=int, default=1)
@click.option("--mining", type=int, default=1)
@click.option("--noise", type=float, default=0.015)
@click.option("--radius", type=float, default=0.75)
@click.option("--p", type=float, default=0.5)
@click.option("--lr", type=float, default=0.0001)
@click.option("--svd", type=int, default=0)
@click.option("--step", type=int, default=20)
@click.option("--limit", type=int, default=392)
def net(
        backbone_names,
        layers_to_extract_from,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize,
        meta_epochs,
        eval_epochs,
        dsc_layers,
        dsc_hidden,
        dsc_margin,
        train_backbone,
        pre_proj,
        mining,
        noise,
        radius,
        p,
        lr,
        svd,
        step,
        limit,
):
    print(f"git {get_git_info()}")
    backbone_names = list(backbone_names)
    if len(backbone_names) > 1:
        layers_to_extract_from_coll = []
        for idx in range(len(backbone_names)):
            layers_to_extract_from_coll.append(layers_to_extract_from)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_glass(input_shape, device,is_training=False,dataloader_len=-1):
        glasses = []
        for backbone_name, layers_to_extract_from in zip(backbone_names, layers_to_extract_from_coll):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(backbone_name.split("-")[-1])
            backbone = backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            glass_inst = glass.GLASS(device)
            glass_inst.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                meta_epochs=meta_epochs,
                eval_epochs=eval_epochs,
                dsc_layers=dsc_layers,
                dsc_hidden=dsc_hidden,
                dsc_margin=dsc_margin,
                train_backbone=train_backbone,
                pre_proj=pre_proj,
                mining=mining,
                noise=noise,
                radius=radius,
                p=p,
                lr=lr,
                svd=svd,
                step=step,
                limit=limit,
                is_training = is_training,
                dataloader_len=dataloader_len,
            )
            glasses.append(glass_inst.to(device))
        return glasses

    return "get_glass", get_glass


@main.command("dataset")
@click.option("--name", type=str,default="mvtec2")
@click.option("--data_path", type=click.Path(exists=True, file_okay=False),default='/home/wj/ai/mldata1/MVTEC/datasets')
@click.option("--aug_path", type=click.Path(exists=True, file_okay=False),default='/home/wj/ai/mldata1/MVTEC/other_datasets/dtd')
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=5, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=288, type=int, show_default=True)
@click.option("--imagesize", default=288, type=int, show_default=True)
@click.option("--imgsz", default=-1, type=int, show_default=True)
@click.option("--rotate_degrees", default=0, type=int)
@click.option("--translate", default=0, type=float)
@click.option("--scale", default=0.0, type=float)
@click.option("--brightness", default=0.0, type=float)
@click.option("--contrast", default=0.0, type=float)
@click.option("--saturation", default=0.0, type=float)
@click.option("--gray", default=0.0, type=float)
@click.option("--hflip", default=0.5, type=float)
@click.option("--vflip", default=0.5, type=float)
@click.option("--distribution", default=0, type=int)
@click.option("--mean", default=0.5, type=float)
@click.option("--std", default=0.1, type=float)
@click.option("--fg", default=1, type=int)
@click.option("--rand_aug", default=1, type=int)
@click.option("--downsampling", default=4, type=int)
@click.option("--augment", is_flag=True)
@click.option("--align", default=32, type=int)
def dataset(
        name,
        data_path,
        aug_path,
        subdatasets,
        batch_size,
        resize,
        imagesize,
        imgsz,
        num_workers,
        rotate_degrees,
        translate,
        scale,
        brightness,
        contrast,
        saturation,
        gray,
        hflip,
        vflip,
        distribution,
        mean,
        std,
        fg,
        rand_aug,
        downsampling,
        augment,
        align,
):
    _DATASETS = {"mvtec": ["datasets.mvtec", "MVTecDataset"], "visa": ["datasets.visa", "VisADataset"],"mvtec2": ["datasets.mvtec2", "MVTecDataset2"],
                 "mpdd": ["datasets.mvtec", "MVTecDataset"], "wfdd": ["datasets.mvtec", "MVTecDataset"], }
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    all_names = ["can"  , "fabric"  , "fruit_jelly"  , "rice"  , "sheet_metal"  , "vial"  , "wallplugs"  , "walnuts"]

    subdatasets = list(subdatasets)
    print(f"subdatasets {subdatasets}")
    subdatasets = list(subdatasets)
    for i,v in enumerate(list(subdatasets)):
        try:
            if len(v)<=2:
                idx = int(v)
                if idx == -1:
                    subdatasets = all_names
                    break
                else:
                    subdatasets[i] = all_names[idx]
        except Exception as e:
            print(e)
            pass
    print(f"subdatasets {subdatasets}")

    def get_dataloaders(seed, test, get_name=name):
        pin_memory = True
        dataloaders = []
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                aug_path,
                classname=subdataset,
                imgsz=imgsz,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
                align_v=align,
            )

            test_dataloader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=pin_memory,
            )

            test_dataloader.name = get_name + "_" + subdataset

            predict_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                aug_path,
                classname=subdataset,
                imgsz=imgsz,
                split=dataset_library.DatasetSplit.PREDICT,
                seed=seed,
                align_v=align,
            )

            predict_dataloader = DataLoader(
                predict_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=pin_memory,
            )

            predict_dataloader.name = get_name + "_" + subdataset

            if test == 'ckpt':
                train_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    aug_path,
                    dataset_name=get_name,
                    classname=subdataset,
                    imgsz=imgsz,
                    split=dataset_library.DatasetSplit.TRAIN,
                    seed=seed,
                    rotate_degrees=rotate_degrees,
                    translate=translate,
                    brightness_factor=brightness,
                    contrast_factor=contrast,
                    saturation_factor=saturation,
                    gray_p=gray,
                    h_flip_p=hflip,
                    v_flip_p=vflip,
                    scale=scale,
                    distribution=distribution,
                    mean=mean,
                    std=std,
                    fg=fg,
                    rand_aug=rand_aug,
                    downsampling=downsampling,
                    augment=augment,
                    batch_size=batch_size,
                    align_v=align,
                )

                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    prefetch_factor=2,
                    pin_memory=pin_memory,
                )

                train_dataloader.name = test_dataloader.name
                print(f"Dataset {subdataset.upper():^20}: train={len(train_dataset)} test={len(test_dataset)}")

                base_train_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    aug_path,
                    dataset_name=get_name,
                    classname=subdataset,
                    imgsz=imgsz,
                    split=dataset_library.DatasetSplit.BASE_TRAIN,
                    seed=seed,
                    rotate_degrees=rotate_degrees,
                    translate=translate,
                    brightness_factor=brightness,
                    contrast_factor=contrast,
                    saturation_factor=saturation,
                    gray_p=gray,
                    h_flip_p=hflip,
                    v_flip_p=vflip,
                    scale=scale,
                    distribution=distribution,
                    mean=mean,
                    std=std,
                    fg=fg,
                    rand_aug=rand_aug,
                    downsampling=downsampling,
                    augment=augment,
                    batch_size=batch_size,
                    align_v=align,
                )

                base_train_dataloader = DataLoader(
                    base_train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    prefetch_factor=2,
                    pin_memory=pin_memory,
                )

                base_train_dataloader.name = test_dataloader.name
            else:
                train_dataloader = test_dataloader
                print(f"Dataset {subdataset.upper():^20}: train={0} test={len(test_dataset)}")

            dataloader_dict = {
                "training": train_dataloader,
                "base_training": base_train_dataloader,
                "testing": test_dataloader,
                "predict": predict_dataloader,
            }
            dataloaders.append(dataloader_dict)

        print("\n")
        return dataloaders

    return "get_dataloaders", get_dataloaders

