from loss import FocalLoss,wvarifocal_loss
from collections import OrderedDict
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Projection, PatchMaker
import tifffile
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os.path as osp
import logging
import os
import math
import torch
import tqdm
import common
import metrics
import sys
import cv2
import utils
import glob
import shutil
import wml.wml_utils as wmlu
import wml.wtorch.summary as wsummary
import wml.wtorch.utils as wtu
import wml.wtorch.train_toolkit as wtt
import wml.wtorch.nn as wnn
import wml.img_utils as wmli
import colorama
import time
from wml.semantic.mask_utils import npresize_mask,resize_mask,npresize_mask_mt
from datadef import get_class_name, get_img_cut_nr
import torchvision
import traceback
from utils import *
from wml.wtorch.wlr_scheduler import WarmupCosLR
from wml.wtorch.ema import ModelEMA

def trace_grad_fn(grad_fn, depth=0):
    if grad_fn is None:
        return
    print("  " * depth, grad_fn.name())
    for next_fn, _ in grad_fn.next_functions:
        if next_fn is not None:
            trace_grad_fn(next_fn, depth + 1)


LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TBWrapper:
    def __init__(self, log_dir):
        self.g_iter = 0
        log_dir = osp.abspath(osp.expanduser(log_dir))
        print(f"TB logdir {log_dir}")
        wmlu.create_empty_dir_remove_if(log_dir,"tb")
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


class GLASS(torch.nn.Module):
    def __init__(self, device):
        super(GLASS, self).__init__()
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(init_scale=100.0)
        self.max_norm = 16
        self.prf = None
        self.f1_w = 9
        self.pauroc_w = 1
        self.sum_w = self.f1_w+self.pauroc_w


    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=3,
            patchstride=1,
            meta_epochs=640,
            eval_epochs=1,
            dsc_layers=2,
            dsc_hidden=1024,
            dsc_margin=0.5,
            train_backbone=False,
            pre_proj=1,
            mining=1,
            noise=0.015,
            radius=0.75,
            p=0.5,
            lr=0.0001,
            svd=0,
            step=20,
            limit=392,
            dataloader_len=100,
            **kwargs,
    ):

        train_backbone = True
        backbone = backbone.to(device)
        self.backbone = wnn.WeakRefmodel(backbone)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device

        if hasattr(backbone,"aggregator") and backbone.aggregator == "NetworkFeatureAggregatorV3":
            feature_aggregator = common.NetworkFeatureAggregatorV3(
                backbone, self.layers_to_extract_from, self.device, train_backbone
            ).to(device)
        elif hasattr(backbone,"aggregator") and backbone.aggregator == "NetworkFeatureAggregatorV4":
            feature_aggregator = common.NetworkFeatureAggregatorV4(
                backbone, self.layers_to_extract_from, self.device, train_backbone
            ).to(device)
        elif hasattr(backbone,"aggregator") and backbone.aggregator == "NetworkFeatureAggregatorV5":
            feature_aggregator = common.NetworkFeatureAggregatorV5(
                backbone, self.layers_to_extract_from, self.device, train_backbone
            ).to(device)
        else:
            feature_aggregator = common.NetworkFeatureAggregatorV2(
                backbone, self.layers_to_extract_from, self.device, train_backbone
            ).to(device)
        self.forward_modules = feature_aggregator

        self.target_embed_dimension = target_embed_dimension

        self.meta_epochs = meta_epochs
        self.lr = lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            #self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)
            self.backbone_opt = self.get_embed_optim()
            self.backbone_ls = WarmupCosLR(self.backbone_opt,400,640*dataloader_len)

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.Adam(self.pre_projection.parameters(), lr, weight_decay=1e-5)
            self.proj_ls = WarmupCosLR(self.proj_opt,400,640*dataloader_len)

        self.eval_epochs = eval_epochs
        self.eval_offset = 0
        if self.eval_epochs<=2:
            print(f"WARNING: eval epochs={self.eval_epochs}")
        else:
            #self.eval_offset = np.random.randint(1,self.eval_epochs)
            print(f"INFO: eval epochs={self.eval_epochs}, eval offset={self.eval_offset}")
        self.dsc_layers = dsc_layers
        self.dsc_hidden = dsc_hidden
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=lr * 2, weight_decay=1e-4)
        self.dsc_ls = WarmupCosLR(self.dsc_opt,400,640*dataloader_len)
        self.dsc_margin = dsc_margin

        self.p = p
        self.radius = radius
        self.mining = mining
        self.noise = noise
        self.svd = svd
        self.step = step
        self.limit = limit
        self.distribution = 0
        self.focal_loss = FocalLoss()

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.anomaly_segmentor = common.RescaleSegmentor(device=self.device, target_size=input_shape[-2:])
        self.model_dir = ""
        self.dataset_name = ""
        self.logger = None
        print(f"Dataloader len {dataloader_len}")
        self.ema = ModelEMA(self,base_updates=dataloader_len*4,decay=1-1.0/(2*dataloader_len))
        print(f"EMA: {self.ema}")

    def get_embed_optim(self):
        bn_weights,weights,biases,unbn_weights,unweights,unbiases = wtt.simple_split_parameters(self.forward_modules,return_unused=True)
        optimizer = torch.optim.AdamW(
                weights,
                lr=self.lr,
                weight_decay=1e-5,
            )
        if len(bn_weights)>0:
            optimizer.add_param_group({"params": bn_weights,"weight_decay":0.0})
        if len(biases)>0:
            optimizer.add_param_group({"params": biases,"weight_decay":0.0})
        #optimizer,unbn_weights,unweights,unbiases
        return optimizer

    def set_model_dir(self, model_dir, dataset_name,run_save_path="./results",tb_dir="tb"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, tb_dir)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)
        self.run_save_path = osp.abspath(run_save_path)

        print(f"Model dir: {model_dir}")
        print(f"ckpt dir: {self.ckpt_dir}")
        print(f"tb dir {self.tb_dir}")
        print(f"Run save path {self.run_save_path}")

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""
        if not evaluation and self.train_backbone:
            self.forward_modules.train()
            features,shapes = self.forward_modules(images, eval=evaluation)
        else:
            self.forward_modules.eval()
            with torch.no_grad():
                features,shapes = self.forward_modules(images)


        return features, shapes #patch_shapes ((H0,W0),(H1,W1),...)

    def get_score(self,f1,pauroc):
        return (f1*self.f1_w+pauroc*self.pauroc_w)/self.sum_w

    def trainer(self, training_data, val_data, base_training_data,name):
        print(self)
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        ckpt_path_save = os.path.join(self.ckpt_dir, "ckpt.pth")

        if base_training_data is None:
            base_training_data = training_data

        '''
        if len(ckpt_path) != 0:
            LOGGER.info("Start testing, ckpt file found!")
            return 0., 0., 0., 0., 0., -1.
        '''

        ckpt_path_best = ""

        self.distribution = training_data.dataset.distribution
        xlsx_path = './datasets/excel/' + name.split('_')[0] + '_distribution.xlsx'
        '''
        try:
            if self.distribution == 1:  # rejudge by image-level spectrogram analysis
                self.distribution = 1
                self.svd = 1
            elif self.distribution == 2:  # manifold
                self.distribution = 0
                self.svd = 0
            elif self.distribution == 3:  # hypersphere
                self.distribution = 0
                self.svd = 1
            elif self.distribution == 4:  # opposite choose by file
                self.distribution = 0
                df = pd.read_excel(xlsx_path)
                self.svd = 1 - df.loc[df['Class'] == name, 'Distribution'].values[0]
            else:  # choose by file
                self.distribution = 0
                df = pd.read_excel(xlsx_path)
                self.svd = df.loc[df['Class'] == name, 'Distribution'].values[0]
        except Exception as e:
            print(f"ERROR: distribution faild, {e}")
            self.distribution = 0
            self.svd = 0
        '''
        self.distribution = 0
        self.svd = 0

        print(f"Model info:")
        wtt.show_model_parameters_info(self)
        pbar = tqdm.tqdm(range(self.meta_epochs), unit='epoch')
        pbar_str1 = ""
        best_record = None
        error_nr = 0
        min_error_nr = 0
        base_epoch = 400
        stop_epoch_nr = 120
        for i_epoch in pbar:
            try:
                self.forward_modules.eval()
                pbar_str, pt, pf = self._train_discriminator_amp(training_data, i_epoch, pbar, pbar_str1)

    
                ckpt_path_cur = os.path.join(self.ckpt_dir, "cur_ckpt.pth".format(i_epoch))
                utils.fix_seeds(i_epoch+1)
    
                if (i_epoch + self.eval_offset) % self.eval_epochs == self.eval_epochs-1:
                    print(f"\nBegin eval...")
                    sys.stdout.flush()
                    t0 = time.time()
                    with torch.cuda.amp.autocast():
                        images, scores, segmentations, labels_gt, masks_gt, img_paths = self.predict(val_data)
                    model_er = self._evaluate(images, scores, segmentations,
                                              labels_gt, masks_gt, name,img_paths=img_paths)
                    image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = model_er
                    best_threshold, best_precision, best_recall, best_f1 = self.prf
                    cur_score = self.get_score(pauroc=pixel_auroc,f1=best_f1)
                    self.show_record([image_auroc, best_precision, pixel_auroc, best_recall, best_f1, i_epoch],"model record")
                    with torch.cuda.amp.autocast():
                        images, scores, segmentations, labels_gt, masks_gt, img_paths = self.ema.ema.predict(val_data)
                    model_ema_er =  self.ema.ema._evaluate(images, scores, segmentations,
                                                             labels_gt, masks_gt, name,img_paths=img_paths)
                    image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = model_ema_er
                    best_threshold, best_precision, best_recall, best_f1 = self.ema.ema.prf
                    cur_ema_score = self.get_score(pauroc=pixel_auroc,f1=best_f1)
                    self.show_record([image_auroc, best_precision, pixel_auroc, best_recall, best_f1, i_epoch],"model ema record")
                    if cur_ema_score>cur_score:
                        print(f"use ema ckpt")
                        use_ema_ckpt = True
                        image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = model_ema_er
                        best_threshold, best_precision, best_recall, best_f1 = self.ema.ema.prf
                    else:
                        print(f"Use model ckpt")
                        use_ema_ckpt = False 
                        image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = model_er
                        best_threshold, best_precision, best_recall, best_f1 = self.prf

                    self.logger.logger.add_scalar("i-auroc", image_auroc, i_epoch)
                    self.logger.logger.add_scalar("Recall", best_recall, i_epoch)
                    self.logger.logger.add_scalar("p-auroc", pixel_auroc, i_epoch)
                    self.logger.logger.add_scalar("Precision", best_precision, i_epoch)
                    self.logger.logger.add_scalar("F1", best_f1, i_epoch)
    
    
                    eval_path = osp.join(self.run_save_path,'eval' , name)
                    train_path = osp.join(self.run_save_path,'training' , name)
                    cur_score = self.get_score(pauroc=pixel_auroc,f1=best_f1)

                    if best_record is not None:
                        best_score = self.get_score(pauroc=best_record[2],f1=best_record[4])
                        print(f"\nBest score {best_score}, current score {cur_score}\n")
                        self.show_record(best_record,"best record")

                    self.show_record([image_auroc, best_precision, pixel_auroc, best_recall, best_f1, i_epoch],"current record")

                    if best_record is None:
                        best_record = [image_auroc, best_precision, pixel_auroc, best_recall, best_f1, i_epoch]
                        ckpt_path_best = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
                        if use_ema_ckpt:
                            print(f"Get EMA state dict")
                            state_dict = self.ema.ema.get_state_dict()
                        else:
                            print(f"Get model state dict")
                            state_dict = self.get_state_dict()
                        torch.save(state_dict, ckpt_path_best)
                        shutil.rmtree(eval_path, ignore_errors=True)
                        if osp.exists(train_path):
                            shutil.copytree(train_path, eval_path)
                        
                    elif cur_score>best_score:
                        best_record = [image_auroc, best_precision, pixel_auroc, best_recall, best_f1, i_epoch]
                        os.remove(ckpt_path_best)
                        ckpt_path_best = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
                        if use_ema_ckpt:
                            print(f"Get EMA state dict")
                            state_dict = self.ema.ema.get_state_dict()
                        else:
                            print(f"Get model state dict")
                            state_dict = self.get_state_dict()
                        torch.save(state_dict, ckpt_path_best)
                        shutil.rmtree(eval_path, ignore_errors=True)
                        if osp.exists(train_path):
                            shutil.copytree(train_path, eval_path)
                        try:
                            ckpt_path_best = osp.abspath(ckpt_path_best)
                            sym_path = wmlu.change_name(ckpt_path_best,basename="ckpt_best")
                            wmlu.symlink(ckpt_path_best,sym_path)
                        except:
                            pass
                    elif cur_score<best_score-0.05:
                        best_epoch_n = best_record[-1]
                        if i_epoch>=base_epoch and i_epoch-best_epoch_n>=stop_epoch_nr:
                            sys.stdout.flush()
                            print(f"----------------------------------------------------------------------\n")
                            print(f"----------------------------------------------------------------------\n")
                            print(f"----------------------------------------------------------------------\n")
                            print(f"Ealy stop, best epoch is {best_epoch_n}, current epoch is {i_epoch}")
                            print(f"----------------------------------------------------------------------\n")
                            print(f"----------------------------------------------------------------------\n")
                            print(f"----------------------------------------------------------------------\n")
                            sys.stdout.flush()
                            break


    
                    pbar_str1 = f" IAUC:{round(image_auroc * 100, 2)}({round(best_record[0] * 100, 2)})" \
                                f" PAUC:{round(pixel_auroc * 100, 2)}({round(best_record[2] * 100, 2)})" \
                                f" F1:{round(best_f1* 100, 2)}({round(best_record[4] * 100, 2)})" \
                                f" Precision:{round(best_precision* 100, 2)}({round(best_record[1] * 100, 2)})" \
                                f" Recall:{round(best_recall* 100, 2)}({round(best_record[3] * 100, 2)})" \
                                f" E:{i_epoch}({best_record[-1]})"
                    pbar_str += pbar_str1
                    pbar.set_description_str(pbar_str)
                    print(f"Eval finish, time cost {time.time()-t0:.3f}s\n")
                    sys.stdout.flush()
    
                torch.save(self.get_state_dict(), ckpt_path_save)
                min_error_nr = 0

            except Exception as e:
                torch.cuda.empty_cache()
                error_nr += 1
                min_error_nr += 1
                if min_error_nr>= 5 and osp.exists(ckpt_path_best):
                    try:
                        print(f"Too many errors, reload ckpt {ckpt_path_best}")
                        self.load_ckpt(ckpt_path_best)
                        print(f"Load {ckpt_path_best} success.")
                        min_error_nr = 0
                    except Exception as e:
                        pass
                traceback.print_exc()
                print(colorama.Fore.RED+f"ERROR:{e}"+colorama.Style.RESET_ALL)
        return best_record

    def _train_discriminator_amp(self, input_data, cur_epoch, pbar, pbar_str1):
        #self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()

        all_loss, all_p_true, all_p_fake, all_r_t, all_r_g, all_r_f = [], [], [], [], [], []
        sample_num = 0
        tda_error_nr = 0
        for i_iter, data_item in enumerate(input_data):
            self.dsc_opt.zero_grad()
            if self.pre_proj > 0:
                self.proj_opt.zero_grad()
            if self.train_backbone:
                self.backbone_opt.zero_grad()

            aug = data_item["aug"]
            aug = aug.to(torch.float).to(self.device)
            img = data_item["image"]
            img = img.to(torch.float).to(self.device)
            B = img.shape[0]
            with torch.cuda.amp.autocast():
                if self.pre_proj > 0:
                    fake_feats = self.pre_projection(self._embed(aug, evaluation=False)[0])
                    fake_feats = fake_feats[0] if len(fake_feats) == 2 else fake_feats
                    true_feats = self.pre_projection(self._embed(img, evaluation=False)[0])
                    true_feats = true_feats[0] if len(true_feats) == 2 else true_feats
                else:
                    fake_feats = self._embed(aug, evaluation=False)[0]
                    fake_feats.requires_grad = True
                    true_feats = self._embed(img, evaluation=False)[0]
                    true_feats.requires_grad = True

                mask_s_gt = data_item["mask_s"].reshape(-1, 1).to(self.device)
                noise = torch.normal(0, self.noise, true_feats.shape).to(self.device)
                gaus_feats = true_feats + noise.to(true_feats.dtype)
    
                true_points = torch.concat([fake_feats[mask_s_gt[:, 0] == 0], true_feats], dim=0)
    
                for step in range(self.step + 1):
                    gaus_feats = torch.tensor(gaus_feats.detach(),requires_grad=True)
                    gaus_scores,gaus_logits_scores = self.discriminator(gaus_feats)
                    #true_scores = scores[:len(true_feats)]
                    #true_logits_scores = logits[:len(true_feats)]
                    with torch.cuda.amp.autocast(enabled=False):
                        gaus_loss = torch.nn.BCEWithLogitsLoss()(gaus_logits_scores.float(), torch.ones_like(gaus_scores.float()))
    
                    if step == self.step:
                        break
    
                    grad = torch.autograd.grad(gaus_loss, [gaus_feats])[0].float()
                    grad_norm = torch.norm(grad, dim=1)
                    grad_norm = grad_norm.view(-1, 1)
                    grad_normalized = grad / (grad_norm + 1e-10)
    
                    with torch.no_grad():
                        gaus_feats.add_(0.001 * grad_normalized)
    
                    if (step + 1) % 5 == 0:
                        proj_feats = true_feats
                        r = 0.5
    
                        h = gaus_feats - proj_feats
                        h_norm = torch.norm(h, dim=1)
                        alpha = torch.clamp(h_norm, r, 2 * r)
                        proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
                        h = proj * h
                        gaus_feats = proj_feats + h # gaus_feats将处理离proj_feats, [r,2r]的范围内
    
                if True:
                    true_scores,true_logits_scores = self.discriminator(true_feats)
                    #true_scores = scores[:len(true_feats)]
                    #gaus_scores = scores[len(true_feats):]
                    #true_logits_scores = logits[:len(true_feats)]
                    #gaus_logits_scores = logits[len(true_feats):]
                    with torch.cuda.amp.autocast(enabled=False):
                        true_loss = torch.nn.BCEWithLogitsLoss()(true_logits_scores.float(), torch.zeros_like(true_scores.float()))
                    #bce_loss = true_loss + gaus_loss
    
                fake_scores,fake_logits = self.discriminator(fake_feats)
                if self.p > 0:
                    fake_dist = (fake_scores - mask_s_gt) ** 2
                    d_hard = torch.quantile(fake_dist, q=self.p)
                    #fake_scores_ = fake_scores[fake_dist >= d_hard].unsqueeze(1)  #使用最难预测的1-self.p的数据点
                    #fake_loss_weights = (fake_dist>=d_hard).float()
                    fake_logits_ = fake_logits[fake_dist >= d_hard].unsqueeze(1)
                    mask_ = mask_s_gt[fake_dist >= d_hard].unsqueeze(1)
                else:
                    #fake_scores_ = fake_scores
                    fake_logits_ = fake_logits
                    mask_ = mask_s_gt
                #output = torch.cat([1 - fake_scores_, fake_scores_], dim=1)
                with torch.cuda.amp.autocast(enabled=False):
                    focal_loss = torchvision.ops.sigmoid_focal_loss(inputs=fake_logits_.float(),targets=mask_.float(),alpha=-1,reduction = "mean")*100
                    #focal_loss = wvarifocal_loss(pred=fake_logits_.float(),target=mask_.float(),reduction = "mean")*20
                    #focal_loss = self.focal_loss(output.float(), mask_.float())*10
                
                model_loss = [true_loss,gaus_loss,focal_loss]
                info = ""
                if not torch.isfinite(true_loss):
                    info += f"true loss is infinite,"
                if not torch.isfinite(gaus_loss):
                    info += f"gaus loss is infinite,"
                if not torch.isfinite(focal_loss):
                    info += f"focal loss is infinite,"
                model_loss = list(filter(torch.isfinite,model_loss))
                if len(model_loss) == 0:
                    print(info)
                    tda_error_nr += 1
                    if tda_error_nr >= 5:
                        raise RuntimeError(f"Too many NaN loss error.")
                        return None,0,0
                if len(model_loss)!=3:
                    print(info)

                loss = sum(model_loss)

            self.scaler.scale(loss).backward()
            if self.pre_proj > 0:
                if self.max_norm is not None:
                    self.scaler.unscale_(self.proj_opt) #将梯度都除以self.scaler._scale
                    self.total_norm_proj = torch.nn.utils.clip_grad_norm_(self.pre_projection.parameters(), max_norm=self.max_norm, norm_type=2)
                self.scaler.step(self.proj_opt)
                self.proj_ls.step()
            if self.train_backbone:
                if self.max_norm is not None:
                    self.scaler.unscale_(self.backbone_opt) #将梯度都除以self.scaler._scale
                    self.total_norm_backbone= torch.nn.utils.clip_grad_norm_(self.forward_modules.train_parameters(), max_norm=self.max_norm, norm_type=2)
                self.scaler.step(self.backbone_opt)
                self.backbone_ls.step()

            if self.max_norm is not None:
                self.scaler.unscale_(self.dsc_opt) #将梯度都除以self.scaler._scale
                self.total_norm_dsc = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.max_norm, norm_type=2)
            self.scaler.step(self.dsc_opt)
            self.scaler.update()
            self.dsc_ls.step()
            self.ema.update(self)

            pix_true = torch.concat([fake_scores.detach() * (1 - mask_s_gt), true_scores.detach()])
            pix_fake = torch.concat([fake_scores.detach() * mask_s_gt, gaus_scores.detach()])
            p_true = ((pix_true < self.dsc_margin).sum() - (pix_true == 0).sum()) / ((mask_s_gt == 0).sum() + true_scores.shape[0])
            p_fake = (pix_fake >= self.dsc_margin).sum() / ((mask_s_gt == 1).sum() + gaus_scores.shape[0])

            if self.logger.g_iter%10 == 0:
                self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
                self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)
                self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
                self.logger.logger.add_scalar("focal_loss", focal_loss, self.logger.g_iter)
                self.logger.logger.add_scalar("gaus_loss", gaus_loss, self.logger.g_iter)
                self.logger.logger.add_scalar("true_loss", true_loss, self.logger.g_iter)

            if self.logger.g_iter%200 == 0:
                log_img = wtu.unnormalize(data_item["image"][:3],mean=common.IMAGENET_MEAN*255,std=common.IMAGENET_STD*255)
                self.logger.logger.add_images("input",log_img.to(torch.uint8),self.logger.g_iter)
                log_img = wtu.unnormalize(data_item["aug"][:3],mean=common.IMAGENET_MEAN*255,std=common.IMAGENET_STD*255)
                self.logger.logger.add_images("aug",log_img.to(torch.uint8),self.logger.g_iter)
                log_img = torch.unsqueeze(data_item["mask_s"][:3],1)*200
                self.logger.logger.add_images("mask_s",log_img.to(torch.uint8),self.logger.g_iter)
                self.logger.logger.add_scalar("ema_old_persent", self.ema.ema_persent, self.logger.g_iter)
                wsummary.log_optimizer(self.logger.logger,self.backbone_opt,self.logger.g_iter,name="backbone_opt")
                wsummary.log_optimizer(self.logger.logger,self.dsc_opt,self.logger.g_iter,name="dsc_opt")
                wsummary.log_optimizer(self.logger.logger,self.proj_opt,self.logger.g_iter,name="proj_opt")
                wsummary.log_all_variable(self.logger.logger,self,global_step=self.logger.g_iter)
            self.logger.step()

            all_loss.append(loss.detach().cpu().item())
            all_p_true.append(p_true.cpu().item())
            all_p_fake.append(p_fake.cpu().item())

            all_loss_ = np.mean(all_loss)
            all_p_true_ = np.mean(all_p_true)
            all_p_fake_ = np.mean(all_p_fake)
            sample_num = sample_num + img.shape[0]

            pbar_str = f"{input_data.dataset.classname}: epoch:{cur_epoch} loss:{all_loss_:.2e}"
            pbar_str += f" pt:{all_p_true_ * 100:.2f}"
            pbar_str += f" pf:{all_p_fake_ * 100:.2f}"
            pbar_str += f" svd:{self.svd}"
            pbar_str += f" sample:{sample_num}"
            pbar_str2 = pbar_str
            pbar_str += pbar_str1
            if self.logger.g_iter % 10 == 1:
                pbar.set_description_str(pbar_str)

            tda_error_nr = 0

            if sample_num > self.limit:
                break

        return pbar_str2, all_p_true_, all_p_fake_


    def _tester(self, test_data, name,ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best_*')
            for cp in ckpt_path:
                try:
                    best_precision, best_recall,best_f1, p_auroc, pixel_ap, pixel_pro, epoch = self._tester(test_data,name,cp)
                    print(f"ckpt: {osp.basename(cp)}, M:{(best_f1+p_auroc)/2:.3f}, pixel_auroc: {p_auroc}, Precision: {best_precision}, Recall: {best_recall}, F1: {best_f1}, best_epoch: {epoch}\n" )
                except:
                    pass
        
            return best_precision, best_recall,best_f1, p_auroc, pixel_ap, pixel_pro, epoch

    def tester(self, test_data, name,ckpt_path=None):
        ckpt_path = self.load_ckpt(ckpt_path=ckpt_path)
        if ckpt_path is not None: 
            images, scores, segmentations, labels_gt, masks_gt,img_paths = self.predict(test_data)
            with wmlu.TimeThis("_eval"):
                image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(images, scores, segmentations,
                                                                                     labels_gt, masks_gt, name, path='eval',img_paths=img_paths)
            try:
                epoch = int(ckpt_path.split('_')[-1].split('.')[0])
            except:
                epoch = 0
        else:
            image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro, epoch = 0., 0., 0., 0., 0., -1.
            LOGGER.info("No ckpt file found!")

        best_threshold, best_precision, best_recall, best_f1 = self.prf
        print("PRF:",self.prf)

        utils.update_threshold_file(self.run_save_path,test_data.dataset.classname,best_threshold)


        return best_precision, best_recall,best_f1, pixel_auroc, pixel_ap, pixel_pro, epoch
    
    def load_ckpt(self,ckpt_path=None):
        if ckpt_path is None:
            s_path = osp.join(self.ckpt_dir,"ckpt_best.pth")
            #s_path = osp.join(self.ckpt_dir,"ckpt.pth")
            if osp.exists(s_path):
                ckpt_path = [s_path]
            else:
                ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        else:
            ckpt_path = [ckpt_path]
        if len(ckpt_path) != 0:
            if len(ckpt_path)>0:
                for cp in ckpt_path:
                    wmlu.ls(cp)
            state_dict = torch.load(ckpt_path[0], map_location=self.device)
            print(f"Load {ckpt_path[0]}")
            if 'discriminator' in state_dict:
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)

            if 'forward_modules' in state_dict:
                wtu.forgiving_state_restore(self.forward_modules,state_dict['forward_modules'])


            try:
                if hasattr(self.forward_modules,"att"):
                    #weights = torch.sigmoid(self.forward_modules.att.weights).cpu().detach().numpy().tolist()
                    weights = torch.sigmoid(self.forward_modules.att.weights).cpu().detach().numpy()
                    nr = len(weights)//3
                    print(f"{get_class_name()} forward_module.att weights")
                    wmlu.show_nparray(weights[:nr])
                    wmlu.show_nparray(weights[nr:2*nr])
                    wmlu.show_nparray(weights[2*nr:])
                    print(f"Mean: {np.mean(weights[:nr])}, {np.mean(weights[nr:2*nr])}, {np.mean(weights[2*nr:])}")
            except:
                pass
            
            return ckpt_path[0]
        else:
            return None

    def _evaluate(self, images, scores, segmentations, labels_gt, masks_gt, name, path='training',img_paths=None,vis_results=False):
        scores = np.squeeze(np.array(scores))
        image_scores = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt, path)
        image_auroc = image_scores["auroc"]
        image_ap = image_scores["ap"]

        if len(masks_gt) > 0:
            eval_segmentations = npresize_mask(segmentations,scale_factor=0.25/get_img_cut_nr())
            eval_masks_gt = [np.squeeze(v,axis=0) for v in masks_gt]
            eval_masks_gt = npresize_mask(eval_masks_gt,scale_factor=0.25/get_img_cut_nr())
            best_threshold, best_precision, best_recall, best_f1 = metrics.compute_best_pr_re(eval_masks_gt,eval_segmentations)
            self.prf = best_threshold, best_precision, best_recall, best_f1
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(eval_segmentations, eval_masks_gt, path)
            pixel_auroc = pixel_scores["auroc"]
            pixel_ap = pixel_scores["ap"]
            if path == 'eval':
                try:
                    pixel_pro = metrics.compute_pro(np.squeeze(np.array(eval_masks_gt)), eval_segmentations)
                except:
                    pixel_pro = 0.
            else:
                pixel_pro = 0.
        else:
            print(f"ERROR: len(masks_gt)==0")
            pixel_auroc = -1.
            pixel_ap = -1.
            pixel_pro = -1.
            return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

        if vis_results:
            defects = np.array(images)
            targets = np.array(masks_gt)
            for i in range(len(defects)):
                defect = utils.torch_format_2_numpy_img(defects[i])
                target = utils.torch_format_2_numpy_img(targets[i])
    
                mask = cv2.cvtColor(cv2.resize(segmentations[i], (defect.shape[1], defect.shape[0])),
                                    cv2.COLOR_GRAY2BGR)
                mask = (mask * 255).astype('uint8')
                mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    
                img_up = np.hstack([defect, target, mask])
                #img_up = cv2.resize(img_up, (256 * 3, 256))
                full_path = osp.join(self.run_save_path, path , name)
                if img_paths is not None and len(img_paths) == len(defects):
                    save_name = osp.basename(img_paths[i])
                    full_path = osp.join(full_path,wmlu.base_name(osp.dirname(img_paths[i])))
                else:
                    save_name = str(i + 1).zfill(3) + '.png'
                utils.del_remake_dir(full_path, del_flag=False)
    
                cv2.imwrite(osp.join(full_path , save_name), img_up)

        if hasattr(self,'run_save_path'):
            print(f"Eval save path: {osp.join(self.run_save_path, path)}")

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

    def _fast_evaluate(self, images, scores, segmentations, labels_gt, masks_gt, name, path='training',img_paths=None,vis_results=False):
        scores = np.squeeze(np.array(scores))
        image_auroc = 0
        image_ap = 0

        if len(masks_gt) > 0:
            eval_segmentations = npresize_mask_mt(segmentations,scale_factor=0.25)
            eval_masks_gt = [np.squeeze(v,axis=0) for v in masks_gt]
            eval_masks_gt = npresize_mask_mt(eval_masks_gt,scale_factor=0.25)
            best_threshold, best_precision, best_recall, best_f1 = metrics.compute_best_pr_re(eval_masks_gt,eval_segmentations)
            self.prf = best_threshold, best_precision, best_recall, best_f1
            results = metrics.ader_evaluator(pr_px=eval_segmentations, pr_sp=scores, gt_px=eval_masks_gt, gt_sp=labels_gt)
            pixel_auroc = results["P-AUROC"]
            pixel_ap = results["P-AP"]
            image_auroc= results["I-AUROC"]
            image_ap = results["I-AP"]
            if path == 'eval':
                try:
                    pixel_pro = metrics.compute_pro(np.squeeze(np.array(eval_masks_gt)), eval_segmentations)
                except:
                    pixel_pro = 0.
            else:
                pixel_pro = 0.
        else:
            pixel_auroc = -1.
            pixel_ap = -1.
            pixel_pro = -1.
            return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

        if vis_results:
            defects = np.array(images)
            targets = np.array(masks_gt)
            for i in range(len(defects)):
                defect = utils.torch_format_2_numpy_img(defects[i])
                target = utils.torch_format_2_numpy_img(targets[i])
    
                mask = cv2.cvtColor(cv2.resize(segmentations[i], (defect.shape[1], defect.shape[0])),
                                    cv2.COLOR_GRAY2BGR)
                mask = (mask * 255).astype('uint8')
                mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    
                img_up = np.hstack([defect, target, mask])
                #img_up = cv2.resize(img_up, (256 * 3, 256))
                full_path = osp.join(self.run_save_path, path , name)
                if img_paths is not None and len(img_paths) == len(defects):
                    save_name = osp.basename(img_paths[i])
                    full_path = osp.join(full_path,wmlu.base_name(osp.dirname(img_paths[i])))
                else:
                    save_name = str(i + 1).zfill(3) + '.png'
                utils.del_remake_dir(full_path, del_flag=False)
    
                cv2.imwrite(osp.join(full_path , save_name), img_up)

        print(f"Eval save path: {full_path}")

        return image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro

    def predict(self, test_dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        self.forward_modules.eval()

        img_paths = []
        images = []
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        print(f"Img cut nr in predict {get_img_cut_nr()}")

        with tqdm.tqdm(test_dataloader, desc="Inferring...", leave=False, unit='batch') as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy())
                    if data.get("mask_gt", None) is not None:
                        masks_gt.extend(data["mask_gt"].numpy())
                    image = data["image"]
                    images.extend(image.numpy())
                    img_paths.extend(data["image_path"])
                #with torch.cuda.amp.autocast():
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)

        return images, scores, masks, labels_gt, masks_gt,img_paths

    def _predict(self, img):
        if get_img_cut_nr() == 1:
            return self._raw_predict(img)
        else:
            img = make_path(img,get_img_cut_nr())
            scores,masks = self._raw_predict(img)
            masks,scores = merge_path(masks,get_img_cut_nr(),scores=scores)
            return scores,masks

    def _raw_predict(self, img):
        """Infer score and mask for a batch of images."""
        img = img.to(torch.float).to(self.device)
        self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()

        with torch.no_grad():

            patch_features, patch_shapes = self._embed(img, provide_patch_shapes=True, evaluation=True)
            if self.pre_proj > 0:
                patch_features = self.pre_projection(patch_features)
                patch_features = patch_features[0] if len(patch_features) == 2 else patch_features

            patch_scores = image_scores = self.discriminator(patch_features)[0]
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=img.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(img.shape[0], scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores,target_size=img.shape[-2:])

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=img.shape[0])
            image_scores = self.patch_maker.score(image_scores)
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

        return image_scores, masks

    def run_predict(self, test_data, name):
        np.seterr(all='warn')
        if self.load_ckpt() is not None:
            images, scores, segmentations, labels_gt, masks_gt,img_paths = self.predict(test_data)
            threshold = utils.read_threshold_file(self.run_save_path,test_data.dataset.classname)
            print(f"Use threshold {threshold} for {test_data.dataset.classname}")
            self._save_predict_results(images, scores, segmentations, img_paths,source=test_data.dataset.source,name=name,threshold=threshold)


    def _save_predict_results(self, images, scores, segmentations, img_paths,source,name,threshold=0.4):
        scores = np.squeeze(np.array(scores))

        defects = images
        for i in range(len(defects)):
            defect = utils.torch_format_2_numpy_img(defects[i])
            mask = cv2.cvtColor(cv2.resize(segmentations[i], (defect.shape[1], defect.shape[0])),
                                cv2.COLOR_GRAY2BGR)
            raw_mask = np.array(segmentations[i]).astype(np.float32)
            mask = (mask * 255).astype('uint8')
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            img_up = np.hstack([defect, mask])
            full_path = osp.join(self.run_save_path, "predict", name)
            utils.del_remake_dir(full_path, del_flag=False)
            cv2.imwrite(osp.join(full_path , str(i + 1).zfill(3) + '.png'), img_up)

            ipath = img_paths[i]
            rpath = wmlu.get_relative_path(ipath,source)
            full_path_tiff = osp.join(self.run_save_path+'_tiff', "predict","anomaly_images",rpath)
            full_path_tiff = wmlu.change_suffix(full_path_tiff,"tiff")
            wmlu.make_dir_for_file(full_path_tiff)
            #i_shape = wmli.get_img_size(ipath)
            i_shape = raw_mask.shape
            raw_mask = cv2.resize(raw_mask,(i_shape[1]//(2*get_img_cut_nr()),i_shape[0]//(2*get_img_cut_nr())),interpolation=cv2.INTER_LINEAR)
            raw_mask_fp16 = raw_mask.astype(np.float16)
            #raw_mask = raw_mask.astype(np.float16)
            tifffile.imwrite(full_path_tiff,raw_mask_fp16)

            full_path_png = osp.join(self.run_save_path+'_tiff', "predict","anomaly_images_thresholded",rpath)
            full_path_png = wmlu.change_suffix(full_path_png,"png")

            raw_mask_thr = ((raw_mask>threshold)*255).astype(np.uint8)
            wmlu.make_dir_for_file(full_path_png)
            cv2.imwrite(full_path_png,raw_mask_thr)

        print(f"\nPredict run save path {self.run_save_path}\n{self.run_save_path+'_tiff'}")

    def show_record(self,record,info=""):
        image_auroc, best_precision, p_auroc, best_recall, best_f1, epoch = record
        score = self.get_score(pauroc=record[2],f1=record[4])
        print(f"{get_class_name()}: {info}: M:{score:.3f}, pixel_auroc: {p_auroc:.3f}, Precision: {best_precision:.3f}, Recall: {best_recall:.3f}, F1: {best_f1:.3f}, best_epoch: {epoch}\n" )


    def get_state_dict(self):
        state_dict = {}
        state_dict["discriminator"] = OrderedDict({
            k: v.detach().cpu()
            for k, v in self.discriminator.state_dict().items()})
        state_dict["forward_modules"] = OrderedDict({
            k: v.detach().cpu()
            for k, v in self.forward_modules.state_dict().items()})
        state_dict["pre_projection"] = OrderedDict({
            k: v.detach().cpu()
            for k, v in self.pre_projection.state_dict().items()})
        
        return state_dict
