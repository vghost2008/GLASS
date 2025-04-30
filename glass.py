from loss import FocalLoss
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
import wml.img_utils as wmli
import colorama
import time
from wml.semantic.mask_utils import npresize_mask,resize_mask,npresize_mask_mt

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
            **kwargs,
    ):

        train_backbone = True
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device

        if hasattr(self.backbone,"aggregator") and self.backbone.aggregator == "NetworkFeatureAggregatorV3":
            feature_aggregator = common.NetworkFeatureAggregatorV3(
                self.backbone, self.layers_to_extract_from, self.device, train_backbone
            ).to(device)
        elif hasattr(self.backbone,"aggregator") and self.backbone.aggregator == "NetworkFeatureAggregatorV4":
            feature_aggregator = common.NetworkFeatureAggregatorV4(
                self.backbone, self.layers_to_extract_from, self.device, train_backbone
            ).to(device)
        else:
            feature_aggregator = common.NetworkFeatureAggregatorV2(
                self.backbone, self.layers_to_extract_from, self.device, train_backbone
            ).to(device)
        self.forward_modules = feature_aggregator

        self.target_embed_dimension = target_embed_dimension

        self.meta_epochs = meta_epochs
        self.lr = lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            #self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)
            self.backbone_opt = self.get_embed_optim()

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.Adam(self.pre_projection.parameters(), lr, weight_decay=1e-5)

        self.eval_epochs = eval_epochs
        self.eval_offset = 0
        if self.eval_epochs<=2:
            print(f"WARNING: eval epochs={self.eval_epochs}")
        else:
            self.eval_offset = np.random.randint(1,self.eval_epochs)
            print(f"INFO: eval epochs={self.eval_epochs}, eval offset={self.eval_offset}")
        self.dsc_layers = dsc_layers
        self.dsc_hidden = dsc_hidden
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=lr * 2, weight_decay=1e-4)
        self.dsc_margin = dsc_margin

        self.c = torch.tensor(0)
        self.c_ = torch.tensor(0)
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
        state_dict = {}
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

        def update_state_dict():
            state_dict["discriminator"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.discriminator.state_dict().items()})
            state_dict["forward_modules"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.forward_modules.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.pre_projection.state_dict().items()})

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

        # judge by image-level spectrogram analysis
        if self.distribution == 1:
            self.forward_modules.eval()
            with torch.no_grad():
                for i, data in enumerate(base_training_data):
                    img = data["image"]
                    img = img.to(torch.float).to(self.device)
                    batch_mean = torch.mean(img, dim=0)
                    if i == 0:
                        self.c = batch_mean
                    else:
                        self.c += batch_mean
                self.c /= len(base_training_data)

            avg_img = utils.torch_format_2_numpy_img(self.c.detach().cpu().numpy())
            self.svd = utils.distribution_judge(avg_img, name)
            os.makedirs(osp.join(self.run_save_path,f'judge/avg/{self.svd}'), exist_ok=True)
            cv2.imwrite(osp.join(self.run_save_path,f'judge/avg/{self.svd}/{name}.png'), avg_img)
            return self.svd

        print(f"Model info:")
        wtt.show_model_parameters_info(self)
        pbar = tqdm.tqdm(range(self.meta_epochs), unit='epoch')
        pbar_str1 = ""
        best_record = None
        error_nr = 0
        min_error_nr = 0
        for i_epoch in pbar:
            try:
                self.forward_modules.eval()
                with torch.cuda.amp.autocast():
                    with torch.no_grad():  # compute center
                        print(f"Get center {i_epoch} ...")
                        gct0 = time.time()
                        sys.stdout.flush()
                        for i, data in enumerate(base_training_data):
                            img = data["image"]
                            img = img.to(torch.float).to(self.device)
                            if self.pre_proj > 0: #True
                                outputs = self.pre_projection(self._embed(img, evaluation=False)[0])
                                outputs = outputs[0] if len(outputs) == 2 else outputs
                            else:
                                outputs = self._embed(img, evaluation=False)[0]
                            outputs = outputs[0] if len(outputs) == 2 else outputs
                            outputs = outputs.reshape(img.shape[0], -1, outputs.shape[-1])
        
                            batch_mean = torch.mean(outputs, dim=0)
                            if i == 0:
                                self.c = batch_mean
                            else:
                                self.c += batch_mean
                        self.c /= len(base_training_data)
                        print(f"Get center finish, time cost {time.time()-gct0:.3f}s.")
                        sys.stdout.flush()
    
                pbar_str, pt, pf = self._train_discriminator_amp(training_data, i_epoch, pbar, pbar_str1)
    
                update_state_dict()
    
                ckpt_path_cur = os.path.join(self.ckpt_dir, "cur_ckpt.pth".format(i_epoch))
                try:
                    if i_epoch%2 == 0:
                        if osp.exists(ckpt_path_cur):
                            os.remove(ckpt_path_cur)
                        torch.save(state_dict, ckpt_path_cur)
                except Exception as e:
                    print(f"ERROR: {e}")
    
                if (i_epoch + self.eval_offset) % self.eval_epochs == 0:
                    print(f"\nBegin eval...")
                    sys.stdout.flush()
                    t0 = time.time()
                    images, scores, segmentations, labels_gt, masks_gt, img_paths = self.predict(val_data)
                    image_auroc, image_ap, pixel_auroc, pixel_ap, pixel_pro = self._evaluate(images, scores, segmentations,
                                                                                             labels_gt, masks_gt, name,img_paths=img_paths)
    
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
                        torch.save(state_dict, ckpt_path_best)
                        shutil.rmtree(eval_path, ignore_errors=True)
                        if osp.exists(train_path):
                            shutil.copytree(train_path, eval_path)
                        
                    elif cur_score>best_score:
                        best_record = [image_auroc, best_precision, pixel_auroc, best_recall, best_f1, i_epoch]
                        os.remove(ckpt_path_best)
                        ckpt_path_best = os.path.join(self.ckpt_dir, "ckpt_best_{}.pth".format(i_epoch))
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
    
                torch.save(state_dict, ckpt_path_save)
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
    
                center = self.c.repeat(img.shape[0], 1, 1)
                center = center.reshape(-1, center.shape[-1])
                true_points = torch.concat([fake_feats[mask_s_gt[:, 0] == 0], true_feats], dim=0)
                c_t_points = torch.concat([center[mask_s_gt[:, 0] == 0], center], dim=0)
                dist_t = torch.norm(true_points - c_t_points, dim=1)
                r_t = torch.tensor([torch.quantile(dist_t, q=self.radius)]).to(self.device)
    
                for step in range(self.step + 1):
                    gaus_feats = torch.tensor(gaus_feats.detach(),requires_grad=True)
                    gaus_scores,gaus_logits_scores = self.discriminator(gaus_feats)
                    #true_scores = scores[:len(true_feats)]
                    #true_logits_scores = logits[:len(true_feats)]
                    with torch.cuda.amp.autocast(enabled=False):
                        gaus_loss = torch.nn.BCEWithLogitsLoss()(gaus_logits_scores.float(), torch.ones_like(gaus_scores.float()))
    
                    if step == self.step:
                        break
                    elif self.mining == 0:
                        dist_g = torch.norm(gaus_feats - center, dim=1)
                        r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device)
                        break
    
                    grad = torch.autograd.grad(gaus_loss, [gaus_feats])[0].float()
                    grad_norm = torch.norm(grad, dim=1)
                    grad_norm = grad_norm.view(-1, 1)
                    grad_normalized = grad / (grad_norm + 1e-10)
    
                    with torch.no_grad():
                        gaus_feats.add_(0.001 * grad_normalized)
    
                    if (step + 1) % 5 == 0:
                        dist_g = torch.norm(gaus_feats - center, dim=1)
                        r_g = torch.tensor([torch.quantile(dist_g, q=self.radius)]).to(self.device)
                        proj_feats = center if self.svd == 1 else true_feats
                        r = r_t if self.svd == 1 else 0.5
    
                        h = gaus_feats - proj_feats
                        h_norm = dist_g if self.svd == 1 else torch.norm(h, dim=1)
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
    
                fake_points = fake_feats[mask_s_gt[:, 0] == 1]
                true_points = true_feats[mask_s_gt[:, 0] == 1]
                c_f_points = center[mask_s_gt[:, 0] == 1]
                dist_f = torch.norm(fake_points - c_f_points, dim=1)
                r_f = torch.tensor([torch.quantile(dist_f, q=self.radius)]).to(self.device)
                proj_feats = c_f_points if self.svd == 1 else true_points
                r = r_t if self.svd == 1 else 1
    
                if self.svd == 1:
                    h = fake_points - proj_feats
                    h_norm = dist_f if self.svd == 1 else torch.norm(h, dim=1)
                    alpha = torch.clamp(h_norm, 2 * r, 4 * r)
                    proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
                    h = proj * h
                    fake_points = proj_feats + h
                    fake_feats[mask_s_gt[:, 0] == 1] = fake_points.to(fake_feats.dtype)
    
                fake_scores,_ = self.discriminator(fake_feats)
                if self.p > 0:
                    fake_dist = (fake_scores - mask_s_gt) ** 2
                    d_hard = torch.quantile(fake_dist, q=self.p)
                    fake_scores_ = fake_scores[fake_dist >= d_hard].unsqueeze(1)  #使用最难预测的1-self.p的数据点
                    mask_ = mask_s_gt[fake_dist >= d_hard].unsqueeze(1)
                else:
                    fake_scores_ = fake_scores
                    mask_ = mask_s_gt
                output = torch.cat([1 - fake_scores_, fake_scores_], dim=1)
                with torch.cuda.amp.autocast(enabled=False):
                    focal_loss = self.focal_loss(output.float(), mask_.float())*10
                
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
            if self.train_backbone:
                if self.max_norm is not None:
                    self.scaler.unscale_(self.backbone_opt) #将梯度都除以self.scaler._scale
                    self.total_norm_backbone= torch.nn.utils.clip_grad_norm_(self.forward_modules.train_parameters(), max_norm=self.max_norm, norm_type=2)
                self.scaler.step(self.backbone_opt)

            if self.max_norm is not None:
                self.scaler.unscale_(self.dsc_opt) #将梯度都除以self.scaler._scale
                self.total_norm_dsc = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=self.max_norm, norm_type=2)
            self.scaler.step(self.dsc_opt)
            self.scaler.update()

            pix_true = torch.concat([fake_scores.detach() * (1 - mask_s_gt), true_scores.detach()])
            pix_fake = torch.concat([fake_scores.detach() * mask_s_gt, gaus_scores.detach()])
            p_true = ((pix_true < self.dsc_margin).sum() - (pix_true == 0).sum()) / ((mask_s_gt == 0).sum() + true_scores.shape[0])
            p_fake = (pix_fake >= self.dsc_margin).sum() / ((mask_s_gt == 1).sum() + gaus_scores.shape[0])

            if self.logger.g_iter%10 == 0:
                self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
                self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)
                self.logger.logger.add_scalar(f"r_t", r_t, self.logger.g_iter)
                self.logger.logger.add_scalar(f"r_g", r_g, self.logger.g_iter)
                self.logger.logger.add_scalar(f"r_f", r_f, self.logger.g_iter)
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
                wsummary.log_all_variable(self.logger.logger,self,global_step=self.logger.g_iter)
            self.logger.step()

            all_loss.append(loss.detach().cpu().item())
            all_p_true.append(p_true.cpu().item())
            all_p_fake.append(p_fake.cpu().item())
            all_r_t.append(r_t.cpu().item())
            all_r_g.append(r_g.cpu().item())
            all_r_f.append(r_f.cpu().item())

            all_loss_ = np.mean(all_loss)
            all_p_true_ = np.mean(all_p_true)
            all_p_fake_ = np.mean(all_p_fake)
            all_r_t_ = np.mean(all_r_t)
            all_r_g_ = np.mean(all_r_g)
            all_r_f_ = np.mean(all_r_f)
            sample_num = sample_num + img.shape[0]

            pbar_str = f"{input_data.dataset.classname}: epoch:{cur_epoch} loss:{all_loss_:.2e}"
            pbar_str += f" pt:{all_p_true_ * 100:.2f}"
            pbar_str += f" pf:{all_p_fake_ * 100:.2f}"
            pbar_str += f" rt:{all_r_t_:.2f}"
            pbar_str += f" rg:{all_r_g_:.2f}"
            pbar_str += f" rf:{all_r_f_:.2f}"
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
                self.forward_modules.load_state_dict(state_dict['forward_modules'])
            
            return ckpt_path[0]
        else:
            return None

    def _evaluate(self, images, scores, segmentations, labels_gt, masks_gt, name, path='training',img_paths=None,vis_results=False):
        scores = np.squeeze(np.array(scores))
        image_scores = metrics.compute_imagewise_retrieval_metrics(scores, labels_gt, path)
        image_auroc = image_scores["auroc"]
        image_ap = image_scores["ap"]

        if len(masks_gt) > 0:
            eval_segmentations = npresize_mask(segmentations,scale_factor=0.25)
            eval_masks_gt = [np.squeeze(v,axis=0) for v in masks_gt]
            eval_masks_gt = npresize_mask(eval_masks_gt,scale_factor=0.25)
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

        with tqdm.tqdm(test_dataloader, desc="Inferring...", leave=False, unit='batch') as data_iterator:
            for data in data_iterator:
                if isinstance(data, dict):
                    labels_gt.extend(data["is_anomaly"].numpy())
                    if data.get("mask_gt", None) is not None:
                        masks_gt.extend(data["mask_gt"].numpy())
                    image = data["image"]
                    images.extend(image.numpy())
                    img_paths.extend(data["image_path"])
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)

        return images, scores, masks, labels_gt, masks_gt,img_paths

    def _predict(self, img):
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

        return list(image_scores), list(masks)

    def run_predict(self, test_data, name):
        np.seterr(all='warn')
        if self.load_ckpt() is not None:
            images, scores, segmentations, labels_gt, masks_gt,img_paths = self.predict(test_data)
            threshold = utils.read_threshold_file(self.run_save_path,test_data.dataset.classname)
            print(f"Use threshold {threshold} for {test_data.dataset.classname}")
            self._save_predict_results(images, scores, segmentations, img_paths,source=test_data.dataset.source,name=name,threshold=threshold)


    def _save_predict_results(self, images, scores, segmentations, img_paths,source,name,threshold=0.4):
        scores = np.squeeze(np.array(scores))

        defects = np.array(images)
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
            raw_mask = cv2.resize(raw_mask,(i_shape[1]//2,i_shape[0]//2),interpolation=cv2.INTER_LINEAR)
            raw_mask_fp16 = raw_mask.astype(np.float16)
            #raw_mask = raw_mask.astype(np.float16)
            tifffile.imwrite(full_path_tiff,raw_mask_fp16)

            full_path_png = osp.join(self.run_save_path+'_tiff', "predict","anomaly_images_thresholded",rpath)
            full_path_png = wmlu.change_suffix(full_path_png,"png")

            raw_mask_thr = ((raw_mask>threshold)*255).astype(np.uint8)
            wmlu.make_dir_for_file(full_path_png)
            cv2.imwrite(full_path_png,raw_mask_thr)

        print(f"Predict run save path {self.run_save_path}/{self.run_save_path+'_tiff'}")

    def show_record(self,record,info=""):
        image_auroc, best_precision, p_auroc, best_recall, best_f1, epoch = record
        score = self.get_score(pauroc=record[2],f1=record[4])
        print(f"{info}: M:{score:.3f}, pixel_auroc: {p_auroc}, Precision: {best_precision}, Recall: {best_recall}, F1: {best_f1}, best_epoch: {epoch}\n" )