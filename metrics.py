from sklearn import metrics
from skimage import measure
from sklearn.metrics import roc_auc_score,  precision_recall_curve, average_precision_score
from wml.object_detection2.metrics.classifier_toolkit import precision_recall_curve
import cv2
import numpy as np
import pandas as pd
import torch
from itertools import groupby

def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()

def get_sample_id(path):
    return int(path.split(".")[0])

def ader_evaluator(pr_px, pr_sp, gt_px, gt_sp, use_metrics = ['I-AUROC', 'I-AP', 'I-F1_max','P-AUROC', 'P-AP', 'P-F1_max', 'AUPRO']):
    if len(gt_px.shape) == 4:
        gt_px = gt_px.squeeze(1)
    if len(pr_px.shape) == 4:
        pr_px = pr_px.squeeze(1)
        
    score_min = min(min(pr_sp),min(gt_sp))
    score_max = max(max(pr_sp),max(gt_sp))
    anomap_min = pr_px.min()
    anomap_max = pr_px.max()
    
    accum = EvalAccumulatorCuda(score_min, score_max, anomap_min, anomap_max, skip_pixel_aupro=False, nstrips=200)
    accum.add_anomap_batch(torch.tensor(pr_px).cuda(non_blocking=True),
                           torch.tensor(gt_px.astype(np.uint8)).cuda(non_blocking=True))
    
    accum.add_image(torch.tensor(pr_sp), torch.tensor(gt_sp))
    
    metrics = accum.summary()
    metric_results = {}
    for metric in use_metrics:
        if metric.startswith('I-AUROC'):
            auroc_sp = roc_auc_score(gt_sp, pr_sp)
            metric_results[metric] = auroc_sp
        elif metric.startswith('I-AP'):
            ap_sp = average_precision_score(gt_sp, pr_sp)
            metric_results[metric] = ap_sp
        elif metric.startswith('I-F1_max'):
            best_f1_score_sp = f1_score_max(gt_sp, pr_sp)
            metric_results[metric] = best_f1_score_sp
        elif metric.startswith('P-AUROC'):
            metric_results[metric] = metrics['p_auroc']
        elif metric.startswith('P-AP'):
            metric_results[metric] = metrics['p_aupr']
        elif metric.startswith('P-F1_max'):
            best_f1_score_px = f1_score_max(gt_px.ravel(), pr_px.ravel())
            metric_results[metric] = best_f1_score_px
        elif metric.startswith('AUPRO'):
            metric_results[metric] = metrics['p_aupro']
    return list(metric_results.values())


def compute_best_pr_re(anomaly_ground_truth_labels, anomaly_prediction_weights):
    """
    Computes the best precision, recall and threshold for a given set of
    anomaly ground truth labels and anomaly prediction weights.
    """
    precision, recall, f1_scores,thresholds = precision_recall_curve(anomaly_prediction_weights,anomaly_ground_truth_labels.astype(np.int32),
        thresholds=np.linspace(0.001,0.999,100))
    idx = np.argmax(f1_scores)
    best_threshold = thresholds[idx]
    best_precision = precision[idx]
    best_recall = recall[idx]
    if best_threshold<=0.1:
        precision, recall, f1_scores,thresholds = precision_recall_curve(anomaly_prediction_weights,anomaly_ground_truth_labels.astype(np.int32),
        thresholds=np.linspace(0.001,0.1,20))
        idx = np.argmax(f1_scores)
        best_threshold = thresholds[idx]
        best_precision = precision[idx]
        best_recall = recall[idx]
    elif best_threshold>=0.9:
        precision, recall, f1_scores,thresholds = precision_recall_curve(anomaly_prediction_weights,anomaly_ground_truth_labels.astype(np.int32),
        thresholds=np.linspace(0.9,1.0,20))
        idx = np.argmax(f1_scores)
        best_threshold = thresholds[idx]
        best_precision = precision[idx]
        best_recall = recall[idx]
    print(best_threshold, best_precision, best_recall)

    return best_threshold, best_precision, best_recall, f1_scores[idx]


def compute_imagewise_retrieval_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels, path='training'):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).
    """
    auroc = metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    ap = 0. if path == 'training' else metrics.average_precision_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    return {"auroc": auroc, "ap": ap}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks, path='train'):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    auroc = metrics.roc_auc_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)
    ap = 0. if path == 'training' else metrics.average_precision_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)

    return {"auroc": auroc, "ap": ap}


def compute_pro(masks, amaps, num_th=200):
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            binary_amap = cv2.dilate(binary_amap.astype(np.uint8), k)
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df, pd.DataFrame({"pro": np.mean(pros), "fpr": fpr, "threshold": th}, index=[0])])

    df = df[df["fpr"] < 0.3]
    df["fpr"] = (df["fpr"] - df["fpr"].min()) / (df["fpr"].max() - df["fpr"].min() + 1e-10)

    pro_auc = metrics.auc(df["fpr"], df["pro"])
    return pro_auc
