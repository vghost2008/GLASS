import numpy as np
import cv2
import wml.semantic.mask_utils as smu
from PIL import Image

def mask2range(mask,limit):
    mmean,mstd,mmin,mmax,imean,istd,imin,imax = limit
    contours,hierarchy = smu.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cont) for cont in contours]

    new_contours = []
    for a,cont in zip(areas,contours):
        if a<=1:
            continue
        if a<imin or a>imax:
            s = np.random.normal(imean,istd)
            if s<imin:
                s = imin
            elif s>imax:
                s=imax
            scale = s/a
            center = np.mean(cont,axis=0,keepdims=True)
            old_type = cont.dtype
            cont = ((cont-center).astype(np.float32)*scale).astype(old_type)+center
            cont[:,0] = np.clip(cont[:,0],a_min=0,a_max=mask.shape[1]-1)
            cont[:,1] = np.clip(cont[:,1],a_min=0,a_max=mask.shape[1]-1)
            new_contours.append(np.array(cont))
        else:
            new_contours.append(cont)

    res = np.zeros_like(mask)
    res = cv2.drawContours(res,new_contours,-1,color=(255), thickness=cv2.FILLED)
    return res

def adjust_mask(mask,limit):
    mmean,mstd,mmin,mmax,imean,istd,imin,imax = limit

    raw_mask = mask
    mask = np.array(mask)

    area = np.sum(mask>0)

    if area<mmin or area>mmax:
        mask = mask2range(mask,limit)
        return Image.fromarray(mask)
    else:
        return raw_mask

