import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
import wml.img_utils as wmli
import wml.wml_utils as wmlu
import os.path as osp
import numpy as np
import math 
import cv2
import copy

def walk_in_img(pos,img,mask,mark,thr=8):
    np_pos = [pos]

    while len(np_pos)>0:
        p_pos = copy.deepcopy(np_pos)
        np_pos = []
        for pos in p_pos:
            v = img[pos[1],pos[0]]
            for dx in [-1,0,1]:
                for dy in [-1,0,1]:
                    if dx==0 and dy==0:
                        continue
                    x = pos[0]+dx
                    y = pos[1]+dy
        
        
                    if x<0 or y<0 or x>=img.shape[1] or y>=img.shape[0]:
                        continue
    
                    if mark[y,x]:
                        continue
                    mark[y,x] = True
                    if math.fabs(v-img[y,x])<thr:
                        mask[y,x] = 255
                        np_pos.append([x,y])
    

def get_vial_mask_for_file(f):
    img = wmli.imread(f)
    img = wmli.nprgb_to_gray(img)
    o_shape = img.shape
    img = wmli.resize_img(img,(o_shape[1]//4,o_shape[0]//4))
    s_0 = (25,25)
    s_1 = (25,img.shape[0]-25)
    mask = np.zeros_like(img,dtype=np.uint8)
    mark = np.zeros_like(img,dtype=bool)
    mask[s_0[1],s_0[0]] = 255
    mask[s_1[1],s_1[0]] = 255
    mark[s_0[1],s_0[0]] = True
    mark[s_1[1],s_1[0]] = True

    img = img.astype(np.int32)

    walk_in_img(s_0,img,mask,mark)
    walk_in_img(s_1,img,mask,mark)

    mask = wmli.resize_img(mask,[o_shape[1],o_shape[0]],interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    mask = np.ones_like(mask)*255-mask

    _contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    for cont in _contours:
        if cv2.contourArea(cont) < 100:
            continue
        contours.append(cont)
    
    mask = np.zeros_like(mask)
    mask = cv2.drawContours(mask,contours,-1,color=(255),thickness=cv2.FILLED)

    return mask



def gen_vial_mask(idir,odir):

    files = wmlu.get_img_files(idir)
    files = list(filter(lambda x:"test" not in x,files))
    for f in files:
        r_path = wmlu.get_relative_path(f,idir)
        s_path = osp.join(odir,r_path)
        s_path = wmlu.change_suffix(s_path,"png")
        mask = get_vial_mask_for_file(f)
        wmlu.make_dir_for_file(s_path)
        wmli.imwrite(s_path,mask)
        s_path = wmlu.change_suffix(s_path,"jpg")
        wmli.read_and_write_img(f,s_path)
        print(f"Save {s_path}")

def get_wallplugs_mask(f):
    img = wmli.imread(f)
    img = wmli.nprgb_to_gray(img)
    o_shape = img.shape
    img = wmli.resize_img(img,(o_shape[1]//4,o_shape[0]//4))
    mask = cv2.threshold(img,thresh=22,maxval=255,type=cv2.THRESH_BINARY)[1]
    mask = wmli.resize_img(mask,[o_shape[1],o_shape[0]],interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    _contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    for cont in _contours:
        if cv2.contourArea(cont) < 2000:
            continue
        contours.append(cont)
    
    mask = np.zeros_like(mask)
    mask = cv2.drawContours(mask,contours,-1,color=(255),thickness=cv2.FILLED)

    return mask

def gen_wallplugs_mask(idir,odir):

    files = wmlu.get_img_files(idir)
    files = list(filter(lambda x:"test" not in x,files))
    for f in files:
        r_path = wmlu.get_relative_path(f,idir)
        s_path = osp.join(odir,r_path)
        s_path = wmlu.change_suffix(s_path,"png")
        mask = get_wallplugs_mask(f)
        wmlu.make_dir_for_file(s_path)
        wmli.imwrite(s_path,mask)
        s_path = wmlu.change_suffix(s_path,"jpg")
        wmli.read_and_write_img(f,s_path)
        print(f"Save {s_path}")


if __name__ == "__main__":
    #gen_vial_mask("/home/wj/ai/mldata1/MVTEC/datasets/vial","/home/wj/ai/mldata1/MVTEC/datasets/fg_mask/vial")
    gen_wallplugs_mask("/home/wj/ai/mldata1/MVTEC/datasets/wallplugs","/home/wj/ai/mldata1/MVTEC/datasets/fg_mask/wallplugs")
