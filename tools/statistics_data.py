import cv2
import numpy as np
from PIL import Image
import wml.wml_utils as wmlu
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="build gif")
    parser.add_argument("src_dir",type=str,help="src dir")
    args = parser.parse_args()
    return args

def read_file(path):
    mask = Image.open(path)
    mask = np.array(mask)
    contours,hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cont) for cont in contours]
    return areas

def statistics_info(data):
    max_ = np.max(data)
    min_ = np.min(data)
    mean = np.mean(data)
    std = np.std(data)

    return f"{mean:.2f}, {std:.2f}, {min_:.2f}, {max_:.2f}"



def statistics_one_dir(dir_path):
    files = wmlu.get_files(dir_path,suffix=".png")
    def is_gt(path):
        return "ground_truth" in path

    files = filter(is_gt,files)
    img_level_areas = []
    instance_level_areas = []
    for f in files:
        areas = read_file(f)
        i_area = np.sum(areas)
        img_level_areas.append(i_area)
        instance_level_areas.extend(areas)
    
    
    print(f"{wmlu.base_name(dir_path)}:[{statistics_info(img_level_areas)},{statistics_info(instance_level_areas)}]")

def statistics(path):
    sub_dirs = wmlu.get_subdir_in_dir(path,absolute_path=True)
    for sd in sub_dirs:
        statistics_one_dir(sd)
    


if __name__ == "__main__":
    args = parse_args()
    statistics(args.src_dir)


