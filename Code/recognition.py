from typing import Tuple
import segmentation
import cv2
import numpy as np
import character_extraction
import torch

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def extract_labels(image_path: str) -> Tuple[int, int, list]: # 返回车牌省份，车牌字母，车牌六位号码
    image_path = image_path.split("/")[-1] # 获取文件名
    image_path = image_path.split(".")[0] # 去除后缀名
    image_path = image_path.split("-")[4] # 分割字符串，获取标签们
    labels_index = int(image_path.split("_")) # 获取标签
    province = labels_index[0] # 省份
    alphabet = labels_index[1] # 字母
    ad = labels_index[2: len(labels_index)] # 后面的六位数字
    return province, alphabet, ad

# 训练
def province_recognition_training(image: cv2.Mat, label_index: int) -> torch.

# 识别
def province_recognition(image: cv2.Mat) -> str: # 识别并返回车牌省份

def test():
    image_path: str = "Data/train/015-91_91-282&460_498&530-496&530_282&526_282&460_498&466-0_0_3_27_29_25_30_33-149-96.jpg"
    image: cv2.Mat = segmentation.pipeline(image_path)
    character_images: list = character_extraction.pipeline(image)


test()
