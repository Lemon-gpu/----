import os
from tkinter.ttk import LabeledScale
import cv2
import character_extraction as ce
import segmentation as seg
import pandas as pd

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"] # 34个
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O'] # 25个
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O'] # 35个

def extract_labels(image_path: str) -> list[int, int, list]: # 返回车牌省份，车牌字母，车牌六位号码
    image_path = image_path.split("/")[-1] # 获取文件名
    image_path = image_path.split(".")[0] # 去除后缀名
    image_path = image_path.split("-")[4] # 分割字符串，获取标签们
    labels_index = image_path.split("_") # 获取标签
    province = labels_index[0] # 省份
    alphabet = labels_index[1] # 字母
    ad = labels_index[2: len(labels_index)] # 后面的六位数字
    return [province, alphabet, ad]

def data_preprocess(img_dir: str, save_dir: str) -> None: # 图片预处理，并保存相关的csv文件
    province_index: int = 0 # 省份索引
    alphabet_index: int = 0 # 字母索引
    ad_index: int = 0 # 六位数字索引

    provinces_list: list = [] # 省份列表
    alphabets_list: list = [] # 字母列表
    ads_list: list = [] # 六位数字列表

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(save_dir + "/provinces"):
        os.makedirs(save_dir + "/provinces")
    
    if not os.path.exists(save_dir + "/alphabets"):
        os.makedirs(save_dir + "/alphabets")

    if not os.path.exists(save_dir + "/ads"):
        os.makedirs(save_dir + "/ads")

    # 遍历文件夹
    for root, dirs, files in os.walk(img_dir): # root: 当前目录，dirs: 当前目录下的子目录，files: 当前目录下的所有文件
        for file in files:
            path = os.path.join(root, file) # 拼合图片路径
            image: cv2.Mat = seg.pipeline(path) # 获得车牌图片
            try:
                character_images: list = ce.pipeline(image) # 获得字符图片
                provinces = character_images[0][0] # 获得省份图片
                alphabets = character_images[1][0] # 获得字母图片
                ads = character_images[2][0] # 获得六位数字图片

                # 将图片保存到文件夹中，并获取标签，并保存到相应的列表中，方便后面的处理
                # 省份
                provinces_path = save_dir + "/provinces/" + str(extract_labels(file)[0]) + "_" + str(province_index) + ".jpg" # 省份图片路径
                cv2.imwrite(provinces_path, provinces) # 保存省份图片
                provinces_list.append([str(extract_labels(file)[0]) + "_" + str(province_index) + ".jpg", str(extract_labels(file)[0])]) # 将省份图片路径和省份标签添加到省份列表
                province_index += 1 # 省份索引加1

                # 字母
                alphabets_path = save_dir + "/alphabets/" + str(extract_labels(file)[1]) + "_" + str(alphabet_index) + ".jpg" # 字母图片路径
                cv2.imwrite(alphabets_path, alphabets) # 保存字母图片
                alphabets_list.append([str(extract_labels(file)[1]) + "_" + str(alphabet_index) + ".jpg", str(extract_labels(file)[1])]) # 将字母图片路径和字母标签添加到字母列表
                alphabet_index += 1 # 字母索引加1

                # 六位数字
                for ad in ads:
                    ads_path = save_dir + "/ads/" + str(extract_labels(file)[2][ad_index % 6]) + "_" + str(ad_index) + ".jpg"
                    cv2.imwrite(ads_path, ad)
                    ads_list.append([str(extract_labels(file)[2][ad_index % 6]) + "_" + str(ad_index) + ".jpg", str(extract_labels(file)[2][ad_index % 6])])
                    ad_index += 1

            except Exception as e:
                # print(e)
                continue

    # 将省份列表、字母列表、六位数字列表保存到csv文件中
    # 省份
    provinces_list = pd.DataFrame(provinces_list, columns=["path", "label"]) # 省份列表转为DataFrame
    provinces_list.to_csv(save_dir + "/provinces.csv", index=False) # 将省份列表保存到csv文件中

    # 字母
    alphabets_list = pd.DataFrame(alphabets_list, columns=["path", "label"]) # 字母列表转为DataFrame
    alphabets_list.to_csv(save_dir + "/alphabets.csv", index=False) # 将字母列表保存到csv文件中

    # 六位数字
    ads_list = pd.DataFrame(ads_list, columns=["path", "label"]) # 六位数字列表转为DataFrame
    ads_list.to_csv(save_dir + "/ads.csv", index=False) # 将六位数字列表保存到csv文件中

def module_test():
    img_dir = 'Data/val'
    save_dir = 'Real_data/val'
    data_preprocess(img_dir, save_dir)
    print("done")

def pipeline(img_dir, save_dir):
    data_preprocess(img_dir, save_dir)
    print("done")



