import os
import cv2
import character_extraction as ce
import segmentation as seg

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

def data_preprocess(img_dir: str, save_dir: str) -> None: # 图片预处理
    province_index: int = 0 # 省份索引
    alphabet_index: int = 0 # 字母索引
    ad_index: int = 0 # 六位数字索引
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 遍历文件夹
    for root, dirs, files in os.walk(img_dir): # root: 当前目录，dirs: 当前目录下的子目录，files: 当前目录下的所有文件
        for file in files:
            path = os.path.join(root, file) # 拼合图片路径
            image: cv2.Mat = seg.pipeline(path) # 获得车牌图片
            try:
                character_images: list = ce.pipeline(image) # 获得字符图片
                provinces = character_images[0][0]
                alphabets = character_images[1][0]
                ads = character_images[2][0]
                # 存储图片
                result = cv2.imwrite(save_dir + "/provinces/" + str(extract_labels(file)[0]) + "_" + str(province_index) + ".jpg", provinces) # 存储省份及其标识
                result = cv2.imwrite(save_dir + "/alphabets/" + str(extract_labels(file)[1]) + "_" + str(alphabet_index) + ".jpg", alphabets) # 存储字母及其标识
                for i in range(len(ads)):
                    result = cv2.imwrite(save_dir + "/ads/" + str(extract_labels(file)[2][i]) + "_" + str(ad_index) + ".jpg", ads[i]) # 存储六位数字及其标识
                    ad_index += 1 # 六位数字索引加1
                
                # 更新索引
                province_index += 1
                alphabet_index += 1
                
            except Exception as e:
                continue


def test():
    img_dir = './Data/train'
    save_dir = 'Real_data'
    data_preprocess(img_dir, save_dir)
    print("done")

test()


