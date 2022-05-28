from typing import Tuple
from unittest import result
import segmentation
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# 一些可能改进的点
# 0、首先我们的目标是为了提取里面的车牌字符，而不是提取车牌的边框，为此我们要想办法能怎么去评估像素的权值重要性
# 1、rgb，或者说bgr三个分量加权综合评估，并使用mldl手段，来尝试获得比较好的weight，以获得更好的结果，我这里偷懒只用了绿色通道的
# 2、我****闲着没事用新能源汽车的车牌干嘛，老老实实用老的数据库不就得了，跟着敲一遍完事
# 3、二值化的时候，这能不能做个根据像素重要性（weight）来可变的阈值，啊当然这个weight，mldl请
# 4、在跑那个，叫啥来着，列求和然后观察折线图、图（execl可以做到）的那个方法，可以不一定要直接求和，可以从正态分布采样赋权求和。很可惜我的方差是“经验值”，又名“瞎**填”，方差可以根据mldl或者传统数学来求？同时，多元正态分布也是一种很好的改进方向
# 5、突然想到个词，加权本身是给重要性评估，“兴趣焦点权值”，莫名想到这个词，搜了搜好像是RQI？
# 6、我恨你偏旁……启发式选择性合并，又名瞎**猜
# 7、那些富有动态模糊美感的图片，我不做了。那玩意要的太难，也许可以通过检测运动向量mldl之类的东西来还原。我时间太不够了，让我再学学d2l和数学吧。
# 8、真这么搞下去建议搞成毕业设计，或者直接灌水——但我没有看多少论文，准确来说一篇都没看过，我相信我这些想法别人一定都想到了。不去关注别人做了啥很容易重复造轮子，可我也不是来发论文的，而且我这些优化，怎么说呢，太tricky了。只要不上mldl数学方法那我觉得我这些trick这些都挺不行的——话说回来mldl本身也挺tricky的。毕竟不是纯理论，这玩意偏应用啥的，tricky就tricky吧。我和程序有一个能跑就行。

def normal_distribution(x: np.float32, sigma: np.float32 = 1):  # 正态分布
    return np.exp(-x ** 2 / 2 * sigma * sigma) / np.sqrt(2 * np.pi)

def classify_image(images: list) -> list[list, list, list]:
    provinces = [images[0]]
    alphabets = [images[1]]
    ads = [images[2: len(images)]]
    return [provinces, alphabets, ads]

def discretization(function, min_value: int, max_value: int, count: int, sigma: np.float32 = 1) -> np.ndarray:  # 离散化函数
    input: np.ndarray = np.linspace(min_value, max_value, count) 
    output: np.ndarray = function(input, sigma)
    return output / output.max()

def thresholding_image(image) -> cv2.Mat: # 二值化
    image = image[:, :, 1] # 只取绿色分量图像，转化颜色空间也许效果会更好
    image = cv2.GaussianBlur(image, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    _, image = cv2.threshold( 
        image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # 二值化， 0是黑色，1/255是白色
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    return image

def get_top_via_ratio(ratio_list: np.ndarray, threshold: np.float32 = 0.95) -> np.ndarray: # 根据阈值获取前面的元素的像素索引，白色是255，也就是说越高越要切割
    index = np.linspace(0, ratio_list.shape[0] - 1, ratio_list.shape[0])
    ratio_list = np.insert(ratio_list, 1, index, axis=1) # 添加一列
    sorted_index = np.argsort(ratio_list[:, 0])[::-1] # 按照第一列排序，反向排序，大的在上面
    ratio_list = ratio_list[sorted_index]
    result = np.sort(ratio_list[ratio_list[:, 0] >= threshold][:, 1], axis=0) # 排序后切割，返回像素索引
    return result.astype(np.int32)

def image_split(image: cv2.Mat, pixel_indices: np.ndarray, axis: int = 0) -> list: # 切割图像, axis = 0为行切割
    pixel_indices = np.insert(pixel_indices, 0, 0, axis=0)
    result = []
    if axis == 0: # 按行切割
        pixel_indices = np.insert(pixel_indices, pixel_indices.shape[0], image.shape[0], axis=0)
        for i in range(0, len(pixel_indices) - 1):
            if pixel_indices[i + 1] - pixel_indices[i] > 50: #筛掉一些比较小的没意义的切割
                result.append(image[pixel_indices[i]:pixel_indices[i + 1], :])
    else: # axis = 1为列切割
        pixel_indices = np.insert(pixel_indices, pixel_indices.shape[0], image.shape[1], axis=0)
        last_index_pointer = 0
        for i in range(0, len(pixel_indices)- 1):
            pixel_range = pixel_indices[i + 1] - pixel_indices[last_index_pointer]
            if pixel_range > 25: #合并掉一些比较小的没意义的切割，专门负责一些他妈的分开的字
                result.append(image[:, pixel_indices[last_index_pointer]:pixel_indices[i + 1]])
            if pixel_range > 25 or pixel_range < 10: #这就真的是纯粹的“启发式”了，啊，懂得都懂……大于25我判断是个完整的字，小于10我觉得就是个没用的切割，两种情况下我都让last_index_pointer++。但是捏，如果是在25到10之间的，我怀疑是个偏旁。这绝对是个可以改进的地方，主要是没啥数学依据，只是一个经验值。经验值，懂得都懂，mldl优化拟合呗……顺带copilot真尼玛好用
                last_index_pointer = i + 1 #啊多bb一句，其实可以根据划分出来的区域内部的黑色像素含量来判断内部，比如百分比30%以上的黑色像素就认为是一个完整的字，这样就不会出现一个字切成两个字的情况了

    return result

def vertical_cutout(image: cv2.Mat) -> list: # 垂直切割，原理就是求列的白色像素个数，然后根据比例切割图像啥的
    normal_distribution_discretization: np.ndarray = discretization(normal_distribution, -1, 1, image.shape[0], 1.15).reshape(image.shape[0], 1) # 这个sigma就是经验值了，可以改进
    ratio_list: np.ndarray = np.dot(image.transpose(), normal_distribution_discretization) / np.dot(image.transpose(), normal_distribution_discretization).max() # 求和的比例，然后根据比例切割图像，逼着我加权了一个1.15
    pixel_indices: np.ndarray = get_top_via_ratio(ratio_list, 0.80) # 根据阈值获取前面的元素的像素索引
    return image_split(image, pixel_indices, 1)


def horizontal_cutout(image: cv2.Mat) -> list: # 水平切割，原理就是求行的白色像素个数，然后根据比例切割图像啥的
    normal_distribution_discretization: np.ndarray = discretization(normal_distribution, -1, 1, image.shape[1], 1.5).reshape(image.shape[1], 1)
    ratio_list = np.dot(image, normal_distribution_discretization) / np.dot(image, normal_distribution_discretization).max() # 求和的比例，然后根据比例切割图像， 加权了一个1.5
    pixel_indices: np.ndarray = get_top_via_ratio(ratio_list, 0.85) # 根据经验阈值获取前面的元素的像素索引
    return image_split(image, pixel_indices, 0)

def get_indices(image: cv2.Mat) -> np.ndarray:
    result = np.array(
        [[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]], 
        dtype=np.float32)
    return result

def transform_image(image: cv2.Mat, source_vertices: np.ndarray, destination_vertices: np.ndarray, destination_size: Tuple[int, int]) -> cv2.Mat: #只转换图片
    perspective_transformation_matrix = cv2.getPerspectiveTransform(source_vertices, destination_vertices) #计算透视变换矩阵
    return cv2.warpPerspective(image, perspective_transformation_matrix, destination_size)

def transform_all_images(images: list) -> list:
    des_vertices = np.array(
        [[0, 0], [30, 0], [30, 90], [0, 90]],
        dtype=np.float32)

    result: list = []

    for image in images:
        temp_image = transform_image(image, get_indices(image), des_vertices, (30, 90))
        result.append(temp_image)
    
    return result

def module_test():
    image = segmentation.pipeline("Data/train/015-91_91-282&460_498&530-496&530_282&526_282&460_498&466-0_0_3_27_29_25_30_33-149-96.jpg")
    cv2.imshow("image", image)
    cv2.waitKey(0)

    image = thresholding_image(image)
    cv2.imshow("image", image)
    cv2.waitKey(0)

    images = vertical_cutout(horizontal_cutout(image)[0])
    images = transform_all_images(images)
    for i in images:
        cv2.imshow("image", i)
        cv2.waitKey(0)

def pipeline(image: cv2.Mat, show_image: bool = False) -> list: # 处理管线，输出的字符都是30*90的。了解函数的话可以从这里开始，如果有啥不明白的还装了copilot的话，可以打个井号看收敛出来的注释
    threshold_image = thresholding_image(image)
    cut_out_images: list = vertical_cutout(horizontal_cutout(threshold_image)[0])
    images: list = transform_all_images(cut_out_images)

    if show_image:
        cv2.imshow("threshold_image", threshold_image)
        cv2.waitKey(0)

        for i in cut_out_images:
            cv2.imshow("image", i)
            cv2.waitKey(0)
        
    if len(images) != 8:
        raise Exception("切割出来的图像数量不对，建议跳过这个图像")
    return classify_image(images) # 分类图像，返回省份，字母，字母+数字
