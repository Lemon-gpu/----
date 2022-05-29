from asyncio.windows_utils import pipe
from typing import Tuple
import cv2
import numpy as np

def get_point(image_path: str) -> np.ndarray: #获取四个点的坐标
    image_path = image_path.split("/")[-1].split('.')[0] #去除后缀名
    vertices_locations: list = image_path.split('-')[3].split('_') #提取四个点的坐标
    result: np.ndarray = np.array([], dtype=np.float32) #初始化
    for vertex in vertices_locations:
        temp = vertex.split('&')
        result = np.append(result, np.array([np.float32(temp[0]), np.float32(temp[1])]))

    return result.reshape(4, 2)

def transform_image(image: cv2.Mat, source_vertices: np.ndarray, destination_vertices: np.ndarray, destination_size: Tuple[int, int]) -> cv2.Mat: #只转换图片
    perspective_transformation_matrix = cv2.getPerspectiveTransform(source_vertices, destination_vertices) #计算透视变换矩阵
    return cv2.warpPerspective(image, perspective_transformation_matrix, destination_size)

# 属于是偷了个大懒了，直接用CCPD的数据标注来提取车牌。也许可以用一些传统数学或者是mldl的优化手段来获得更自动化的结果
def get_license_plate(image: cv2.Mat, image_path: str) -> cv2.Mat: #获取车牌
    result_vertices: np.ndarray = np.array(  # 车牌的四个点，就是在这里定义了150*450。图像的坐标系和我们平常的有点不同，需要注意
        [[450, 150], [0, 150], [0, 0], [450, 0]],
        dtype=np.float32
    )
    vertices_locations: list = get_point(image_path) #获取四个点的坐标
    return transform_image(image, vertices_locations, result_vertices, (450, 150))

def get_image(image_path: str) -> cv2.Mat: #获取图片
    image = cv2.imread(image_path)
    return image

def module_test():
    pipeline("Data/train/015-91_91-282&460_498&530-496&530_282&526_282&460_498&466-0_0_3_27_29_25_30_33-149-96.jpg", True)

def pipeline(image_path: str, show_image: bool = False) -> cv2.Mat: #整个流水线，输出的车牌都是150*450的。阅读函数的话可以从这里开始
    image = get_image(image_path)
    license_plate = get_license_plate(image, image_path)
    if show_image:
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.imshow('license_plate', license_plate)
        cv2.waitKey(0)
    return license_plate
