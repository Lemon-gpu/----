from typing import Tuple
from matplotlib import image

import torch
import segmentation
import cv2
import numpy as np
import character_extraction
import torch.nn as nn
import torch.utils.data as data
import os
import pandas as pd
from torchvision.io import read_image

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"] # 34个
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O'] # 25个
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O'] # 35个

def classify_image(images: list) -> list[list, list, list]:
    provinces = [image[0]]
    alphabets = [image[1]]
    ads = [image[2: len(image)]]
    return [provinces, alphabets, ads]

# 这个是重度“借鉴”了torch的quickstart的代码，自己的数据集
class image_dataset(data.Dataset):
    def __init__(self, img_dir: str, annotations_file_path: str): # 初始化一些参数
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotations_file_path, header=0)
        self.transform = nn.Flatten()

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image: torch.Tensor = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        image = self.transform(image)
        image = image.float()
        return image, label


# 识别
class Provinces_Recognition(nn.Module): # 省份识别
    def __init__(self, input_size: int, output_size: int):
        super(Provinces_Recognition, self).__init__()
        self.flatten = nn.Flatten()
        self.pipeline = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size), 
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.pipeline(x)
        return x


# 这个是重度“借鉴”了torch的quickstart的代码
def train(dataloader: data.DataLoader, model: nn.Module, loss_fn, optimizer: torch.optim.Optimizer):
    device = "cuda" if torch.cuda.is_available() else "cpu" # 判断是否有GPU
    model.to(device) # 将模型放到GPU上
    model.train() # 将当前model设置为训练模式
    size = len(dataloader.dataset) # 数据集大小

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device) # 将数据放到GPU上

        # 前向传播
        pred = model(X) # 预测
        loss = loss_fn(pred, y) # 计算损失

        # 反向传播
        optimizer.zero_grad() # 清空梯度
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

        if batch % 10 == 0: # 每100个batch打印一次
            loss, current = loss.item(), batch * len(X) # 获取损失和当前的batch数
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") # 打印损失和当前的batch数
    
    model.eval() # 将当前model恢复为测试模式

def training_driver(image_dir: str, annotations_file_path: str, label_count: int, image_size: int = 2700, epoch: int = 10) -> nn.Module: # 训练主函数
    # 初始化数据集
    dataset = image_dataset(image_dir, annotations_file_path)
    # 初始化数据集的迭代器
    data_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    # 初始化网络
    net = Provinces_Recognition(image_size, label_count)
    # 初始化优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # 初始化损失函数
    loss_func = nn.CrossEntropyLoss() # 先用着交叉熵
    # 训练
    for e in range(epoch): # 训练十轮
        train(data_loader, net, loss_func, optimizer)
    
    return net


# 这个是重度“借鉴”了torch的quickstart的代码，单轮测试
def test(dataloader: data.DataLoader, model: nn.Module, loss_fn):
    device = "cuda" if torch.cuda.is_available() else "cpu" # 判断是否有GPU
    size = len(dataloader.dataset) # 数据集大小
    num_batches = len(dataloader) # 批次数量
    model.eval() # 将当前model设置为测试模式
    test_loss, correct = 0, 0 # 初始化损失和正确率
    with torch.no_grad(): # 不使用梯度
        for X, y in dataloader: # 遍历数据集
            X, y = X.to(device), y.to(device) # 将数据放到GPU上
            pred = model(X) # 预测
            test_loss += loss_fn(pred, y).item() # 计算损失
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()   # 计算正确率

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def test_driver(image_dir:str, annotations_file_path: str, label_count: int, model: nn.Module) -> None: # 测试主函数
    # 初始化数据集
    dataset = image_dataset(image_dir, annotations_file_path)
    # 初始化数据集的迭代器
    data_loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    # 初始化损失函数
    loss_func = nn.CrossEntropyLoss() # 先用着交叉熵
    # 测试
    test(data_loader, model, loss_func)


# 模块测试
def module_test():
    # 初始化模型
    # 省份
    province_net = training_driver("./Real_data/train/provinces", "./Real_data/train/provinces.csv", 34)
    # 字母
    alphabets_net = training_driver("./Real_data/train/alphabets", "./Real_data/train/alphabets.csv", 25)
    # 字符
    characters_net = training_driver("./Real_data/train/ads", "./Real_data/train/ads.csv", 35)

    # 测试
    test_driver("./Real_data/test/provinces", "./Real_data/test/provinces.csv", 34, province_net) # 省份
    test_driver("./Real_data/test/alphabets", "./Real_data/test/alphabets.csv", 25, alphabets_net) # 字母
    test_driver("./Real_data/test/ads", "./Real_data/test/ads.csv", 35, characters_net) # 字符

module_test()

    


