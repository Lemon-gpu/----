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


class CustomImageDataset(data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



# 识别
class Provinces_Recognition(nn.Module): # 省份识别
    def __init__(self):
        super(Provinces_Recognition, self).__init__()
        self.flatten = nn.Flatten()
        self.pipeline = nn.sequential(
            nn.Linear(90 * 30, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 34),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.pipeline(x)
        return x

class alphabets_Recognition(nn.Module): # 字母识别
    def __init__(self):
        super(Provinces_Recognition, self).__init__()
        self.flatten = nn.Flatten()
        self.pipeline = nn.sequential(
            nn.Linear(90 * 30, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 25),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.pipeline(x)
        return x
    
class ads_Recognition(nn.Module): # 后六个字符识别
    def __init__(self):
        super(Provinces_Recognition, self).__init__()
        self.flatten = nn.Flatten()
        self.pipeline = nn.sequential(
            nn.Linear(90 * 30, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 35),
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

        if batch % 100 == 0: # 每100个batch打印一次
            loss, current = loss.item(), batch * len(X) # 获取损失和当前的batch数
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]") # 打印损失和当前的batch数
    
    model.eval() # 将当前model恢复为测试模式


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


def module_test():
    image_path: str = "Data/train/015-91_91-282&460_498&530-496&530_282&526_282&460_498&466-0_0_3_27_29_25_30_33-149-96.jpg"
    image: cv2.Mat = segmentation.pipeline(image_path)
    character_images: list = character_extraction.pipeline(image, True)


