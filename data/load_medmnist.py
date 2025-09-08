import medmnist
from medmnist import PathMNIST
import torch
from torch.utils.data import Dataset, random_split
import numpy as np


# 1. 加载数据
class MedMNISTDataset(Dataset):
    def __init__(self, split='train', transform=None):
        # 加载PathMNIST数据
        self.dataset = PathMNIST(split=split, download=True)
        self.transform = transform
        # 类别映射
        self.class_names = ['Adenocarcinoma', 'Benign tissue', 'Fibrotic tissue', 'Inflammatory tissue',
                            'Normal tissue', 'Poorly differentiated carcinoma', 'Well differentiated carcinoma']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label.item()


# 2. 划分数据集 & 构造输入
def prepare_datasets():
    from torchvision import transforms

    # 定义图像预处理（仅做ToTensor，避免通道数不匹配导致归一化报错）
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 加载训练集
    full_train_dataset = MedMNISTDataset(split='train', transform=transform)

    # 假设 "Adenocarcinoma" 的标签是 0
    FORGET_LABEL = 0

    # 创建索引列表
    retain_indices = []
    forget_indices = []

    for idx in range(len(full_train_dataset)):
        _, label = full_train_dataset[idx]
        if label == FORGET_LABEL:
            forget_indices.append(idx)
        else:
            retain_indices.append(idx)

    # 定义数据项构造函数，直接返回单个字典，便于 trainer/eval 使用
    def construct_item(image, label, question="这是一张病理切片，请诊断这是什么类型的组织？"):
        return {"image": image, "text": question, "answer": int(label)}

    # 创建三个数据集（元素均为字典）
    # 保留集 (Dr)
    retain_dataset = [construct_item(full_train_dataset[i][0], full_train_dataset[i][1]) for i in retain_indices]

    # 遗忘集 (Df)
    forget_dataset = [construct_item(full_train_dataset[i][0], full_train_dataset[i][1]) for i in forget_indices]

    # 加载验证集
    val_dataset = MedMNISTDataset(split='val', transform=transform)
    val_data = [construct_item(val_dataset[i][0], val_dataset[i][1]) for i in range(len(val_dataset))]

    return retain_dataset, forget_dataset, val_data

# 在 main.py 中调用
# retain_data, forget_data, val_data = prepare_datasets()