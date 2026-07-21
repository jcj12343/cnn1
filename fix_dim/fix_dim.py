import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# 修复模块导入路径
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
sys.path.append(os.path.join(root_path, "cnn_model"))

# 加载CNN模型
from cnn_model import CustomCNN

# 获取运算设备
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集类
class NpzMNISTDataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# 加载三份npz数据集，返回dataloader
def get_dataloaders(batch_size=64):
    train_npz = np.load(os.path.join(root_path, "data_classify/train.npz"))
    val_npz = np.load(os.path.join(root_path, "data_classify/val.npz"))
    test_npz = np.load(os.path.join(root_path, "data_classify/test.npz"))

    x_train, y_train = train_npz["x_train"], train_npz["y_train"]
    x_val, y_val = val_npz["x_val"], val_npz["y_val"]
    x_test, y_test = test_npz["x_test"], test_npz["y_test"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = NpzMNISTDataset(x_train, y_train, transform)
    val_ds = NpzMNISTDataset(x_val, y_val, transform)
    test_ds = NpzMNISTDataset(x_test, y_test, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"Train set: {len(train_ds)} | Val set: {len(val_ds)} | Test set: {len(test_ds)}")
    return train_loader, val_loader, test_loader, len(train_ds), len(val_ds), len(test_ds)

# 初始化模型、损失、优化器
def get_train_components(lr=1e-3):
    device = get_device()
    model = CustomCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return device, model, criterion, optimizer