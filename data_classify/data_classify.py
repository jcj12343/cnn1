import numpy as np
from torch.utils.data import random_split

# 1. 读取原始mnist.npz
data = np.load(r"C:\Users\jcj\Downloads\mnist.npz")
x_train_all = data["x_train"]
y_train_all = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

# 2. 划分训练集、验证集（6000张val）
total_num = len(x_train_all)
val_size = 6000
train_size = total_num - val_size

# 随机划分索引
indices = np.arange(total_num)
np.random.shuffle(indices)
train_idx = indices[:train_size]
val_idx = indices[train_size:]

# 切分图像、标签
x_train = x_train_all[train_idx]
y_train = y_train_all[train_idx]
x_val = x_train_all[val_idx]
y_val = y_train_all[val_idx]

# 3. 保存为三个独立npz文件
np.savez("train.npz", x_train=x_train, y_train=y_train)
np.savez("val.npz", x_val=x_val, y_val=y_val)
np.savez("test.npz", x_test=x_test, y_test=y_test)

print(f"train: {len(x_train)} 张")
print(f"val: {len(x_val)} 张")
print(f"test: {len(x_test)} 张")