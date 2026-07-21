import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # 第一层卷积：in=1通道，输出16通道，卷积核3×3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        # 第二层卷积：输入16通道，输出32通道，卷积核3×3
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 最大池化层，kernel=2，步长默认2
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        # 全连接层：展平后维度32*7*7，输出10分类
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # 第一组：Conv -> ReLU -> MaxPool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool(x)

        # 第二组：Conv -> ReLU -> MaxPool
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool(x)

        # Flatten：只展平图像维度，保留batch维度
        x = torch.flatten(x, start_dim=1)

        # 最终线性输出
        out = self.fc(x)
        return out

# 测试模型输入输出
if __name__ == "__main__":
    model = CustomCNN()
    # 模拟输入：batch=4，通道1，28×28
    dummy_input = torch.randn(4, 1, 28, 28)
    output = model(dummy_input)
    print(f"输入张量形状: {dummy_input.shape}")  # torch.Size([4, 1, 28, 28])
    print(f"输出张量形状: {output.shape}")      # torch.Size([4, 10])