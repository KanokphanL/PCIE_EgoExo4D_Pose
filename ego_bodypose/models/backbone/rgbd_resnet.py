import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50


class RGBDResNet(nn.Module):
    def __init__(self, resnet_version=18):
        super().__init__()
        if resnet_version == 18:
            self.features = resnet18(weights='DEFAULT')
        elif resnet_version == 34:
            self.features = resnet34(weights='DEFAULT')
        elif resnet_version == 50:
            self.features = resnet50(weights='DEFAULT')
        else:
            raise ValueError("ResNet version must be 18, 34, or 50.")

        # 修改第一层卷积层以接受 4 通道输入
        self.features.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 移除最后一层全连接层
        self.features = nn.Sequential(*list(self.features.children())[:-1])

    def forward(self, x):
        # 前向传播
        features = self.features(x)
        return features


# 示例使用
if __name__ == "__main__":
    batch_size = 2
    H, W = 224, 224
    input_tensor = torch.randn(batch_size, 4, H, W)

    model = RGBDResNet(resnet_version=18)
    output = model(input_tensor)
    print("Output shape:", output.shape)
    