import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn as nn
import sys

def get_model(name='resnet18', num_classes=10):
    if name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif name == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif name == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {name}")

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class CBAM(nn.Module):
    """混合注意力机制（通道+空间）"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 通道注意力
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        channel_att = (avg_out + max_out).view(x.size(0), -1, 1, 1)
        x = x * channel_att

        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        return x * spatial_att

# 在 train_cbam.py 或者 standalone 文件中替换此部分

import torch.nn as nn
import torchvision.models as models
# from CBAM import CBAM

def get_resnet_cbam(name='resnet18', num_classes=10):
    """
    返回一个在每个 Stage 末尾插入 CBAM 模块的 ResNet。
    支持 resnet18、resnet34 和 resnet50。
    """
    # print(">>> name1: ", name)
    # sys.exit()
    # 1) 选择基础骨干网络
    if name == 'resnet18':
        backbone = models.resnet18(pretrained=True)
        channels = [64, 128, 256, 512]
    elif name == 'resnet34':
        backbone = models.resnet34(pretrained=True)
        channels = [64, 128, 256, 512]
    elif name == 'resnet50':
        backbone = models.resnet50(pretrained=True)
        channels = [256, 512, 1024, 2048]
    else:
        raise ValueError(f"Unsupported model: {name}")

    # 2) 修改第一层以适配 CIFAR-10 32×32 输入
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()

    # 3) 在每个 layer 的末尾插入对应通道数的 CBAM
    #    channels 列表与 layer1~layer4 对应
    backbone.layer1.add_module("cbam1", CBAM(channels[0]))
    backbone.layer2.add_module("cbam2", CBAM(channels[1]))
    backbone.layer3.add_module("cbam3", CBAM(channels[2]))
    backbone.layer4.add_module("cbam4", CBAM(channels[3]))

    # 4) 替换最后的全连接为 CIFAR-10 的输出
    backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
    return backbone

