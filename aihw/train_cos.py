import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from tqdm import tqdm

# 改进1：增强数据预处理
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=train_transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=test_transform)

batch_size = 64
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=4, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2, pin_memory=True)


# 改进2：增强模型结构
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


class ResidualBlock(nn.Module):
    """改进的残差块"""

    def __init__(self, in_channels, expansion=4):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            CBAM(hidden_dim),  # 使用CBAM代替SE
            nn.Conv2d(hidden_dim, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.shortcut = nn.Identity()
        # 初始化最后一层卷积权重为零
        nn.init.constant_(self.conv[-2].weight, 0)

    def forward(self, x):
        return F.gelu(self.conv(x) + self.shortcut(x))


class EnhancedNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入层
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            CBAM(64)
        )

        # 残差阶段
        # 残差阶段
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 3, stride=2)
        self.layer3 = self._make_layer(128, 256, 3, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)  # 新增层

        # 分类头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 10)
        )

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        # 下采样层（添加完整的预处理）
        if stride != 1 or in_channels != out_channels:
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU()
            ))

        # 添加残差块
        for _ in range(blocks):
            layers.append(ResidualBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # 新增层前向传播
        return self.head(x)


# 改进3：优化训练策略
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = EnhancedNet().to(device)

    # 使用AdamW优化器
    optimizer = optim.SGD(
        net.parameters(),
        lr=0.1,          # 初始学习率（通常高于 Adam）
        momentum=0.9,     # 动量加速收敛‌
        weight_decay=1e-5, # 权重衰减防过拟合
        nesterov=True
    )

    # 余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=200,  # 周期长度（总epoch数）
        eta_min=1e-5  # 最小学习率
    )

    # 标签平滑损失
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0.0
    # patience = 15


    for epoch in range(200):
        net.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        pbar = tqdm(trainloader, desc=f'Epoch {epoch + 1}/200', ncols=100)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Mixup数据增强
            lam = torch.distributions.beta.Beta(1.0, 1.0).sample().to(device)
            index = torch.randperm(inputs.size(0)).to(device)
            mixed_inputs = lam * inputs + (1 - lam) * inputs[index]

            optimizer.zero_grad()

            outputs = net(mixed_inputs)
            loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[index])

            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            optimizer.step()
            with torch.no_grad():
                total_loss += loss.item()
                clean_outputs = net(inputs)
                _, predicted = clean_outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix({
                    'lr': f"{current_lr:.4f}",
                    'loss': f"{total_loss / (len(pbar) + 1):.4f}",
                    'acc': f"{100. * correct / total:.2f}%"
                })
        scheduler.step()
        # 验证集评估
        net.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                test_correct += outputs.argmax(1).eq(labels).sum().item()
                test_total += labels.size(0)

        val_acc = 100. * test_correct / test_total
        print(f"Val Acc: {val_acc:.2f}%")

        # 早停机制
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), 'best_model.pth')


if __name__ == '__main__':
    train_model()
