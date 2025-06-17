# # train_cbam.py

# import os
# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm

# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader

# from utils import plot_curve

# from model import get_resnet_cbam


# # -----------------------------
# # 2. 数据加载
# # -----------------------------
# def get_dataloaders(batch_size, num_workers):
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,) * 3, (0.5,) * 3)
#     ])
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,) * 3, (0.5,) * 3)
#     ])

#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
#                                             transform=transform_train)
#     testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
#                                             transform=transform_test)

#     train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
#                               num_workers=num_workers, pin_memory=True)
#     test_loader  = DataLoader(testset,  batch_size=batch_size, shuffle=False,
#                               num_workers=num_workers, pin_memory=True)
#     return train_loader, test_loader


# # -----------------------------
# # 3. 训练与验证函数
# # -----------------------------
# def train_one_epoch(model, loader, criterion, optimizer, device):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for inputs, labels in tqdm(loader, desc="Training", leave=False):
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         preds = outputs.argmax(dim=1)
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)

#     avg_loss = running_loss / len(loader)
#     acc = 100.0 * correct / total
#     return avg_loss, acc


# def evaluate(model, loader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for inputs, labels in tqdm(loader, desc="Validating", leave=False):
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             running_loss += loss.item()
#             preds = outputs.argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#     avg_loss = running_loss / len(loader)
#     acc = 100.0 * correct / total
#     return avg_loss, acc



# # -----------------------------
# # 5. 主函数: 解析参数 + 训练循环
# # -----------------------------
# def main():
#     parser = argparse.ArgumentParser(description='Train ResNet with CBAM on CIFAR-10')
#     parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'],
#                         help='ResNet architecture (resnet18 or resnet34)')
#     parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
#     parser.add_argument('--epochs', type=int, default=50, help='Total epochs to run (default: 50)')
#     parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (default: 0.1)')
#     parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
#     parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')
#     parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers (default: 4)')
#     parser.add_argument('--gpuid', type=int, default=0, help='GPU id to use (default: 0)')
#     parser.add_argument('--save_dir', type=str, default='cbam_res',
#                         help='Directory to save checkpoints (default: cbam_res)')
#     args = parser.parse_args()

#     # 设备设置
#     os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f'Using device: {device}, ResNet: {args.arch}, GPUID: {args.gpuid}')

#     # 创建保存目录
#     save_dir = os.path.join(args.save_dir, args.arch)
#     os.makedirs(save_dir, exist_ok=True)
#     best_ckpt_path = os.path.join(save_dir, f'{args.arch}_best.pth')
#     loss_plot_path = os.path.join(save_dir, 'loss_curve.png')
#     acc_plot_path  = os.path.join(save_dir, 'acc_curve.png')

#     # 准备数据
#     train_loader, test_loader = get_dataloaders(batch_size=args.batch_size,
#                                                 num_workers=args.num_workers)

#     # 构建模型、损失、优化器
#     # print(">>> args.arch: ", args.arch)
#     model = get_resnet_cbam(name=args.arch, num_classes=10).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=args.lr,
#                           momentum=args.momentum, weight_decay=args.weight_decay)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

#     # 训练循环
#     best_acc = 0.0
#     train_losses, train_accs, test_losses, test_accs = [], [], [], []

#     for epoch in range(1, args.epochs + 1):
#         print(f'\nEpoch {epoch}/{args.epochs}')

#         # 1) 训练
#         train_loss, train_acc = train_one_epoch(model, train_loader,
#                                                 criterion, optimizer, device)
#         # 2) 验证
#         test_loss, test_acc = evaluate(model, test_loader, criterion, device)

#         print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
#         print(f'Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%')

#         train_losses.append(train_loss)
#         train_accs.append(train_acc)
#         test_losses.append(test_loss)
#         test_accs.append(test_acc)

#         # 保存最佳模型
#         if test_acc > best_acc:
#             best_acc = test_acc
#             torch.save(model.state_dict(), best_ckpt_path)
#             print(f' Best model saved to {best_ckpt_path} with Test Acc: {best_acc:.2f}%')

#         scheduler.step()

#     print(f'\n==> Training Finished. Best Test Acc: {best_acc:.2f}%')

#     # 绘制并保存损失和准确率曲线
#     plot_curve(train_losses, 'Epoch', 'Train Loss', 'Training Loss', loss_plot_path)
#     plot_curve(train_accs,  'Epoch', 'Train Acc (%)', 'Training Accuracy', acc_plot_path)
#     print(f' Loss curve: {loss_plot_path}')
#     print(f' Acc curve:  {acc_plot_path}')


# if __name__ == '__main__':
#     main()


# train_cbam.py

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import plot_curve
from model import get_resnet_cbam
import matplotlib.pyplot as plt


# -----------------------------
# 2. 数据加载
# -----------------------------
def get_dataloaders(batch_size, num_workers):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                            transform=transform_train)
    testset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                            transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(testset,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# -----------------------------
# 3. 训练与验证函数
# -----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(loader)
    acc = 100.0 * correct / total
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(loader)
    acc = 100.0 * correct / total
    return avg_loss, acc


# -----------------------------
# 绘制每类准确率函数
# -----------------------------
def plot_classwise_accuracy(model, loader, device, save_path):
    """
    计算并绘制每个类别的准确率条形图（CIFAR-10）
    """
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    correct_pred = {classname: 0 for classname in classes}
    total_pred   = {classname: 0 for classname in classes}

    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Classwise Acc", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                cls_name = classes[label]
                if label == prediction:
                    correct_pred[cls_name] += 1
                total_pred[cls_name] += 1

    plt.figure(figsize=(10, 5))
    for i, cls in enumerate(classes):
        acc = correct_pred[cls] / total_pred[cls]
        plt.barh(i, acc, color='orange')
    plt.yticks(range(len(classes)), classes)
    plt.xlabel('Accuracy')
    plt.title('Class-wise Accuracy')
    plt.xlim(0, 1.0)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# -----------------------------
# 5. 主函数: 解析参数 + 训练循环
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description='Train ResNet with CBAM on CIFAR-10')
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='ResNet architecture (resnet18/resnet34/resnet50)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, help='Total epochs to run (default: 50)')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (default: 5e-4)')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers (default: 4)')
    parser.add_argument('--gpuid', type=int, default=2, help='GPU id to use (default: 0)')
    parser.add_argument('--save_dir', type=str, default='cbam_res',
                        help='Directory to save checkpoints and plots (default: cbam_res)')
    args = parser.parse_args()

    # 设备设置
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}, ResNet: {args.arch}, GPUID: {args.gpuid}')

    # 创建保存目录
    save_dir = os.path.join(args.save_dir, args.arch)
    os.makedirs(save_dir, exist_ok=True)
    best_ckpt_path = os.path.join(save_dir, f'{args.arch}_best.pth')
    loss_plot_path = os.path.join(save_dir, 'loss_curve.png')
    acc_plot_path  = os.path.join(save_dir, 'acc_curve.png')
    class_acc_path = os.path.join(save_dir, 'classwise_accuracy.png')

    # 准备数据
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size,
                                                num_workers=args.num_workers)

    # 构建模型、损失、优化器
    model = get_resnet_cbam(name=args.arch, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # 训练循环
    best_acc = 0.0
    train_losses, train_accs = [], []
    test_losses, test_accs   = [], []

    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')

        # 1) 训练
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, device)
        # 2) 验证
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%')

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_ckpt_path)
            print(f' Best model saved to {best_ckpt_path} with Test Acc: {best_acc:.2f}%')

        scheduler.step()

    print(f'\n==> Training Finished. Best Test Acc: {best_acc:.2f}%')

    # 绘制并保存损失和准确率曲线
    plot_curve(train_losses, 'Epoch', 'Train Loss', 'Training Loss', loss_plot_path)
    plot_curve(train_accs,  'Epoch', 'Train Acc (%)', 'Training Accuracy', acc_plot_path)
    print(f' Loss curve saved to {loss_plot_path}')
    print(f' Acc curve saved to {acc_plot_path}')

    # 绘制并保存每类准确率条形图
    plot_classwise_accuracy(model, test_loader, device, class_acc_path)
    print(f' Class-wise accuracy plot saved to {class_acc_path}')


if __name__ == '__main__':
    main()
