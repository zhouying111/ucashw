# train_with_plots.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import plot_curve
from dataset import get_dataloaders    # 假设此函数返回 train_loader, test_loader
from model import get_model            # 假设此函数返回指定的 ResNet 模型
import matplotlib.pyplot as plt


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, desc='Training', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    return total_loss / len(loader), acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    return total_loss / len(loader), acc


def plot_classwise_accuracy(model, loader, device, save_path):
    """
    计算并绘制每个类别的准确率条形图
    """
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    correct_pred = {classname: 0 for classname in classes}
    total_pred   = {classname: 0 for classname in classes}

    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Classwise Acc', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train ResNet on CIFAR-10 with Curves')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'], help='backbone architecture')
    parser.add_argument('--batch_size',  type=int,   default=128)
    parser.add_argument('--epochs',      type=int,   default=100)
    parser.add_argument('--lr',          type=float, default=0.1)
    parser.add_argument('--momentum',    type=float, default=0.9)
    parser.add_argument('--weight_decay',type=float, default=5e-4)
    parser.add_argument('--num_workers', type=int,   default=8)
    parser.add_argument('--gpuid',       type=int,   default=1)
    parser.add_argument('--save_dir',    type=str,   default='results')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    save_dir = os.path.join(args.save_dir, args.arch)
    os.makedirs(save_dir, exist_ok=True)

    # 数据加载
    train_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 模型、损失、优化器
    model = get_model(name=args.arch, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    best_test_acc = 0.0
    best_train_acc_at_best_test = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device
        )

        print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'Test  Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 更新最佳测试准确率及对应的训练准确率
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_train_acc_at_best_test = train_acc

        scheduler.step()

    # 打印最佳测试准确率及当时训练准确率
    print(f'\nBest Test Accuracy: {best_test_acc:.2f}%')
    print(f'Corresponding Train Accuracy: {best_train_acc_at_best_test:.2f}%')

    # 绘制并保存 Loss 曲线
    loss_curve_path = os.path.join(save_dir, 'loss_curve.png')
    plot_curve(train_losses, 'Epoch', 'Train Loss', 'Training Loss', loss_curve_path)
    print(f'Training loss curve saved to {loss_curve_path}')

    # 绘制并保存训练准确率曲线
    acc_curve_path = os.path.join(save_dir, 'accuracy_curve.png')
    plot_curve(train_accs, 'Epoch', 'Train Acc (%)', 'Training Accuracy', acc_curve_path)
    print(f'Training accuracy curve saved to {acc_curve_path}')

    # 绘制每类准确率条形图
    class_acc_path = os.path.join(save_dir, 'classwise_accuracy.png')
    plot_classwise_accuracy(model, test_loader, device, class_acc_path)
    print(f'Class-wise accuracy plot saved to {class_acc_path}')


if __name__ == '__main__':
    main()
