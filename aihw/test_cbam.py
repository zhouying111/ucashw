# test_cbam.py

import os
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from CBAM import CBAM
from train_cbam import get_cbam_resnet  # 直接复用定义


def get_test_loader(batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return test_loader, testset.classes


def main():
    parser = argparse.ArgumentParser(description='Test ResNet+CBAM on CIFAR-10')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet34'],
                        help='ResNet architecture used in training')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to saved .pth checkpoint file')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--gpuid', type=int, default=0, help='GPU id to use')
    args = parser.parse_args()

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}, ResNet: {args.arch}, GPUID: {args.gpuid}')

    # 构建模型 & 加载权重
    model = get_cbam_resnet(name=args.arch, num_classes=10).to(device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    # 测试集加载
    test_loader, classes = get_test_loader(batch_size=args.batch_size,
                                           num_workers=args.num_workers)

    # 评估
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc='Testing'):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    acc = 100.0 * correct / total
    print(f'\nTest Loss: {avg_loss:.4f} | Test Accuracy: {acc:.2f}%')

    # 打印每类准确率
    class_correct = {cls: 0 for cls in classes}
    class_total   = {cls: 0 for cls in classes}

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            for label, pred in zip(labels, preds):
                cls_name = classes[label]
                class_total[cls_name] += 1
                if label == pred:
                    class_correct[cls_name] += 1

    print('\nPer-class Accuracy:')
    for cls in classes:
        acc_cls = 100.0 * class_correct[cls] / class_total[cls]
        print(f'  {cls:5s}: {acc_cls:5.2f}%')


if __name__ == '__main__':
    from tqdm import tqdm
    main()
