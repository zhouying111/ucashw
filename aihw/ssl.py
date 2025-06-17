# 自监督
# simclr_cifar10.py
# 简单复现 SimCLR，模块化: dataloader, model, train, linear evaluation with argparse超参

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import get_model
from utils import plot_curve


# -----------------------------
# 1. DataLoader Module
# -----------------------------
class TwoCropTransform:
    """Create two augmented versions of an image."""
    def __init__(self, base_transform):
        self.base = base_transform
    def __call__(self, x):
        return self.base(x), self.base(x)


def get_simclr_dataloaders(batch_size, num_workers):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True,
        transform=TwoCropTransform(transform)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True
    )

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True,
        transform=test_transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers
    )
    return train_loader, test_loader


# def get_cifar_resnet18(pretrained=False):
#     model = torchvision.models.resnet18(pretrained=pretrained)
#     model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#     model.maxpool = nn.Identity()
#     return model


# -----------------------------
# 2. Model Module
# -----------------------------
class SimCLR(nn.Module):
    def __init__(self, base_encoder, proj_dim=128):
        super().__init__()
        self.encoder = base_encoder
        self.feature_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, proj_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)


# -----------------------------
# 3. Loss Function
# -----------------------------
def nt_xent_loss(z1, z2, temperature=0.3):
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    mask = (~torch.eye(2 * batch_size, dtype=bool, device=z.device))
    exp_sim = torch.exp(sim) * mask
    pos = torch.cat([
        torch.exp((z1 * z2).sum(dim=-1) / temperature),
        torch.exp((z2 * z1).sum(dim=-1) / temperature)
    ], dim=0)
    loss = -torch.log(pos / exp_sim.sum(dim=1))
    return loss.mean()


# -----------------------------
# 4. Training Module for SimCLR
# -----------------------------
def train_simclr(model, train_loader, optimizer, epochs, device, save_dir):
    model.to(device)
    os.makedirs(save_dir, exist_ok=True)
    loss_curve, acc_curve = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_pairs = 0

        for (x1, x2), _ in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', leave=False):
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            z1, z2 = model(x1), model(x2)
            loss = nt_xent_loss(z1, z2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            preds = torch.argmax(torch.matmul(z1, z2.T), dim=1)
            total_correct += torch.sum(preds == torch.arange(z1.size(0), device=device)).item()
            total_pairs += z1.size(0)

        avg_loss = total_loss / len(train_loader)
        acc = 100.0 * total_correct / total_pairs
        loss_curve.append(avg_loss)
        acc_curve.append(acc)

        print(f'Epoch {epoch} Loss: {avg_loss:.4f}, Acc: {acc:.2f}%')

    torch.save({
        'state_dict': model.encoder.state_dict(),
        'feature_dim': model.feature_dim
    }, os.path.join(save_dir, 'simclr_encoder.pth'))

    plot_curve(loss_curve, 'Epoch', 'Loss', 'SimCLR Training Loss', os.path.join(save_dir, 'simclr_loss_curve.png'))
    plot_curve(acc_curve,  'Epoch', 'Accuracy (%)', 'SimCLR Training Accuracy', os.path.join(save_dir, 'simclr_acc_curve.png'))
    print(f"SimCLR curves saved in {save_dir}")


# -----------------------------
# 5. Test Module (Linear Evaluation)
# -----------------------------
class LinearClassifier(nn.Module):
    def __init__(self, encoder, feature_dim, num_classes=10):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


def evaluate_linear(encoder, feature_dim, train_loader, test_loader,
                    epochs, lr, device, save_dir):
    encoder.to(device)
    os.makedirs(save_dir, exist_ok=True)

    model = LinearClassifier(encoder, feature_dim).to(device)
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    train_loss_curve, train_acc_curve = [], []
    test_loss_curve, test_acc_curve   = [], []
    best_test_acc = 0.0
    best_train_acc_at_best_test = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_acc, total_samples = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100.0 * total_acc / total_samples
        train_loss_curve.append(avg_train_loss)
        train_acc_curve.append(train_acc)

        model.eval()
        total_loss, total_acc, total_samples = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                total_acc += (outputs.argmax(1) == labels).sum().item()
                total_samples += labels.size(0)
        avg_test_loss = total_loss / len(test_loader)
        test_acc = 100.0 * total_acc / total_samples
        test_loss_curve.append(avg_test_loss)
        test_acc_curve.append(test_acc)

        print(f'Linear Epoch {epoch} | Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%')

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_train_acc_at_best_test = train_acc

    print(f'\nBest Linear Test Acc: {best_test_acc:.2f}%')
    print(f'Corresponding Train Acc: {best_train_acc_at_best_test:.2f}%')

    plot_curve(train_loss_curve, 'Epoch', 'Train Loss', 'Linear Train Loss', os.path.join(save_dir, 'linear_train_loss.png'))
    plot_curve(train_acc_curve,  'Epoch', 'Train Acc (%)', 'Linear Train Accuracy', os.path.join(save_dir, 'linear_train_acc.png'))
    plot_curve(test_loss_curve,   'Epoch', 'Test Loss',  'Linear Test Loss',  os.path.join(save_dir, 'linear_test_loss.png'))
    plot_curve(test_acc_curve,    'Epoch', 'Test Acc (%)',  'Linear Test Accuracy',  os.path.join(save_dir, 'linear_test_acc.png'))
    print(f"Linear evaluation curves saved in {save_dir}")

    plot_classwise_accuracy(model, test_loader, device, os.path.join(save_dir, 'linear_classwise_accuracy.png'))
    print(f"Linear class-wise accuracy plot saved in {save_dir}")


def plot_classwise_accuracy(model, loader, device, save_path):
    """
    计算并绘制每个类别在测试集上的准确率条形图（CIFAR-10）
    """
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    correct_pred = {cls: 0 for cls in classes}
    total_pred   = {cls: 0 for cls in classes}

    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Classwise Acc", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            for label, pred in zip(labels, preds):
                cls = classes[label]
                if label == pred:
                    correct_pred[cls] += 1
                total_pred[cls] += 1

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
# 6. Main with argparse
# -----------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SimCLR + Linear Evaluation on CIFAR-10')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'], help='backbone architecture')
    parser.add_argument('--batch_size', type=int,   default=512, help='batch size for SimCLR pretraining')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs for SimCLR')
    parser.add_argument('--simclr_lr',     type=float, default=1e-3, help='learning rate for SimCLR (Adam)')
    parser.add_argument('--linear_epochs', type=int, default=10, help='number of epochs for linear eval')
    parser.add_argument('--linear_lr',     type=float, default=0.01, help='learning rate for linear eval (SGD)')
    parser.add_argument('--proj_dim',      type=int, default=512, help='projection dimension for SimCLR')
    parser.add_argument('--num_workers',   type=int, default=8, help='number of DataLoader workers')
    parser.add_argument('--gpuid',         type=int, default=0, help='GPU id to use')
    parser.add_argument('--simclr_save_dir', type=str, default='simclr_results', help='save directory for SimCLR')
    parser.add_argument('--linear_save_dir', type=str, default='ssl_res', help='save directory for linear eval')
    args = parser.parse_args()

    save_dir = os.path.join(args.simclr_save_dir, args.arch)
    os.makedirs(save_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device} (GPUID={args.gpuid})')

    # 1) SimCLR 预训练
    simclr_train_loader, simclr_test_loader = get_simclr_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    # base_encoder = get_model(pretrained=False)
    base_encoder = get_model(name=args.arch, num_classes=10).to(device)
    simclr_model = SimCLR(base_encoder, proj_dim=args.proj_dim)
    simclr_optimizer = torch.optim.Adam(simclr_model.parameters(), lr=args.simclr_lr)

    train_simclr(
        simclr_model,
        simclr_train_loader,
        simclr_optimizer,
        epochs=args.epochs,
        device=device,
        save_dir=save_dir
    )

    # 2) Linear evaluation
    # encoder = get_cifar_resnet18(pretrained=False)
    encoder = get_model(name=args.arch, num_classes=10).to(device)
    ckpt = torch.load(os.path.join(save_dir, 'simclr_encoder.pth'), map_location='cpu')
    feature_dim = ckpt['feature_dim']
    encoder.fc = nn.Identity()
    encoder.load_state_dict(ckpt['state_dict'])

    linear_train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    ])
    linear_train_loader = DataLoader(
        torchvision.datasets.CIFAR10('./data', train=True, download=False, transform=linear_train_transform),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    linear_test_loader = simclr_test_loader

    evaluate_linear(
        encoder,
        feature_dim,
        linear_train_loader,
        linear_test_loader,
        epochs=args.linear_epochs,
        lr=args.linear_lr,
        device=device,
        save_dir=save_dir
    )
