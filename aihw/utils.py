
def accuracy(output, target):
    return (output.argmax(1) == target).float().mean().item()

# -----------------------------
# 4. 保存并绘制曲线
# -----------------------------
def plot_curve(values, xlabel, ylabel, title, save_path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, len(values) + 1), values, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

