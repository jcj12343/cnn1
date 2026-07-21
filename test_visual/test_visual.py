import sys
import os
import torch
import matplotlib.pyplot as plt

def test_model(model, test_loader, criterion, test_len, device, save_dir):
    model.eval()
    test_correct = 0
    test_total_loss = 0.0
    sample_imgs, sample_true, sample_pred = [], [], []

    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(test_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            test_total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            test_correct += (preds == labels).sum().item()

            if idx == 0:
                sample_imgs = imgs[:5].cpu()
                sample_true = labels[:5].cpu().numpy()
                sample_pred = preds[:5].cpu().numpy()

    avg_test_loss = test_total_loss / len(test_loader)
    test_acc = test_correct / test_len

    print("\n" + "="*50)
    print("【Final Test Evaluation Result】")
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("="*50)
    if test_acc < 0.95:
        print("Warning: Test accuracy is lower than 95%, need to optimize model or training hyperparameters")

    # 绘制5张测试样本，自动保存到result文件夹
    plt.figure(figsize=(12, 3))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(sample_imgs[i].squeeze(), cmap="gray")
        plt.title(f"True:{sample_true[i]}\nPred:{sample_pred[i]}")
        plt.axis("off")
    plt.suptitle("Visualization of 5 Test Samples")
    save_path = os.path.join(save_dir, "test_samples.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Test sample image saved to: {save_path}")

    return avg_test_loss, test_acc