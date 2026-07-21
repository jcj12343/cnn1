import sys
import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import torch

def eval_one_epoch(model, val_loader, criterion, dataset_len, device):
    model.eval()
    total_val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            val_correct += (pred == labels).sum().item()

    avg_loss = total_val_loss / len(val_loader)
    avg_acc = val_correct / dataset_len
    return avg_loss, avg_acc