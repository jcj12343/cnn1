import sys
import os
import torch
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from fix_dim.fix_dim import get_device

def train_one_epoch(model, train_loader, criterion, optimizer, dataset_len, device):
    model.train()
    total_train_loss = 0.0
    train_correct = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        pred = torch.argmax(outputs, dim=1)
        train_correct += (pred == labels).sum().item()

    avg_loss = total_train_loss / len(train_loader)
    avg_acc = train_correct / dataset_len
    return avg_loss, avg_acc