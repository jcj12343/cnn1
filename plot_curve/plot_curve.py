import matplotlib.pyplot as plt
import os

def draw_loss_acc_curve(log_history, save_dir):
    plt.figure(figsize=(12, 5))
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(log_history["train_loss"], label="Train Loss")
    plt.plot(log_history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.title("Loss Curve")

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(log_history["train_acc"], label="Train Acc")
    plt.plot(log_history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy Curve")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "loss_accuracy_curve.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Loss & Accuracy curve saved to: {save_path}")