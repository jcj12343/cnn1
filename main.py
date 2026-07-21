import sys
import os
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)

# 导入5个独立模块
from fix_dim.fix_dim import get_dataloaders, get_train_components
from train.train import train_one_epoch
from eval.eval import eval_one_epoch
from test_visual.test_visual import test_model
from plot_curve.plot_curve import draw_loss_acc_curve

# 全局超参
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3

if __name__ == "__main__":
    # 1. 创建result文件夹
    result_dir = os.path.join(root_path, "result")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # 2. 定义日志文件路径（放在主函数内，作用域匹配）
    log_file_path = os.path.join(result_dir, "log.txt")

    # 清空/新建日志文件
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"Training Start | Epochs:{EPOCHS}, Batch:{BATCH_SIZE}, LR:{LR}\n")
        f.write("="*60 + "\n")

    # Step1: 加载数据集、模型、损失、优化器
    train_loader, val_loader, test_loader, train_len, val_len, test_len = get_dataloaders(BATCH_SIZE)
    device, model, criterion, optimizer = get_train_components(LR)

    # 记录训练日志
    log_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # Step2: 循环训练+评估
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, train_len, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, val_len, device)

        log_history["train_loss"].append(train_loss)
        log_history["train_acc"].append(train_acc)
        log_history["val_loss"].append(val_loss)
        log_history["val_acc"].append(val_acc)

        # 控制台打印
        print(f"\n【Epoch {epoch+1:02d}/{EPOCHS}】")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # 写入日志文件
        log_text = f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss:{train_loss:.4f} Train Acc:{train_acc:.4f} | Val Loss:{val_loss:.4f} Val Acc:{val_acc:.4f}\n"
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(log_text)

    # Step3: 测试集预测+可视化（图片自动存入result）
    test_loss, test_acc = test_model(model, test_loader, criterion, test_len, device, result_dir)

    # 写入最终测试结果到日志
    final_text = "\n" + "="*50 + "\n"
    final_text += "Final Test Evaluation Result\n"
    final_text += f"Average Test Loss: {test_loss:.4f}\n"
    final_text += f"Test Accuracy: {test_acc:.4f}\n"
    final_text += "="*50 + "\n"
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(final_text)

    # Step4: 绘制loss/acc收敛曲线，存入result
    draw_loss_acc_curve(log_history, result_dir)
    print(f"\nAll results saved to folder: {result_dir}")
    print(f"Training log saved to: {log_file_path}")