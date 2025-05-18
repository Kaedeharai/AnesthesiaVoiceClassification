from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import torch
from typing import Union, Optional, List
import sys
import os


class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = os.path.join("log", filename)
        self.add_flag = add_flag
        # self.log = open(filename, 'a+')

    def write(self, message):
        if self.add_flag:
            with open(self.filename, 'a+') as log:
                self.terminal.write(message)
                log.write(message)
        else:
            with open(self.filename, 'w') as log:
                self.terminal.write(message)
                log.write(message)

    def flush(self):
        pass


class ConfusionMatrixCurve():
    def __init__(self, all_labels, all_preds, num_classes=2):
        self.all_labels = all_labels
        self.all_preds = all_preds
        self.num_classes = num_classes

    def plot(self, epoch, save_path):
        os.makedirs(save_path, exist_ok=True)
        cm = confusion_matrix(self.all_labels, self.all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[f'Class {i}' for i in range(self.num_classes)],
                    yticklabels=[f'Class {i}' for i in range(self.num_classes)])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(save_path, f'confusion_matrix_epoch_{epoch}.png'))
        plt.close()


class AccuracyCurve():
    def __init__(self, all_labels, all_preds):
        self.all_labels = all_labels
        self.all_preds = all_preds

    def plot(self, epoch, save_path):
        os.makedirs(save_path, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(self.all_labels, label='True Labels', color='blue')
        plt.plot(self.all_preds, label='Predicted Labels', color='orange')
        plt.xlabel('Sample Index')
        plt.ylabel('Label')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.savefig(os.path.join(save_path, f'accuracy_curve_{epoch}.png'))
        # plt.show()
        plt.close()


class PrecisionRecallCurve():
    def __init__(self, all_labels, all_preds):
        self.all_labels = all_labels
        self.all_preds = all_preds

    def plot(self, epoch, save_path):
        os.makedirs(save_path, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(self.all_labels, label='True Labels', color='blue')
        plt.plot(self.all_preds, label='Predicted Labels', color='orange')
        plt.xlabel('Sample Index')
        plt.ylabel('Label')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.savefig(os.path.join(save_path, f'precision_recall_curve_{epoch}.png'))
        # plt.show()
        plt.close()


def plot_confidence_distribution(
    confidences: Union[np.ndarray, torch.Tensor],  # 模型输出的置信度（每个样本的最大概率值）
    preds: Optional[Union[np.ndarray, torch.Tensor]] = None,  # 模型预测的类别索引（形状需与labels一致）
    labels: Optional[Union[np.ndarray, torch.Tensor]] = None,  # 真实标签（形状需与preds一致）
    plot_type: str = "histogram",  # 绘图类型：["histogram", "boxplot", "calibration_curve"]
    class_names: Optional[List[str]] = None,  # 类别名称列表（用于箱线图的横轴标签）
    split_correct: bool = False,  # 是否在直方图中区分正确/错误预测（仅当plot_type="histogram"时生效）
    save_path: Optional[str] = None,  # 图片保存路径（如 "confidence_plot.png"）
    **kwargs  # 其他参数（例如标题、颜色等）
) -> None:
    """
    绘制模型预测的置信度分布图

    参数说明：
    - confidences: 模型对每个样本预测的置信度（一维数组，取值范围 [0, 1]）
    - preds: 模型预测的类别索引（一维数组，可选）
    - labels: 真实标签（一维数组，可选）
    - plot_type: 
        - "histogram": 直方图（全局或分正确/错误）
        - "boxplot": 分类别的置信度箱线图
        - "calibration_curve": 校准曲线（需提供labels）
    - class_names: 类别名称列表（长度需等于类别总数）
    - split_correct: 在直方图中区分正确和错误预测（需提供preds和labels）
    - save_path: 图片保存路径
    - **kwargs: 额外参数（例如 title="My Plot", color="red"）
    """
    # 转换为 NumPy 数组（兼容 PyTorch Tensor）
    confidences = confidences.cpu().numpy() if isinstance(confidences, torch.Tensor) else confidences
    preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    # 检查输入合法性
    if plot_type == "calibration_curve" and labels is None:
        raise ValueError("校准曲线需要提供真实标签 `labels`")
    if split_correct and (preds is None or labels is None):
        raise ValueError("split_correct=True 需要提供 `preds` 和 `labels`")
    if plot_type == "boxplot" and labels is None:
        raise ValueError("箱线图需要提供真实标签 `labels`")

    # 初始化画布
    plt.figure(figsize=kwargs.get("figsize", (10, 6)))

    # --------------------------
    # 1. 绘制直方图
    # --------------------------
    if plot_type == "histogram":
        title = kwargs.get("title", "Confidence Distribution")
        color = kwargs.get("color", "skyblue")
        bins = kwargs.get("bins", 20)

        if split_correct:
            # 区分正确/错误预测的置信度
            correct_mask = (preds == labels)
            sns.histplot(
                data=[confidences[correct_mask], confidences[~correct_mask]],
                bins=bins,
                kde=True,
                label=["Correct Predictions", "Incorrect Predictions"],
                alpha=0.6,
                palette=["green", "red"]
            )
            title += " (Correct vs. Incorrect)"
        else:
            # 全局置信度分布
            sns.histplot(confidences, bins=bins, kde=True, color=color)
        
        plt.title(title)
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        if split_correct:
            plt.legend()

    # --------------------------
    # 2. 绘制分类别箱线图
    # --------------------------
    elif plot_type == "boxplot":
        # 确保类别名称与标签匹配
        unique_labels = np.unique(labels)
        if class_names is None:
            class_names = [f"Class {i}" for i in unique_labels]
        elif len(class_names) != len(unique_labels):
            raise ValueError("class_names 长度与类别数量不匹配")

        # 构建 DataFrame
        import pandas as pd
        df = pd.DataFrame({
            "Class": [class_names[label] for label in labels],
            "Confidence": confidences
        })

        sns.boxplot(x="Class", y="Confidence", data=df, palette=kwargs.get("palette", "Set3"))
        plt.title(kwargs.get("title", "Confidence Distribution per Class"))
        plt.xlabel("Class")
        plt.ylabel("Confidence")
        plt.xticks(rotation=45)

    # --------------------------
    # 3. 绘制校准曲线
    # --------------------------
    elif plot_type == "calibration_curve":
        prob_true, prob_pred = calibration_curve(labels, confidences, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label="Model")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Perfectly Calibrated")
        plt.title(kwargs.get("title", "Calibration Curve"))
        plt.xlabel("Mean Predicted Confidence")
        plt.ylabel("Fraction of Positives")
        plt.legend()

    else:
        raise ValueError(f"不支持的绘图类型: {plot_type}")

    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"图表已保存至 {save_path}")
    else:
        plt.show()