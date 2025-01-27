import copy
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import config


def set_seed(seed):
    random.seed(seed)
    SEED = random.randint(1, 1000)
    torch.manual_seed(SEED)


def stable(dataloader, seed):
    set_seed(seed)
    return dataloader


def calculate_result(true_label, pred_label):
    true_label = copy.deepcopy(true_label)
    pred_label = copy.deepcopy(pred_label)

    # 构建混淆矩阵标签集合
    labels = [i for i in range(len(config.LabelList))]

    # 混淆矩阵
    cm = confusion_matrix(y_true=true_label, y_pred=pred_label, labels=labels)

    # 二分类
    binary_result = [[0, 0], [0, 0]]
    binary_result[0][0] = cm[0][0]
    for i in range(1, config.LabelNum + 1):
        binary_result[0][1] += cm[0][i]
        binary_result[1][0] += cm[i][0]
        for j in range(1, config.LabelNum + 1):
            binary_result[1][1] += cm[i][j]

    ACC = (binary_result[0][0] + binary_result[1][1]) / np.sum(binary_result)
    precision = binary_result[0][0] / (binary_result[0][0] + binary_result[1][0])
    recall = binary_result[0][0] / (binary_result[0][0] + binary_result[0][1])
    F1 = (2 * precision * recall) / (precision + recall)

    print("binary")
    print(binary_result[0])
    print(binary_result[1])
    print("ACC: %.4f" % ACC)
    print(" F1: %.4f" % F1)
    print()

    # 多分类
    multi_cr = classification_report(
        y_true=true_label,
        y_pred=pred_label,
        labels=labels,
        target_names=config.LabelList,
        digits=4,
    )
    print("multi")
    print(cm, "\n")
    print(multi_cr, "\n")


def modifyLabel(label_true, label_pred):
    labelDict = {
        "Benign": "Benign",
        "FTP Brute Force": "FTP BF",
        "SSH Brute Force": "SSH BF",
        "DoS Hulk": "Hulk",
        "DoS Slowloris": "Slowloris",
        "DDoS LOIT": "LOIT",
        "Port Scan": "Port Scan",
        "DDoS HOIC": "HOIC",
        "Unknown": "Unknown",
    }

    label_name = copy.deepcopy(config.LabelList)
    for i in range(len(label_name)):
        label_name[i] = labelDict[label_name[i]]
    for i in range(len(label_true)):
        label_true[i] = label_name[label_true[i]]
    for i in range(len(label_pred)):
        label_pred[i] = label_name[label_pred[i]]

    return label_true, label_pred, label_name


def draw_confusion_matrix(
    label_true,
    label_pred,
):
    label_true = copy.deepcopy(label_true)
    label_pred = copy.deepcopy(label_pred)
    label_true, label_pred, label_name = modifyLabel(label_true, label_pred)
    cm = confusion_matrix(
        y_true=label_true, y_pred=label_pred, labels=label_name, normalize="true"
    )

    plt.imshow(cm, cmap="Blues")
    plt.xlabel("Predict label")
    plt.ylabel("Truth label")
    plt.yticks(range(label_name.__len__()), label_name)
    plt.xticks(range(label_name.__len__()), label_name, rotation=90)

    plt.tight_layout()
    plt.colorbar()

    for i in range(label_name.__len__()):
        for j in range(label_name.__len__()):
            value = float(format("%.2f" % cm[j, i]))
            color = (1, 1, 1) if value > 0.7 else (0, 0, 0)  # 对角线字体白色, 其他黑色
            plt.text(
                i,
                j,
                value,
                verticalalignment="center",
                horizontalalignment="center",
                color=color,
                fontsize=9,
            )

    savePath = config.Dataset + ".jpg"
    plt.savefig(savePath, bbox_inches="tight", dpi=1000)
    plt.close("all")
