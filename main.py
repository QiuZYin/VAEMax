import time
import torch
import random

import config
from dataset import getDatasets
from train import train
from test import test, testWOvae, testWOevt


if __name__ == "__main__":
    # 实验数据集
    print("Dataset:", config.Dataset)
    # 实验设置
    print("Seed:", config.Seed)
    random.seed(config.Seed)

    # 指定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 记录开始时间
    st = time.time()

    # 获取数据
    print("\nProcessing Data...")
    trainDataset, testDataset = getDatasets()

    # train(trainDataset, device)

    test(testDataset, device)
    testWOvae(testDataset, device)
    testWOevt(testDataset, device)

    # 记录开始时间
    et = time.time()

    # 输出总耗时
    print("\nTotal Time:", et - st)
