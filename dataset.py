import os
import torch
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

import config


class SessionDataset(Dataset):
    def __init__(self, payload, subpayload, label):
        self.payload = payload
        self.subpayload = subpayload
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        return {
            "payload": torch.tensor(self.payload[i], dtype=torch.float32),
            "subpayload": torch.tensor(self.subpayload[i], dtype=torch.float32),
            "label": torch.tensor(self.label[i], dtype=torch.long),
        }

    def getLabelData(self):
        labelDatas = {}
        for i in range(len(self.label)):
            lab = self.label[i]
            if lab not in labelDatas:
                labelDatas[lab] = [[], [], []]

            labelDatas[lab][0].append(self.payload[i])
            labelDatas[lab][1].append(self.subpayload[i])
            labelDatas[lab][2].append(self.label[i])

        return labelDatas


def getDatasets():
    statisticsPath = config.StatisticsPath
    payloadPath = config.PayloadPath
    subpayloadPath = config.SubPayloadPath
    # 统计特征文件夹路径
    files = os.listdir(payloadPath)

    # 训练集
    trainCategory = []
    trainPLD = []
    trainSPD = []
    # 测试集
    testCategory = []
    testPLD = []
    testSPD = []

    MAX_SESS_LEN = config.MAX_SESS_LEN

    # 遍历所有文件
    for f in files:
        # if f == "Benign.csv":
        #     continue
        # 获取文件路径
        statisticsFile = os.path.join(statisticsPath, f)
        payloadFile = os.path.join(payloadPath, f)
        subpayloadFile = os.path.join(subpayloadPath, f)

        # 获取文件内容
        statisticsFile = np.array(pd.read_csv(statisticsFile, header=None))
        payloadFile = np.array(pd.read_csv(payloadFile, header=None))
        subpayloadFile = np.array(pd.read_csv(subpayloadFile, header=None))

        # 数据量
        dataNum = statisticsFile.shape[0]
        # 类别
        category = f[:-4]

        # 在未知攻击文件列表
        if category in config.UnknownAttackSet:
            # 添加到测试集中
            for idx in range(dataNum):
                l = MAX_SESS_LEN * idx
                r = MAX_SESS_LEN * (idx + 1)
                testCategory.append("Unknown")
                testPLD.append(payloadFile[l:r])
                testSPD.append(subpayloadFile[l:r])
        else:
            indexList = [i for i in range(dataNum)]
            random.shuffle(indexList)
            trainPos = int(0.8 * dataNum)
            testPos = int(0.8 * dataNum)

            for i in range(trainPos):
                idx = indexList[i]
                l = MAX_SESS_LEN * idx
                r = MAX_SESS_LEN * (idx + 1)
                trainCategory.append(statisticsFile[idx][378])
                trainPLD.append(payloadFile[l:r])
                trainSPD.append(subpayloadFile[l:r])

            for i in range(testPos, dataNum):
                idx = indexList[i]
                l = MAX_SESS_LEN * idx
                r = MAX_SESS_LEN * (idx + 1)
                testCategory.append(statisticsFile[idx][378])
                testPLD.append(payloadFile[l:r])
                testSPD.append(subpayloadFile[l:r])

    # 转成numpy.array格式
    trainPLD = np.array(trainPLD, dtype=float)
    trainSPD = np.array(trainSPD, dtype=float)
    testPLD = np.array(testPLD, dtype=float)
    testSPD = np.array(testSPD, dtype=float)

    """------------------- PLD -------------------"""
    # 对数据集进行标准化
    trainPLD = trainPLD / 256
    testPLD = testPLD / 256
    """------------------- PLD -------------------"""

    """------------------- SPD -------------------"""
    # 变换形状
    trainSPD = trainSPD.reshape((-1, 12))
    testSPD = testSPD.reshape((-1, 12))
    # 获取已知攻击的均值和方差
    SPDscale = StandardScaler().fit(trainSPD)
    # 对数据集进行标准化
    trainSPD = SPDscale.transform(trainSPD)
    testSPD = SPDscale.transform(testSPD)
    # 变换形状
    trainSPD = trainSPD.reshape((-1, MAX_SESS_LEN, 12))
    testSPD = testSPD.reshape((-1, MAX_SESS_LEN, 12))
    """------------------- SPD -------------------"""

    """------------------- CAT -------------------"""
    totalCategory = config.LabelList
    Cate2Label = {}
    idx = 0
    for cate in totalCategory:
        Cate2Label[cate] = idx
        idx += 1

    trainLabel = []
    testLabel = []
    for cate in trainCategory:
        trainLabel.append(Cate2Label[cate])
    for cate in testCategory:
        testLabel.append(Cate2Label[cate])

    trainLabel = np.array(trainLabel)
    testLabel = np.array(testLabel)
    """------------------- CAT -------------------"""

    # 合成会话流数据集
    trainDataset = SessionDataset(trainPLD, trainSPD, trainLabel)
    testDataset = SessionDataset(testPLD, testSPD, testLabel)

    return trainDataset, testDataset
