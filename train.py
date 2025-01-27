import os
import torch
import shutil
import numpy as np
import torch.nn as nn


import config
from model import OpenMax, PayloadEncoder, VAE
from dataset import SessionDataset
from utils import set_seed, stable

from test import test


def trainPLDFE(trainDataset, device):
    trainDataLoader = torch.utils.data.DataLoader(
        trainDataset, batch_size=32, shuffle=True
    )
    # 设置随机种子
    set_seed(config.Seed)

    pld_fea_enc = PayloadEncoder().cuda()
    lr = 0.001
    celoss = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(pld_fea_enc.parameters(), lr=lr, weight_decay=1e-4)

    print("Train Encoder")
    print("Epoch: [ 1 / 1 ]")
    actVecs = {}
    # 遍历良性流量样本
    for data in stable(trainDataLoader, 7):
        # 获取良性流量数据
        payload = data["payload"]
        subpayload = data["subpayload"]
        label = data["label"]

        # 进行模型推理
        activeVector, output = pld_fea_enc(payload.to(device), subpayload.to(device))
        # 计算损失
        loss = celoss(output, label.to(device))

        # 更新编码器参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        label_ = label.tolist()
        activeVector = activeVector.cpu().detach().numpy()
        for i in range(len(label_)):
            lb = label_[i]
            if lb not in actVecs:
                actVecs[lb] = []
            actVecs[lb].append(activeVector[i])

    openmax = OpenMax()
    openmax.fitWeibull(actVecs)

    # 保存模型
    torch.save(pld_fea_enc, config.PLDFE_PATH)
    torch.save(openmax, config.OPENMAX_PATH)


def trainVAE(trainDataset: SessionDataset, device):
    # 读取模型
    pld_fea_enc = torch.load(config.PLDFE_PATH)

    labelDatas = trainDataset.getLabelData()
    mseloss = nn.MSELoss().to(device)

    # 设置随机种子
    set_seed(config.Seed)

    vaes = {}
    for i in range(config.LabelNum):
        vae = VAE().cuda()
        opt_vae = torch.optim.AdamW(vae.parameters(), lr=0.001, weight_decay=1e-4)
        vaes[i] = [vae, opt_vae]

    print("Train VAE")
    for epoch in range(config.VAETrainEpoch):
        print("Epoch: [", epoch + 1, "/", config.VAETrainEpoch, "]")
        for i in range(config.LabelNum):
            PLD, SPD, LAB = labelDatas[i]
            labelDataset = SessionDataset(PLD, SPD, LAB)
            dataLoader = torch.utils.data.DataLoader(
                labelDataset, batch_size=16, shuffle=True
            )

            vae = vaes[i][0]
            opt_vae = vaes[i][1]

            errors = []
            # 遍历良性流量样本
            for data in stable(dataLoader, 7 + epoch):
                # 获取良性流量数据
                pld = data["payload"]
                sub = data["subpayload"]

                # 进行模型推理
                fusionFeature = pld_fea_enc.getFeature(
                    pld.to(device), sub.to(device)
                ).detach()

                z_x = vae.z_x(fusionFeature)

                loss = 0

                con_errors = []
                for j in range(5):
                    z = z_x.rsample()
                    recFeature = vae.x_z(z)

                    construction_error = torch.sum(
                        torch.pow(recFeature - fusionFeature, 2), 1
                    )
                    construction_error = list(construction_error.cpu().detach().numpy())
                    con_errors.append(construction_error)

                    # 计算损失
                    lossj = mseloss(recFeature, fusionFeature)
                    loss += lossj

                con_errors = np.array(con_errors)
                con_errors = np.mean(con_errors, axis=0)
                errors.extend(con_errors)

                loss /= 5

                # 更新编码器参数
                opt_vae.zero_grad()
                loss.backward()
                opt_vae.step()

            if i == 0:
                pos = int(len(errors) * config.BenignAlpha)
            else:
                pos = int(len(errors) * config.AttackAlpha)

            threshold = sorted(errors)[pos - 1]
            vae.threshold = threshold
            vaes[i] = [vae, opt_vae]

    for i in range(config.LabelNum):
        savePath = config.VAE_PATH + "VAE" + str(i) + ".pkl"
        torch.save(vaes[i][0], savePath)


def train(trainDataset, testDataset, device):
    # 创建文件夹
    if os.path.exists(config.ModelPath) == True:
        print("Deleting Old Model Dict...")
        shutil.rmtree(config.ModelPath)
    os.makedirs(config.ModelPath)
    os.mkdir(config.VAE_PATH)

    trainPLDFE(trainDataset, device)
    trainVAE(trainDataset, testDataset, device)
