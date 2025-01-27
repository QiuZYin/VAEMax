import torch
import numpy as np


import config
from utils import calculate_result, draw_confusion_matrix


def test(testDataset, device):
    pld_fea_enc = torch.load(config.PLDFE_PATH)
    openmax = torch.load(config.OPENMAX_PATH)
    vaes = []
    for i in range(config.LabelNum):
        savePath = config.VAE_PATH + "VAE" + str(i) + ".pkl"
        vae = torch.load(savePath)
        vaes.append(vae)

    # 构建数据
    testDataLoader = torch.utils.data.DataLoader(
        testDataset, batch_size=128, shuffle=False
    )
    unknownIdx = config.LabelNum
    true_label = []
    pred_label = []
    # 不保存梯度信息
    with torch.no_grad():
        # 遍历测试集
        for data in testDataLoader:
            # 获取良性流量数据
            payload = data["payload"]
            subpayload = data["subpayload"]
            label = data["label"]

            # 进行模型推理
            activeVector, output = pld_fea_enc(
                payload.to(device), subpayload.to(device)
            )
            openmax_pred = openmax.recalibrate(
                activeVector.cpu().detach().numpy(),
                output.cpu().detach().numpy(),
            )
            openmax_pred = np.argmax(openmax_pred, axis=1).tolist()

            fusionFeature = pld_fea_enc.getFeature(
                payload.to(device), subpayload.to(device)
            )
            for i in range(len(openmax_pred)):
                lab = openmax_pred[i]
                if lab == len(config.LabelList) - 1:
                    continue

                vae = vaes[lab]
                threshold = vae.threshold

                z_x = vae.z_x(fusionFeature[i])

                for j in range(5):
                    z = z_x.rsample()
                    recFeature = vae.x_z(z)
                    construction_error = torch.sum(
                        torch.pow(fusionFeature[i] - recFeature, 2)
                    )
                    if construction_error >= threshold:
                        openmax_pred[i] = unknownIdx
                        break

            pred_label.extend(openmax_pred)
            true_label.extend(label.tolist())

    calculate_result(true_label, pred_label)
    draw_confusion_matrix(true_label, pred_label)


def testWOvae(testDataset, device):
    pld_fea_enc = torch.load(config.PLDFE_PATH)
    openmax = torch.load(config.OPENMAX_PATH)

    # 构建数据
    testDataLoader = torch.utils.data.DataLoader(
        testDataset, batch_size=128, shuffle=False
    )
    true_label = []
    pred_label = []
    # 不保存梯度信息
    with torch.no_grad():
        # 遍历测试集
        for data in testDataLoader:
            # 获取良性流量数据
            payload = data["payload"]
            subpayload = data["subpayload"]
            label = data["label"]

            # 进行模型推理
            activeVector, output = pld_fea_enc(
                payload.to(device), subpayload.to(device)
            )
            openmax_pred = openmax.recalibrate(
                activeVector.cpu().detach().numpy(),
                output.cpu().detach().numpy(),
            )
            openmax_pred = np.argmax(openmax_pred, axis=1).tolist()

            pred_label.extend(openmax_pred)
            true_label.extend(label.tolist())

    calculate_result(true_label, pred_label)


def testWOevt(testDataset, device):
    pld_fea_enc = torch.load(config.PLDFE_PATH)
    openmax = torch.load(config.OPENMAX_PATH)
    vaes = []
    for i in range(config.LabelNum):
        savePath = config.VAE_PATH + "VAE" + str(i) + ".pkl"
        vae = torch.load(savePath)
        vaes.append(vae)

    # 构建数据
    testDataLoader = torch.utils.data.DataLoader(
        testDataset, batch_size=128, shuffle=False
    )
    unknownIdx = config.LabelNum
    true_label = []
    pred_label = []
    # 不保存梯度信息
    with torch.no_grad():
        # 遍历测试集
        for data in testDataLoader:
            # 获取良性流量数据
            payload = data["payload"]
            subpayload = data["subpayload"]
            label = data["label"]

            # 进行模型推理
            activeVector, output = pld_fea_enc(
                payload.to(device), subpayload.to(device)
            )
            output = np.argmax(output.cpu().detach().numpy(), axis=1).tolist()

            fusionFeature = pld_fea_enc.getFeature(
                payload.to(device), subpayload.to(device)
            )
            for i in range(len(output)):
                lab = output[i]
                if lab == len(config.LabelList) - 1:
                    continue

                vae = vaes[lab]
                threshold = vae.threshold

                z_x = vae.z_x(fusionFeature[i])

                for j in range(5):
                    z = z_x.rsample()
                    recFeature = vae.x_z(z)
                    construction_error = torch.sum(
                        torch.pow(fusionFeature[i] - recFeature, 2)
                    )
                    if construction_error >= threshold:
                        output[i] = unknownIdx
                        break

            pred_label.extend(output)
            true_label.extend(label.tolist())

    calculate_result(true_label, pred_label)
