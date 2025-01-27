"""
先训练cnn, 再训练vae, 每个类型构建自己的vae
"""

import time
import libmr
import torch
import random
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix


import config
from dataset import getDatasets, SessionDataset
from model import PayloadEncoder, VAE
from utils import set_seed, stable, calculate_result


def openmax(act_vecs):
    labelNum = len(config.LabelList) - 1
    weibull_models = {}
    tailalpha = 0.05
    for i in range(labelNum):
        weibull_models[i] = {}

        act_vec = act_vecs[i]
        act_vec = np.array(act_vec)
        mean_vec = np.mean(act_vec, axis=0)

        dist = []
        for vec in act_vec:
            dis = np.sqrt(np.sum(np.square(vec - mean_vec)))
            dist.append(dis)

        tailsize = int(len(dist) * tailalpha)
        tailsize = max(10, tailsize)
        tailtofit = sorted(dist)[-tailsize:]

        mr = libmr.MR()
        mr.fit_high(tailtofit, len(tailtofit))

        weibull_models[i]["mean_vec"] = mean_vec
        weibull_models[i]["weibull_model"] = mr

    return weibull_models


def recalibrate_pred(weibull_models, act_vecs, preds):
    labelNum = len(config.LabelList) - 1
    dataNum = act_vecs.shape[0]
    alpha_weights = [(labelNum - i - 2) / float(labelNum) for i in range(labelNum)]
    attenuation = [config.Attenuation for i in range(labelNum)]
    openmax_scores = []

    for i in range(dataNum):
        act_vec = act_vecs[i]
        pred = preds[i]
        ranked_list = pred.argsort().ravel()[::-1]
        ranked_alpha = [0 for j in range(labelNum)]
        for j in range(len(alpha_weights)):
            ranked_alpha[ranked_list[j]] = alpha_weights[j]

        openmax_score = []
        openmax_unknown = 0
        for j in range(labelNum):
            mean_vec = weibull_models[j]["mean_vec"]
            mr = weibull_models[j]["weibull_model"]
            dis = np.sqrt(np.sum(np.square(act_vec[j] - mean_vec)))
            wscore = mr.w_score(dis)
            modified = pred[j] * (1 - wscore * ranked_alpha[j] * attenuation[j])
            openmax_score.append(modified)
            openmax_unknown += pred[j] - modified
        openmax_score.append(openmax_unknown)

        openmax_scores.append(openmax_score)

    return openmax_scores


def test(testDataset, session, weibull_models, vaes, device):
    print("all")
    testDataLoader = torch.utils.data.DataLoader(
        testDataset, batch_size=128, shuffle=False
    )
    unknownIdx = len(config.LabelList) - 1
    true_label = []
    pred_label = []
    with torch.no_grad():
        for data in testDataLoader:
            payload = data["payload"]
            subpayload = data["subpayload"]
            label = data["label"]

            activeVector, output = session(payload.to(device), subpayload.to(device))
            openmax_pred = recalibrate_pred(
                weibull_models,
                activeVector.cpu().detach().numpy(),
                output.cpu().detach().numpy(),
            )
            openmax_pred = np.argmax(openmax_pred, axis=1).tolist()

            fusionFeature = session.getFeature(
                payload.to(device), subpayload.to(device)
            )
            for i in range(len(openmax_pred)):
                lab = openmax_pred[i]
                if lab == len(config.LabelList) - 1:
                    continue

                vae = vaes[lab][0]
                threshold = vaes[lab][2]

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

    # labels = [i for i in range(len(config.LabelList))]

    # results = confusion_matrix(y_true=true_label, y_pred=pred_label, labels=labels)

    # print(results)
    # right = np.sum(np.diagonal(results))
    # total = np.sum(results)
    # print("ACC", right / total)

    calculate_result(true_label, pred_label)


def testWOvae(testDataset, session, weibull_models, device):
    print("wo vae")
    testDataLoader = torch.utils.data.DataLoader(
        testDataset, batch_size=128, shuffle=False
    )
    true_label = []
    pred_label = []
    with torch.no_grad():
        for data in testDataLoader:
            payload = data["payload"]
            subpayload = data["subpayload"]
            label = data["label"]

            activeVector, output = session(payload.to(device), subpayload.to(device))
            openmax_pred = recalibrate_pred(
                weibull_models,
                activeVector.cpu().detach().numpy(),
                output.cpu().detach().numpy(),
            )
            openmax_pred = np.argmax(openmax_pred, axis=1).tolist()

            pred_label.extend(openmax_pred)
            true_label.extend(label.tolist())

    labels = [i for i in range(len(config.LabelList))]

    results = confusion_matrix(y_true=true_label, y_pred=pred_label, labels=labels)

    print(results)
    right = np.sum(np.diagonal(results))
    total = np.sum(results)
    print("ACC", right / total)


def testWOevt(testDataset, session, vae, threshold, device):
    print("wo evt")
    testDataLoader = torch.utils.data.DataLoader(
        testDataset, batch_size=128, shuffle=False
    )
    unknownIdx = len(config.LabelList) - 1
    true_label = []
    pred_label = []
    with torch.no_grad():
        for data in testDataLoader:
            payload = data["payload"]
            subpayload = data["subpayload"]
            label = data["label"]

            activeVector, output = session(payload.to(device), subpayload.to(device))
            softmax_pred = np.argmax(output.cpu().detach().numpy(), axis=1).tolist()

            fusionFeature = session.getFeature(
                payload.to(device), subpayload.to(device)
            )
            for i in range(len(softmax_pred)):
                if softmax_pred[i] != 0:
                    continue

                z_x = vae.z_x(fusionFeature[i])

                for j in range(5):
                    z = z_x.rsample()
                    recFeature = vae.x_z(z)
                    construction_error = torch.sum(
                        torch.pow(fusionFeature[i] - recFeature, 2)
                    )
                    if construction_error >= threshold:
                        softmax_pred[i] = unknownIdx
                        break

            pred_label.extend(softmax_pred)
            true_label.extend(label.tolist())

    labels = [i for i in range(len(config.LabelList))]

    results = confusion_matrix(y_true=true_label, y_pred=pred_label, labels=labels)

    print(results)


def trainVAE(trainDataset, testDataset, session, weibull_models, device):
    set_seed(config.Seed)

    labelDatas = trainDataset.getLabelData()

    mseloss = nn.MSELoss().to(device)

    vaes = {}
    for i in range(len(config.LabelList) - 1):
        vae = VAE().cuda()
        opt_vae = torch.optim.AdamW(vae.parameters(), lr=0.001, weight_decay=1e-4)
        vaes[i] = [vae, opt_vae, 0]

    TrainEpoch = config.VAETrainEpoch
    print("Train VAE")
    for epoch in range(TrainEpoch):
        print("Epoch: [", epoch + 1, "/", TrainEpoch, "]")
        for i in range(len(config.LabelList) - 1):
            PLD, SPD, LAB = labelDatas[i]
            labelDataset = SessionDataset(PLD, SPD, LAB)
            dataLoader = torch.utils.data.DataLoader(
                labelDataset, batch_size=16, shuffle=True
            )

            vae = vaes[i][0]
            opt_vae = vaes[i][1]

            errors = []
            for data in stable(dataLoader, 7 + epoch):
                pld = data["payload"]
                sub = data["subpayload"]

                fusionFeature = session.getFeature(
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

                    lossj = mseloss(recFeature, fusionFeature)
                    loss += lossj

                con_errors = np.array(con_errors)
                con_errors = np.mean(con_errors, axis=0)
                errors.extend(con_errors)

                loss /= 5

                opt_vae.zero_grad()
                loss.backward()
                opt_vae.step()

            if i == 0:
                pos = int(len(errors) * config.BenignAlpha)
            else:
                pos = int(len(errors) * config.AttackAlpha)

            threshold = sorted(errors)[pos - 1]

            vaes[i] = [vae, opt_vae, threshold]

        test(testDataset, session, weibull_models, vaes, device)

    # testWOvae(testDataset, session, weibull_models, device)
    # testWOevt(testDataset, session, vae, threshold, device)


def train(trainDataset, testDataset, device):
    trainDataLoader = torch.utils.data.DataLoader(
        trainDataset, batch_size=32, shuffle=True
    )
    set_seed(config.Seed)

    session = PayloadEncoder().cuda()
    lr = 0.001
    celoss = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(session.parameters(), lr=lr, weight_decay=1e-4)

    print("Train Encoder")
    print("Epoch: [ 1 / 1 ]")
    actVecs = {}
    for data in stable(trainDataLoader, 7):
        payload = data["payload"]
        subpayload = data["subpayload"]
        label = data["label"]

        activeVector, output = session(payload.to(device), subpayload.to(device))
        loss = celoss(output, label.to(device))

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

    weibull_models = openmax(actVecs)

    trainVAE(trainDataset, testDataset, session, weibull_models, device)


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

    train(
        trainDataset,
        testDataset,
        device,
    )

    # 记录开始时间
    et = time.time()

    # 输出总耗时
    print("\nTotal Time:", et - st)
