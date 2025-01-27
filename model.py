import libmr
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import config

MAX_SESS_LEN = 16
MAX_PKT_LEN = 128
LATENT_DIM = 33
LabelNum = config.LabelNum


class OpenMax(nn.Module):
    def __init__(self):
        self.weibull_models = {}
        self.tailalpha = 0.05
        self.tailFitNumMin = 10
        self.attenuationRate = config.Attenuation

    def fitWeibull(self, act_vecs):
        for i in range(LabelNum):
            self.weibull_models[i] = {}

            act_vec = act_vecs[i]
            act_vec = np.array(act_vec)
            mean_vec = np.mean(act_vec, axis=0)

            dist = []
            for vec in act_vec:
                dis = np.sqrt(np.sum(np.square(vec - mean_vec)))
                dist.append(dis)

            tailsize = int(len(dist) * self.tailalpha)
            tailsize = max(self.tailFitNumMin, tailsize)
            tailtofit = sorted(dist)[-tailsize:]

            mr = libmr.MR()
            mr.fit_high(tailtofit, len(tailtofit))

            self.weibull_models[i]["mean_vec"] = mean_vec
            self.weibull_models[i]["weibull_model"] = mr

    def recalibrate(self, act_vecs, preds):
        dataNum = act_vecs.shape[0]
        alpha_weights = [(LabelNum - i - 2) / float(LabelNum) for i in range(LabelNum)]
        attenuation = [self.attenuationRate for i in range(LabelNum)]
        openmax_scores = []

        for i in range(dataNum):
            act_vec = act_vecs[i]
            pred = preds[i]
            ranked_list = pred.argsort().ravel()[::-1]
            ranked_alpha = [0 for j in range(LabelNum)]
            for j in range(len(alpha_weights)):
                ranked_alpha[ranked_list[j]] = alpha_weights[j]

            openmax_score = []
            openmax_unknown = 0
            for j in range(LabelNum):
                mean_vec = self.weibull_models[j]["mean_vec"]
                mr = self.weibull_models[j]["weibull_model"]
                dis = np.sqrt(np.sum(np.square(act_vec[j] - mean_vec)))
                wscore = mr.w_score(dis)
                modified = pred[j] * (1 - wscore * ranked_alpha[j] * attenuation[j])
                openmax_score.append(modified)
                openmax_unknown += pred[j] - modified
            openmax_score.append(openmax_unknown)

            openmax_scores.append(openmax_score)

        return openmax_scores


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(inplace=True),
        )
        torch.nn.init.normal_(self.conv[0].weight, mean=0.0, std=0.02)

    def forward(self, x):
        x = x.unsqueeze(1)
        return self.conv(x).permute(0, 2, 1)


class PayloadEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = ConvBlock(1, 1, 4, 2)
        self.cnn2 = ConvBlock(1, 1, 3, 3)
        self.clf = nn.Linear(16 * 33, LabelNum)

    def forward(self, payload, subpayload):
        fusionFeature = self.getFeature(payload, subpayload)

        activeVector = self.clf(fusionFeature)
        output = F.softmax(activeVector, dim=1)

        return activeVector, output

    def getFeature(self, payload, subpayload):
        batch_size = payload.shape[0]

        pld = torch.reshape(payload, (-1, 128))
        pld = self.cnn1.forward(pld)
        pld = pld.squeeze()

        pld = self.cnn2.forward(pld)
        pld = pld.squeeze()
        pld = torch.reshape(pld, (batch_size, 16, -1))

        fusionFeature = torch.cat([pld, subpayload], dim=2)
        fusionFeature = torch.reshape(fusionFeature, (batch_size, -1))

        return fusionFeature


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.threshold = 0
        self.l_z_x = nn.Sequential(
            nn.Linear(16 * 33, 8 * 33),
            nn.Sigmoid(),
            nn.Linear(8 * 33, 4 * 33),
            nn.Sigmoid(),
            nn.Linear(4 * 33, 2 * LATENT_DIM),
        )
        self.l_x_z = nn.Sequential(
            nn.Linear(LATENT_DIM, 2 * 33),
            nn.Sigmoid(),
            nn.Linear(2 * 33, 4 * 33),
            nn.Sigmoid(),
            nn.Linear(4 * 33, 8 * 33),
            nn.Sigmoid(),
            nn.Linear(8 * 33, 16 * 33),
        )

    def z_x(self, x):
        mu, logsigma = self.l_z_x(x).chunk(2, dim=-1)
        return Normal(mu, logsigma.exp() + 0.001)

    def x_z(self, z):
        return self.l_x_z(z)
