# 数据集名称
Dataset = "CICIDS2018"
# 数据集路径
StatisticsPath = "../../Datas/" + Dataset + "/sample data/statistics/"
PayloadPath = "../../Datas/" + Dataset + "/sample data/payload/"
SubPayloadPath = "../../Datas/" + Dataset + "/sample data/subpayload/"
# 模型保存文件夹
ModelPath = "./modelDict/" + Dataset + "/"

# 模型保存路径
PLDFE_PATH = ModelPath + "PLDFE.pkl"
OPENMAX_PATH = ModelPath + "OPENMAX.pkl"
VAE_PATH = ModelPath + "VAE/"

Seed = 369

MAX_SESS_LEN = 16


if Dataset == "CICIDS2017":
    VAETrainEpoch = 20

    BenignAlpha = 0.97
    AttackAlpha = 1
    Attenuation = 0.73

    BenignSet = ["Benign"]
    KnownAttackSet = [
        "FTP Brute Force",
        "DoS Hulk",
        "DoS Slowloris",
        "Port Scan",
        "DDoS LOIT",
    ]
    UnknownAttackSet = [
        "SSH Brute Force",
        "DoS Slowhttptest",
        "DoS GoldenEye",
        "Botnet",
        "Web Attack Brute Force",
        "Web Attack XSS",
        "Web Attack Sql Injection",
    ]
elif Dataset == "CICIDS2018":
    VAETrainEpoch = 50

    BenignAlpha = 0.96
    AttackAlpha = 0.96
    Attenuation = 0.82

    BenignSet = ["Benign"]
    KnownAttackSet = [
        "DDoS HOIC",
        "DoS Hulk",
        "DoS Slowloris",
        "SSH Brute Force",
    ]
    UnknownAttackSet = [
        "Botnet",
        "DDoS LOIC",
        "DoS GoldenEye",
        "Web Attack Brute Force",
        "Web Attack Sql Injection",
        "Web Attack XSS",
    ]

LabelList = ["Benign"]
LabelList.extend(KnownAttackSet)
LabelList.append("Unknown")
LabelNum = len(LabelList) - 1
