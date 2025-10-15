import torch
import torch.nn as nn
from models.da_models import classifier

class PreTrainModel(nn.Module): #预训练模型类

    def __init__(self, backbone, configs, hparams): #初始化函数，接受骨干网络、配置和超参数
        super(PreTrainModel, self).__init__()
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)
        self.configs = configs
        self.hparams = hparams

    def forward(self, x): #前向传播函数，接受输入数据x
        feat, _ = self.feature_extractor(x) #编码器，提取特征
        out = self.classifier(feat) #分类头
        return out
