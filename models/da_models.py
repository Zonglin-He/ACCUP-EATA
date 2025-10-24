import torch
from torch import nn

from models.timesnet_backbone import TimesNet

def get_backbone_class(backbone_name): #根据给定的名称返回对应的类
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]

#特征提取器
class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.aap = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x_flat = self.aap(x).view(x.shape[0], -1) #展平

        return x_flat, x

# 分类头
class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)
        #输入维度 = feature_extractor输出向量长度 (final_out_channels * features_len)，输出维度 = 类别数

    def forward(self, x): #前向传播，对输入特征 x 进行线性变换得到每类的 logits
        predictions = self.logits(x)
        return predictions
