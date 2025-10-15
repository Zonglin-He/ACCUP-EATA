import torch
import torch.nn.functional as F
import torch.nn as nn

class ConditionalEntropyLoss(torch.nn.Module): # 算香农熵（对每个样本的预测分布）
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)

class CrossEntropyLabelSmooth(nn.Module): #带标签平滑的交叉熵损失
    def __init__(self, num_classes, device, epsilon=0.1): #num_classes:类别数，epsilon:平滑参数
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.device = device
    def forward(self, inputs, targets): #inputs:模型输出的logits, targets:真实标签
        #标签平滑：将one-hot标签转换为平滑标签
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).to(self.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()

        return loss