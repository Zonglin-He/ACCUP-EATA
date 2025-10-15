import torch
import torch.nn.functional as F

def domain_contrastive_loss(domains_features, domains_labels, temperature,device):
    anchor_feature = domains_features
    anchor_feature = F.normalize(anchor_feature, dim=1)
    labels = domains_labels
    labels= labels.contiguous().view(-1, 1)
     # 规范化特征向量 & 整理标签 shape：(N,

    mask = torch.eq(labels, labels.T).float().to(device) # shape：(N,N)，相同标签为1，不同为0
    anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, anchor_feature.T), temperature)
    # 计算所有样本两两之间的点积相似度（归一化后即cos相似度）除以温度

    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # shape：(N,1)，每行最大值
    logits = anchor_dot_contrast - logits_max.detach() # 防止数值不稳定, 减去每行最大值

    logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(anchor_feature.shape[0]).view(-1, 1).to(device), 0)
    #创建 N×N 矩阵，若第i和j个样本标签相同则mask[i,j]=1，否则=0。这识别出所有正样本对，包括i=j（自身与自身标签当然相同，稍后会去除）
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask # 计算对比概率
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mask_sum = mask.sum(1) # 每个样本的正样本数量
    zeros_idx = torch.where(mask_sum == 0)[0] # 找到没有正样本的样本
    mask_sum[zeros_idx] = 1 # 防止除0错误
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum # 每个样本的平均正样本对比概率

    loss = (- 1 * mean_log_prob_pos)
    loss = loss.mean() # 对所有样本求平均

    return loss

"""
domain_contrastive_loss 实现了监督式对比损失（Sup-Contrastive Loss）的计算，
用于鼓励具有相同伪标签的特征靠近、不同伪标签的特征分开
ACCUP 在 forward_and_adapt 中构建 cat_p 及对应 cat_pseudo_labels 传入此函数。
cat_p 包含一个批次每个样本的原始预测、增强预测、以及平均预测（共3*N条向量）。c
at_pseudo_labels 则让这些视图共享同一个伪标签。这样在 domain_contrastive_loss 内，mask会标识

原始样本、本样本增强、本样本平均 三者互为正样本对（因为它们伪标签相同）。
同一批内不同样本如果伪标签恰好相同，也算正样本对
（这意味着该对比损失不仅把同一样本的多视图拉近，也会拉近不同样本但预测为同类的特征，促成类内聚集）。
不同伪标签的样本都是负样本，相似度将被压低。

通过此对比损失，ACCUP实现了增强对比聚类模块
最大化正样本对（一方面跨视图同一样本视图之间；另一方面同类不同样本之间）的相似度，最小化与负样本（不同类样本）的相似度。
该方法相比传统交叉熵对错误伪标签的鲁棒性更高,
即使某些伪标签错误，只要多数同类是正确的，它们形成的聚类仍能引导模型学习到正确的特征划分，而少数噪声标签影响会被稀释。

温度参数 temperature 对应公式中的 τ。较小的temperature会让loss更注重区分难样本。
ACCUP在不同数据集对 temperature 有不同设置（如EEG 0.3, HAR 0.7, FD 0.6），在超参数配置里体现。
这会在domain_contrastive_loss中体现出来，调节softmax的陡峭程度
"""