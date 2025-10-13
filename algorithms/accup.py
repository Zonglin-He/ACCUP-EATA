import torch
import torch.nn as nn
from .base_tta_algorithm import BaseTestTimeAlgorithm, softmax_entropy
import torch.nn.functional as F
from loss.sup_contrast_loss import domain_contrastive_loss

class ACCUP(BaseTestTimeAlgorithm): #定义ACCUP类，继承自BaseTestTimeAlgorithm

    def __init__(self, configs, hparams, model, optimizer): #初始化方法 传入配置参数、超参数、模型和优化器
        super(ACCUP, self).__init__(configs, hparams, model, optimizer) #调用父类的初始化方法
        self.featurizer = model.feature_extractor #提取模型的特征提取器部分 （CNN）
        self.classifier = model.classifier #提取模型的分类器部分 （全连接层）
        self.filter_K = hparams['filter_K'] #获取超参数filter_K
        self.tau = hparams['tau'] #获取超参数tau
        self.temperature = hparams['temperature'] #获取超参数temperature
        self.num_classes =  configs.num_classes #获取类别数
        warmup_supports = self.classifier.logits.weight.data.detach() #获取分类器的权重作为初始支持样本
        self.warmup_supports = warmup_supports #保存初始支持样本
        warmup_prob = self.classifier(self.warmup_supports) #通过分类器计算初始支持样本的输出概率（预训练模型）
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes = self.num_classes).float() #获取初始支持样本的伪标签
        self.warmup_ent = softmax_entropy(warmup_prob, warmup_prob) #计算交叉熵
        self.warmup_cls_scores = F.softmax(warmup_prob, 1) #计算初始支持样本的分类器输出的softmax概率

        self.supports = self.warmup_supports.data #储存初始化warmup样本
        self.labels = self.warmup_labels.data #储存初始化warmup伪标签
        self.ents = self.warmup_ent.data #储存初始化warmup交叉熵
        self.cls_scores = self.warmup_cls_scores .data #储存初始化warmup分类器输出的softmax概率

        """利用预训练源模型分类器的权重向量初始化每个类别的原型。
        代码中通过 self.classifier.logits.weight 获取线性分类器各类别的权重（大小为[num_classes, 特征维度]），
        这些权重向量视为每个类别初始的支持特征（原型）。然后将这些支持特征输入原分类器，得到预测 warmup_prob，
        取其最大概率类别作为该支持的初始伪标签（one-hot形式），并计算其预测的熵 warmup_ent 作为不确定性度量。
        熵值越低表示预测越自信。初始每个类别原型的概率分布也计算得到 warmup_cls_scores。"""

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        raw_data, aug_data = batch_data[0], batch_data[1] #获取原始数据和增强数据，在dataloder中已经做了增强
        r_feat, r_seq_feat = model.feature_extractor(raw_data) #通过特征提取器提取原始数据的特征
        r_output = model.classifier(r_feat) #通过分类器得到原始数据的输出
        a_feat, a_seq_feat = model.feature_extractor(aug_data) #通过特征提取器提取增强数据的特征
        a_output = model.classifier(a_feat) #通过分类器得到增强数据的输出
        z = (r_feat + a_feat) / 2.0 # 计算原始和增强特征的平均值 z（增强集成特征）
        p = (r_output + a_output) / 2.0 # 计算原始和增强输出的平均 logits p（增强集成预测）
        yhat = F.one_hot(p.argmax(1), num_classes=self.num_classes).float() # 得到平均预测 p 的伪标签 yhat（取p中最大概率的类别，转为one-hot向量）
        ent = softmax_entropy(p, p) #计算最终输出的熵
        cls_scores = F.softmax(p, 1) #计算最终输出的softmax概率

        with torch.no_grad(): # 使用 no_grad 以免梯度回传影响支持集存储
            self.supports = self.supports.to(z.device) 
            self.labels = self.labels.to(z.device) 
            self.ents = self.ents.to(z.device)
            # 确保当前支持集的数据转移到当前计算设备
            self.cls_scores = self.cls_scores.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ents = torch.cat([self.ents, ent])
            self.cls_scores = torch.cat([self.cls_scores, cls_scores])
            # 将当前批次的增强特征z、伪标签yhat、熵ent和概率cls_scores 添加到记忆支持集中
            # 这样记忆集就有了过去样本的信息

        supports, labels, indices = self.select_supports() # 从更新后的支持集中选择每个类别熵值最低的K个支持（最自信的）
        loss = 0.
        prt_scores = self.compute_logits(z, supports, labels)  # 计算当前批次样本与各类别原型的相似度 logits
        prt_ent = softmax_entropy(prt_scores, prt_scores) 
        idx = prt_ent < ent # 选择与原型相似度更高（熵更低）的样本
        idx_un = idx.unsqueeze(1).expand(-1, prt_scores.shape[1]) # 扩展索引以匹配 logits 形状
        select_pred = torch.where(idx_un, prt_scores, cls_scores) 
        # 根据熵比较结果选择预测：若某样本原型预测熵低则采用prt_scores，否则采用原模型预测cls_scores
        pseudo_labels = select_pred.max(1, keepdim=False)[1] # 确定每个样本最终的伪标签 pseudo_labels（选定预测的最大概率类别索引）

        cat_p = torch.cat([r_output, a_output, p], dim=0) # 将原始、增强和平均的输出 logits 拼接（形成包含三种视图的集合C = X ∪ A ∪ 平均)
        cat_pseudo_labels = pseudo_labels.repeat(3) # 将每个样本的最终伪标签重复3次，对应拼接后的原始、增强、平均三个视图
        loss += domain_contrastive_loss(cat_p, cat_pseudo_labels, temperature=self.temperature, device=z.device)

    # 计算领域对比损失，促使同类样本在特征空间更接近，不同类样本更远离
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 基于计算的对比损失，反向传播并更新模型参数

        return  select_pred # 返回当前批次样本的最终预测结果

    def get_topk_neighbor(self, feature, supports, cls_scores, k_neighbor): # 计算每个输入样本在支持集中余弦相似度最高的k_neighbor个邻居的预测分布
        feature = F.normalize(feature, dim=1) #对特征归一化
        supports = F.normalize(supports, dim=1) #对支持集归一化（增强集成特征，伪标签，熵，概率）
        sim_matrix = feature @ supports.T #计算输入样本与支持集之间的余弦相似度矩阵
        _, idx_near = torch.topk(sim_matrix, k_neighbor, dim=1) #获取每个输入样本在支持集中余弦相似度最高的k_neighbor个邻居的索引
        """sim_matrix 形状是 [B, N]：每个输入样本（B 行）对所有支持样本（N 列）的相似度分数。
            torch.topk(t, k, dim=1) 会在“每一行”里取 最大的 k 个数，返回两个张量：(values, indices)。
            代码里只要索引（indices）所以用 _ 忽略了 values：
            _ , idx_near = torch.topk(sim_matrix, k_neighbor, dim=1)
            得到的 idx_near 形状是 [B, k]：对每个输入样本，给出它最相似的 k 个支持样本的列索引。
            随后用这些索引去 cls_scores 里把邻居们的类别分布取出来"""
        cls_score_near = cls_scores[idx_near].detach().clone() #根据邻居索引获取对应的预测分布，并克隆一份

        return cls_score_near

    def compute_logits(self, z, supports, labels): #计算输入样本与支持集之间的相似度 logits
        B, dim = z.size() #获取当前批次样本的数量和特征维度
        N, dim_ = supports.size() #获取支持集的数量和特征维度
        assert (dim == dim_)  #确保特征维度匹配
        temp_centroids = (labels / (labels.sum(dim=0, keepdim=True) + 1e-12)).T @ supports 
        """
        把 labels 每一列除以该类的支持数，相当于给“同类的每个支持样本”权重 1/|类|；再用
        (归一化后的 labels).T @ supports，就得到各类原型（每类支持特征的平均），形状 [C, D]
        labels.sum(dim=0) 计算每个类别有多少支持样本。
        labels / labels.sum(dim=0) 对每个支持样本的标签向量按其所属类别总数进行归一化（相当于给该支持样本一个权重）。
        (labels / sum).T @ supports 相当于对每个类别取其支持样本特征的平均值，得到临时质心矩阵 temp_centroids，每一行是一个类别原型向量。
        这样每个类别的原型就是该类别所有支持样本特征的加权平均，权重是 1/该类支持数，确保每类原型不受支持数多少影响。
        """
        temp_z = F.normalize(z, dim=1)
        temp_centroids = F.normalize(temp_centroids, dim=1) #行归一化为单位向量，为了让后面的点积=余弦相似度
        logits = self.tau * temp_z @ temp_centroids.T #计算每个样本与各类原型的余弦相似度并乘以温度系数 tau（锐化/放大差异）

        return logits ## 得到输出 logits（形状 B×C），表示每个输入样本到各类别原型的匹配分数

    def select_supports(self): # 根据熵值选择support集索引：
        ent_s = self.ents #获取当前支持集的熵值
        y_hat = self.labels.argmax(dim=1).long() #获取当前支持集的伪标签
        filter_K = self.filter_K #获取超参数filter_K 每个类别最多保留的低熵支持样本数
        if filter_K == -1: #如果filter_K为-1，表示不进行筛选，保留所有支持样本
            indices = torch.LongTensor(list(range(len(ent_s))))
        else:
            indices = [] #数组用来储存每个support的索引
            indices1 = torch.LongTensor(list(range(len(ent_s))))
            for i in range(self.num_classes): #提取属于该类别的所有支持样本的熵 
                _, indices2 = torch.sort(ent_s[y_hat == i]) #从中取出前 filter_K 个索引（熵最低的K个）
                indices.append(indices1[y_hat == i][indices2][:filter_K]) #映射回全局索引 indices1，收集入 indices 列表
            indices = torch.cat(indices) #将各类别的索引拼接成一个长索引张量

        self.supports = self.supports[indices] #根据筛选出的索引更新支持集
        self.labels = self.labels[indices] #根据筛选出的索引更新伪标签
        self.ents = self.ents[indices] #根据筛选出的索引更新熵值
        self.cls_scores = self.cls_scores[indices] #根据筛选出的索引更新预测概率

        return self.supports, self.labels, indices #返回筛选后的支持集、伪标签和索引

    def configure_model(self, model): #配置模型在测试时自适应过程中的梯度行为
        model.train() #设置模型为训练模式
        model.requires_grad_(False) #冻结模型所有参数的梯度

        for module in model.modules(): #遍历模型的所有子模块
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d): # 对模型内部的所有BatchNorm层
                module.track_running_stats = False # 停止跟踪运行均值和方差（使其不更新统计量）
                module.running_mean = None
                module.running_var = None   # 清除其运行均值和方差，以仅依赖当前批次统计
                module.weight.requires_grad_(True)  # 清除其运行均值和方差，以仅依赖当前批次统计
                module.bias.requires_grad_(True)

        for name, module in model.feature_extractor.named_children():
            if name == 'conv_block1' or name == 'conv_block2' or name == 'conv_block3':
                for sub_module in module.children():
                    if isinstance(sub_module, nn.Conv1d):
                        sub_module.requires_grad_(True)
        # 这里将三个卷积模块中的Conv1d层设为requires_grad=True
        return model

def update_ema_variables(ema_model, model, alpha_teacher): #让一个“教师模型”始终保持平滑的历史平均权重
    for ema_param, param in zip(ema_model.parameters(), model.parameters()): 
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
        #对每个参数 θ，有 EMA ← α·EMA + (1−α)·θ, 现在普遍用 α=0.99~0.999。α越接近1，更新越慢、越平滑；α小则更跟得上新权重变化
    return ema_model 

def softmax_kl_loss(input_logits, target_logits): # 计算两个 logits 分布之间的 KL 散度
    assert input_logits.size() == target_logits.size() #确保输入和目标 logits 形状匹配
    input_log_softmax = F.log_softmax(input_logits, dim=1) #计算输入 logits 的 log softmax
    target_softmax = F.softmax(target_logits, dim=1) #计算目标 logits 的 softmax

    kl_div = F.kl_div(input_log_softmax, target_softmax, reduction='none') #计算逐元素的 KL 散度
    return kl_div #返回 KL 散度张量

"""update_ema_variables 和 softmax_kl_loss 这两个函数在 ACCUP 类中未被调用，可能是为其他用途预留的辅助函数。"""