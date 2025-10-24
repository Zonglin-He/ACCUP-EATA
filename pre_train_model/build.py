import torch
from pre_train_model.pre_train_model import PreTrainModel
from models.loss import CrossEntropyLabelSmooth
from copy import deepcopy
from sklearn.metrics import accuracy_score, f1_score

def get_algorithm_class(algorithm_name): #返回指定名称的算法类
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

def pre_train_model(backbone, configs, hparams, src_dataloader, avg_meter, logger, device): #预训练模型函数
    # model.
    pre_trained_model = PreTrainModel(backbone, configs, hparams) #初始化预训练模型
    pre_trained_model = pre_trained_model.to(device) #将模型移动到指定设备（CPU或GPU）
    # optimizer Adam
    pre_optimizer = torch.optim.Adam( 
        pre_trained_model.network.parameters(),
        lr=hparams['pre_learning_rate'],
        weight_decay=hparams['weight_decay']
    )
    # loss.
    cross_entropy = CrossEntropyLabelSmooth(configs.num_classes, device, epsilon=0.1)

    # 预训练模型
    for epoch in range(1, hparams['num_epochs'] + 1): #遍历每个训练周期
        pred_list = [] #用于存储预测结果
        label_list = [] #用于存储真实标签
        for step, (src_x, src_y, _) in enumerate(src_dataloader): # 遍历源域dataloader
            # input src data
            if isinstance(src_x, list): #如果输入是增强数据列表
                src_x, src_y = src_x[0].float().to(device), src_y.long().to(device)  # list: (raw_data, aug1, aug2)
            else:
                src_x, src_y = src_x.float().to(device), src_y.long().to(device)  # raw_data

              
            src_pred = pre_trained_model(src_x)# 提取特征并进行分类，得到各类Logits 
            src_cls_loss = cross_entropy(src_pred, src_y)# 计算分类损失
            pre_optimizer.zero_grad()# 优化器梯度清零
            src_cls_loss.backward()# 反向传播
            pre_optimizer.step()# 更新参数

            # acculate loss
            avg_meter['Src_cls_loss'].update(src_cls_loss.item(), 32) #更新平均损失
            #sum += val * n; count += n; avg = sum / count update(val, n) 在这里n=32

            # append prediction
            pred_list.extend(src_pred.argmax(dim=1).detach().cpu().numpy())
            label_list.extend(src_y.detach().cpu().numpy())

        acc = accuracy_score(label_list, pred_list) #计算准确率
        f1 = f1_score(label_list, pred_list, average='macro') #计算宏观F1分数
        print('source acc:', acc, 'source f1:', f1) #打印准确率和F1分数

        logger.debug(f'[Epoch : {epoch}/{hparams["num_epochs"]}]') #日志记录
        for key, val in avg_meter.items():
            logger.debug(f'{key}\t: {val.avg:2.4f}')
        logger.debug(f'-------------------------------------')

    src_only_model = deepcopy(pre_trained_model.network.state_dict()) # 深拷贝网络权重

    return src_only_model, pre_trained_model


