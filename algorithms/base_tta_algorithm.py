import torch
import torch.nn as nn
from copy import deepcopy

class BaseTestTimeAlgorithm(torch.nn.Module): #是一个抽象接口，各个算法会继承该类并具体实现
    
    def __init__(self, configs, hparams, model, optimizer): #初始化方法 传入配置参数、超参数、模型和优化器
        super(BaseTestTimeAlgorithm, self).__init__() #调用父类的初始化方法
        self.configs = configs #保存配置参数
        self.hparams = hparams #保存超参数
        self.model = self.configure_model(model) #配置模型
        params, param_names = self.collect_params(self.model) #收集模型参数
        if len(param_names) == 0: #如果没有可训练参数
            self.optimizer = None #优化器设为None
        else: 
            self.optimizer = optimizer(params) #否则使用传入的优化器

        self.steps = self.hparams['steps'] #获取超参数steps
        assert self.steps > 0, "requires >= 1 step(s) to forward and update" #确保steps大于0

    def collect_params(self, model: nn.Module): #收集模型中所有需要梯度更新的参数
        names = [] #参数名称列表
        params = [] #参数列表

        for n, p in model.named_parameters(): #遍历模型的所有参数
            if p.requires_grad: #如果参数需要梯度更新
                names.append(n) #添加参数名称到列表
                params.append(p) #添加参数到列表

        return params, names #返回参数列表和名称列表

    def configure_model(self, model):
        raise NotImplementedError # 占位：具体模型配置由子类实现（如 ACCUP 实现解冻部分参数、设置BN等）


    def forward_and_adapt(self, *args, **kwargs):
        raise NotImplementedError # 占位：具体前向传播和适应方法由子类实现

    def forward(self, x, trg_idx = None): # 定义模型前向传播接口
        for _ in range(self.steps): #根据steps决定前向传播和适应的次数
            if trg_idx != None:
                outputs = self.forward_and_adapt(x, self.model, self.optimizer, trg_idx)
            else:
                outputs = self.forward_and_adapt(x, self.model, self.optimizer)
            # 如果提供了 trg_idx 参数，则将其传给 forward_and_adapt
        return outputs #返回模型输出

    @staticmethod 
    def build_ema(model): # 构建并返回模型的 EMA 副本：
        ema_model = deepcopy(model) # 深拷贝模型
        for param in ema_model.parameters(): # 遍历 EMA 模型的所有参数
            param.detach_() # 将 ema_model 所有参数 detach_()，使其不参与梯度计算 
            #在反向传播时不会被 autograd 跟踪、不会累积 grad、不会被优化器更新。只会通过你写的 EMA 公式去更新
        return ema_model # 返回 EMA 模型
 
@torch.jit.script
def softmax_entropy(x, x_ema): #逐样本交叉熵
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

def symm_softmax_entropy(x, x_ema): #对称交叉熵
    alpha = 0.3
    return -(1-alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) - alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)

@torch.jit.script 
def im_softmax_entropy(x, x_ema): #批次的“边际分布”来算交叉熵
    return - (x_ema.softmax(1).mean(0) * torch.log(x.softmax(1).mean(0)) + 1e-5).sum()