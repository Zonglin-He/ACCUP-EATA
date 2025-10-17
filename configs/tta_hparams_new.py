import math

def get_hparams_class(dataset_name):  # 根据给定数据集名称字符串返回对应的数据集配置类
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5,
            'steps': 1,
            'optim_method': 'adam',
            'momentum': 0.9
        }
        self.alg_hparams = {
            'ACCUP': {
                'pre_learning_rate': 0.001,
                'learning_rate': 3e-4,
                'filter_K': 10,
                'tau': 20,
                'temperature': 0.7,

                # EATA 相关
                'use_eata_select': True,
                'use_eata_reg': True,
                # e_margin = ln(num_classes) * e_margin_scale
                'e_margin_scale': 0.55,   # 建议先试 0.45（更严格）或 0.35（更宽松）
                'd_margin': 0.04,         # 密度阈值，0.03~0.08 之间试两档
                'lambda_eata': 1.0,# 正则强度，0.5~1.5 之间可扫两档
                'e_margin_scale': 0.55,
                'd_margin': 0.04,
                'memory_size': 4096,

            },
            'NoAdap': {'pre_learning_rate': 0.001}
        }

class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 96,        # ↑ 从 64 提到 96：更稳定的 BN/统计，CPU 也扛得住
            'weight_decay': 5e-5,
            'step_size': 20,         # ↓ 让预训练在第20轮衰减一次学习率
            'lr_decay': 0.5,
            'steps': 1,
            'optim_method': 'adam',
            'momentum': 0.9,
            'grad_clip': 0.5,        # ↑ 统一为 0.5，避免双处设置相互打架
            'grad_clip_value': None
        }
        self.alg_hparams = {
            'ACCUP': {
                'pre_learning_rate': 5e-4,   # ↓ 预训练更稳，避免过拟合导致适配不稳
                'learning_rate': 3e-4,       # ↓ TTA 时更小 LR，减少负迁移
                'filter_K': 25,              # ↑ 密度估计更可靠（原 15）
                'tau': 10,
                'temperature': 0.70,         # ↓ 略增置信度分离度，利于选样
                'warmup_min': 48,            # ↑ 热身更长，先累计安全样本再适配
                'quantile': 0.75,            # ↑ 选择更“靠前”的高置信度样本
                'safety_keep_frac': 0.30,    # ↓ 更果断剔除不确定样本

                # EATA
                'use_eata_select': True,
                'use_eata_reg': True,
                'e_margin_scale': 0.45,      # ↑ 从 0.32 → 0.40：更严格的熵边界
                'd_margin': 0.06,            # ↑ 从 0.05 → 0.06：提高密度阈值
                'lambda_eata': 1.4,          # ↓ 从 1.4 → 1.0：避免过强正则压制更新
                'memory_size': 2048,         # ↑ 记忆更长一点，统计更稳
                'use_quantile': True,

                'grad_clip': 0.5,
                'grad_clip_value': None
            },
            'NoAdap': {'pre_learning_rate': 5e-4}
        }


class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 15,  # 30 -> 16（避免源域过拟合，利于 TTA）
            'batch_size': 16,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5,
            'steps': 1,
            'optim_method': 'adam',
            'momentum': 0.9,
            'grad_clip': 0.1,
            'grad_clip_value': None
        }
        # 关键：加入 EATA 的开关和超参
        self.alg_hparams = {
            'ACCUP': {
                'pre_learning_rate': 5e-4,
                'learning_rate': 3e-5,  # TTA 基础 lr；如果代码支持分组，BN 用 5e-5
                'filter_K': 10,  # 7 -> 9，密度估计更稳
                'tau': 20,
                'temperature': 2.0,  # 0.6 -> 0.55，略锐化

                # EATA
                'use_eata_select': True,
                'use_eata_reg': True,
                'e_margin_scale': 0.70,  # 0.35 -> 0.30，放宽
                'd_margin': 0.05,
                'lambda_eata': 1.0,
                'warmup_min': 24,
                'quantile': 0.90,
                'safety_keep_frac': 0.65,  # 0.4 -> 0.5，保底更多
                'memory_size': 4096,
                'use_quantile': True,

                'grad_clip': 1.0,
                'grad_clip_value': 0.5
            },
            'NoAdap': {'pre_learning_rate': 1e-3}
        }

