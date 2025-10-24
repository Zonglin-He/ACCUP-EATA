import math


def scenario(src_id, trg_id):
    """Normalize场景ID为字符串元组，避免'几to几'写错或类型不一致。"""
    return str(src_id), str(trg_id)


HAR_ACCUP_SCENARIO_OVERRIDES = {
    scenario(6, 23): {
        'learning_rate': 2.2753505223126986e-05,
        'pre_learning_rate': 0.0003338822801641277,
        'filter_K': 23,
        'tau': 30,
        'temperature': 1.1745975040622931,
        'warmup_min': 77,
        'quantile': 0.9408771415733412,
        'safety_keep_frac': 0.215975500824709,
        'lambda_eata': 2.3520584592343377,
        'e_margin_scale': 0.5529453366693213,
        'd_margin': 0.08753985821873647,
        'memory_size': 1792,
        'fisher_alpha': 5271.57289066215
    },
    scenario(7, 13): {
        'learning_rate': 2.4202173680587405e-05,
        'pre_learning_rate': 0.0004905457868560195,
        'filter_K': 21,
        'tau': 30,
        'temperature': 1.77551406001387,
        'warmup_min': 103,
        'quantile': 0.9350629148701761,
        'safety_keep_frac': 0.1590404711695088,
        'lambda_eata': 1.7527194782975355,
        'e_margin_scale': 0.8869776773057166,
        'd_margin': 0.02059152893472859,
        'memory_size': 2560,
        'fisher_alpha': 1576.4941902477176

    },
    scenario(9, 18): {
           'learning_rate': 1.1511585680068082e-05,
            'pre_learning_rate': 0.00017903636106246677,
            'filter_K': 23,
            'tau': 30,
            'temperature': 1.3680608746243965,
            'warmup_min': 83,
            'quantile': 0.9417889258036682,
            'safety_keep_frac': 0.29519346015191145,
            'lambda_eata': 2.3634089504869173,
            'e_margin_scale': 0.4999073720347249,
            'd_margin': 0.03852076679814576,
            'memory_size': 1792,
            'fisher_alpha': 7052.155092372549

    },
    scenario(12, 16): {
        'learning_rate': 2.0740580122695722e-05,
        'pre_learning_rate': 0.00040512511160526344,
        'filter_K': 21,
        'tau': 30,
        'temperature': 2.5764790019089028,
        'warmup_min': 214,
        'quantile': 0.9499134672993275,
        'safety_keep_frac': 0.21602750119636374,
        'lambda_eata': 0.9397129269263443,
        'e_margin_scale': 0.4735038935494198,
        'd_margin': 0.07395967204894886,
        'memory_size': 2560,
        'fisher_alpha': 5866.516763771302
    },
}

EEG_ACCUP_SCENARIO_OVERRIDES = {
    # 调参完成后把结果写到这里，例如：
    # scenario(0, 11): {'learning_rate': ...},
}

def get_hparams_class(dataset_name):  # 根据给定数据集名称字符串返回对应的数据集配置类
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class FD():
    def __init__(self):
        super(FD, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 128,
            'weight_decay': 1e-4,
            'step_size': 30,
            'lr_decay': 0.5,
            'steps': 1,
            'optim_method': 'adam',
            'momentum': 0.9,
            'grad_clip': 0.5,
            'grad_clip_value': None
        }
        self.alg_hparams = {
            'ACCUP': {
                'pre_learning_rate': 3e-4,
                'learning_rate': 1e-4,
                'filter_K': 21,
                'tau': 12,
                'temperature': 0.60,
                'warmup_min': 128,
                'quantile': 0.85,
                'safety_keep_frac': 0.08,

                # EATA
                'use_eata_select': True,
                'use_eata_reg': True,
                'e_margin_scale': 0.40,
                'd_margin': 0.02,
                'lambda_eata': 1.4,
                'memory_size': 4096,
                'use_quantile': True,
                'fisher_alpha': 2000.0,
                'online_fisher': True,
                'include_warmup_support': True,
                'max_fisher_updates': -1,

                'train_full_backbone': True,   # allow full backbone (e.g. TimesNet) to receive gradients
                'train_classifier': True,
                'freeze_bn_stats': False,

                'scenario_overrides': dict(EEG_ACCUP_SCENARIO_OVERRIDES),

                'grad_clip': 0.5,
                'grad_clip_value': None
            },
            'NoAdap': {'pre_learning_rate': 5e-4}
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
                'pre_learning_rate': 5e-4,
                'learning_rate': 3e-4,
                'filter_K': 25,
                'tau': 10,
                'temperature': 0.70,
                'warmup_min': 48,
                'quantile': 0.75,
                'safety_keep_frac': 0.30,

                # EATA
                'use_eata_select': True,
                'use_eata_reg': True,
                'e_margin_scale': 0.45,
                'd_margin': 0.06,
                'lambda_eata': 1.4,
                'memory_size': 2048,
                'use_quantile': True,
                'fisher_alpha': 2000.0,
                'online_fisher': True,
                'include_warmup_support': True,
                'max_fisher_updates': -1,

                'train_full_backbone': True,   # allow full backbone (e.g. TimesNet) to receive gradients
                'train_classifier': True,
                'freeze_bn_stats': False,

                'scenario_overrides': {
                    # ('src_id', 'trg_id'): {'learning_rate': 5e-4, 'tau': 12},
                },

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
                'fisher_alpha': 2000.0,
                'online_fisher': True,
                'include_warmup_support': True,
                'max_fisher_updates': -1,

                'train_full_backbone': True,   # allow full backbone (e.g. TimesNet) to receive gradients
                'train_classifier': True,
                'freeze_bn_stats': False,

                'scenario_overrides': dict(HAR_ACCUP_SCENARIO_OVERRIDES),

                'grad_clip': 1.0,
                'grad_clip_value': 0.5
            },
            'NoAdap': {'pre_learning_rate': 1e-3}
        }

