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
    # Optuna best overrides for EEG (study: tta_optuna)
    scenario(0, 11): {
        'learning_rate': 1.064530947529722e-05,
        'pre_learning_rate': 0.00014248091411844338,
        'filter_K': 5,
        'tau': 26,
        'temperature': 2.1761793626991457,
        'warmup_min': 107,
        'quantile': 0.8693817812530925,
        'use_quantile': True,
        'safety_keep_frac': 0.09959713627586461,
        'lambda_eata': 0.7172471252245723,
        'e_margin_scale': 0.4114897934784141,
        'd_margin': 0.1464654118232368,
        'memory_size': 768,
        'fisher_alpha': 1611.3281858813957,
        'use_eata_select': True,
        'use_eata_reg': False,
        'online_fisher': False,
        'include_warmup_support': False,
        'max_fisher_updates': 1024,
        'train_full_backbone': False,
        'train_classifier': False,
        'freeze_bn_stats': False,
        'grad_clip': 0.7135659192554825,
        'grad_clip_value': 0.05,
        'num_epochs': 41,
        'batch_size': 80,
        'weight_decay': 7.928811456831559e-05,
        'step_size': 30,
        'lr_decay': 0.5037338946474635,
        'steps': 1,
        'momentum': 0.7904006722067805
    },
    scenario(12, 5): {
        'learning_rate': 1.0000318430740127e-05,
        'pre_learning_rate': 0.00012483102695352874,
        'filter_K': 9,
        'tau': 25,
        'temperature': 2.0824090570211613,
        'warmup_min': 115,
        'quantile': 0.8808501320114017,
        'use_quantile': False,
        'safety_keep_frac': 0.19080553145716495,
        'lambda_eata': 0.7421164696840489,
        'e_margin_scale': 0.4456162543895605,
        'd_margin': 0.050108522682583985,
        'memory_size': 1024,
        'fisher_alpha': 1776.7407957592718,
        'use_eata_select': False,
        'use_eata_reg': True,
        'online_fisher': True,
        'include_warmup_support': True,
        'max_fisher_updates': 32,
        'train_full_backbone': True,
        'train_classifier': True,
        'freeze_bn_stats': True,
        'grad_clip': 0.2587599113991441,
        'grad_clip_value': 0.05,
        'num_epochs': 43,
        'batch_size': 88,
        'weight_decay': 6.540132883420885e-06,
        'step_size': 26,
        'lr_decay': 0.6433162091238267,
        'steps': 1,
        'momentum': 0.8008310363555619
    },
    scenario(7, 18): {
        'learning_rate': 1.8283705675885796e-05,
        'pre_learning_rate': 8.471577582131651e-05,
        'filter_K': 11,
        'tau': 19,
        'temperature': 2.8760928857383083,
        'warmup_min': 169,
        'quantile': 0.7616015084219857,
        'use_quantile': False,
        'safety_keep_frac': 0.32396379042790513,
        'lambda_eata': 0.523332751635693,
        'e_margin_scale': 0.3023729920046602,
        'd_margin': 0.03741254907879476,
        'memory_size': 1280,
        'fisher_alpha': 2224.7405910909306,
        'use_eata_select': False,
        'use_eata_reg': True,
        'online_fisher': True,
        'include_warmup_support': True,
        'max_fisher_updates': 32,
        'train_full_backbone': True,
        'train_classifier': True,
        'freeze_bn_stats': True,
        'grad_clip': 0.34618518109214413,
        'grad_clip_value': 0.05,
        'num_epochs': 48,
        'batch_size': 96,
        'weight_decay': 6.161454303862985e-06,
        'step_size': 25,
        'lr_decay': 0.6866097075200951,
        'steps': 1,
        'momentum': 0.8242475740594322
    },
    scenario(16, 1): {
        'learning_rate': 1.3831057852051844e-05,
        'pre_learning_rate': 9.004343447348951e-05,
        'filter_K': 11,
        'tau': 22,
        'temperature': 2.834073106046921,
        'warmup_min': 142,
        'quantile': 0.7083294529196547,
        'use_quantile': False,
        'safety_keep_frac': 0.37450280132071784,
        'lambda_eata': 0.6465185555231232,
        'e_margin_scale': 0.376553345269517,
        'd_margin': 0.018453418040454875,
        'memory_size': 1792,
        'fisher_alpha': 3976.767190034433,
        'use_eata_select': False,
        'use_eata_reg': True,
        'online_fisher': True,
        'include_warmup_support': True,
        'max_fisher_updates': 32,
        'train_full_backbone': True,
        'train_classifier': True,
        'freeze_bn_stats': True,
        'grad_clip': 0.19102411062144053,
        'grad_clip_value': 0.05,
        'num_epochs': 50,
        'batch_size': 96,
        'weight_decay': 5.461937405940922e-06,
        'step_size': 22,
        'lr_decay': 0.7499042383693464,
        'steps': 1,
        'momentum': 0.8726447565477226
    },
    scenario(9, 14): {
        'learning_rate': 1.568533425515559e-05,
        'pre_learning_rate': 0.00011827610143036819,
        'filter_K': 11,
        'tau': 21,
        'temperature': 2.9052062193008985,
        'warmup_min': 142,
        'quantile': 0.7086218845896661,
        'use_quantile': False,
        'safety_keep_frac': 0.4017360348282116,
        'lambda_eata': 0.6258589716516471,
        'e_margin_scale': 0.3871086996146498,
        'd_margin': 0.014153961912928203,
        'memory_size': 1536,
        'fisher_alpha': 4348.936117832797,
        'use_eata_select': False,
        'use_eata_reg': True,
        'online_fisher': True,
        'include_warmup_support': True,
        'max_fisher_updates': 512,
        'train_full_backbone': True,
        'train_classifier': True,
        'freeze_bn_stats': True,
        'grad_clip': 0.2167093180605843,
        'grad_clip_value': 0.05,
        'num_epochs': 49,
        'batch_size': 96,
        'weight_decay': 5.444880895393821e-06,
        'step_size': 23,
        'lr_decay': 0.7227371671340116,
        'steps': 1,
        'momentum': 0.8405705594776002
    },
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

                'scenario_overrides': dict(EEG_ACCUP_SCENARIO_OVERRIDES),

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

