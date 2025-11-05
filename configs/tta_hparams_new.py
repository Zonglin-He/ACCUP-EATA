import math


def scenario(src_id, trg_id):
    """Normalize场景ID为字符串元组，避免'几to几'写错或类型不一致。"""
    return str(src_id), str(trg_id)


HAR_ACCUP_SCENARIO_OVERRIDES = {
    scenario(6, 23): {
        'batch_size': 16,
        'd_margin': 0.003388034261102267,
        'e_margin_scale': 0.41548216566981844,
        'filter_K_bn_only': 9,
        'fisher_alpha': 627.2931431022805,
        'freeze_bn_stats_bn_only': True,
        'grad_clip_bn_only': 0.2501756037496996,
        'grad_clip_value_bn_only': 0.05,
        'include_warmup_support': True,
        'lambda_eata': 1.0476214136193907,
        'learning_rate': 7.042099599242353e-05,
        'lr_decay': 0.6825465967385823,
        'max_fisher_updates': 256,
        'memory_size': 3584,
        'momentum': 0.9413699698385589,
        'num_epochs': 14,
        'online_fisher': False,
        'pre_learning_rate': 0.0002693113248886997,
        'quantile': 0.6020845912948787,
        'safety_keep_frac_bn_only': 0.21838279792869253,
        'step_size': 69,
        'steps': 4,
        'tau': 12,
        'temperature': 1.9794413584494728,
        'train_scope': 'bn_only',
        'use_eata_reg': True,
        'use_eata_select': False,
        'use_quantile': False,
        'warmup_min': 108,
        'weight_decay': 0.0006228807802651315,
    },

    scenario(7, 13): {
        'batch_size': 24,
        'd_margin': 0.1360412845577631,
        'e_margin_scale': 0.8250038419880311,
        'filter_K_partial': 7,
        'fisher_alpha': 940.6944599290841,
        'freeze_bn_stats_partial': True,
        'grad_clip_partial': 0.7252972981863884,
        'grad_clip_value_partial': 0.05,
        'include_warmup_support': True,
        'lambda_eata': 1.7882825949366934,
        'learning_rate': 0.0004752251087670758,
        'lr_decay': 0.6861728425632769,
        'max_fisher_updates': 64,
        'memory_size': 2816,
        'momentum': 0.8741442369922959,
        'num_epochs': 8,
        'online_fisher': True,
        'partial_module_bundle': 'conv1',
        'pre_learning_rate': 0.000447190594237787,
        'quantile': 0.1861722873027812,
        'safety_keep_frac_partial': 0.6151097609602402,
        'step_size': 55,
        'steps': 5,
        'tau': 13,
        'temperature': 2.7534330479684455,
        'train_classifier_partial': False,
        'train_scope': 'partial',
        'use_eata_reg': True,
        'use_eata_select': False,
        'use_quantile': True,
        'warmup_min': 297,
        'weight_decay': 2.2391145838853417e-05,
    },

    scenario(9, 18): {
        'batch_size': 16,
        'd_margin': 0.12660590366332933,
        'e_margin_scale': 0.28199538299296195,
        'filter_K_bn_only': 9,
        'fisher_alpha': 3803.8384653937164,
        'freeze_bn_stats_bn_only': False,
        'grad_clip_bn_only': 0.3105534089615374,
        'grad_clip_value_bn_only': None,
        'include_warmup_support': False,
        'lambda_eata': 0.895575083854534,
        'learning_rate': 3.977455280911086e-05,
        'lr_decay': 0.6703761497310525,
        'max_fisher_updates': 128,
        'memory_size': 512,
        'momentum': 0.7393110429711749,
        'num_epochs': 16,
        'online_fisher': False,
        'pre_learning_rate': 0.00011413582761563282,
        'quantile': 0.730888128823268,
        'safety_keep_frac_bn_only': 0.3743831771864864,
        'step_size': 38,
        'steps': 2,
        'tau': 8,
        'temperature': 2.3423530546732216,
        'train_scope': 'bn_only',
        'use_eata_reg': True,
        'use_eata_select': False,
        'use_quantile': True,
        'warmup_min': 82,
        'weight_decay': 0.0003146427050142341,
    },

    scenario(12, 16): {
        'batch_size': 24,
        'd_margin': 0.017223239272635218,
        'e_margin_scale': 0.6793211838787142,
        'filter_K_partial': 21,
        'fisher_alpha': 3508.2328843195514,
        'freeze_bn_stats_partial': True,
        'grad_clip_partial': 0.3695518546882944,
        'grad_clip_value_partial': 0.25,
        'include_warmup_support': True,
        'lambda_eata': 2.219775545733619,
        'learning_rate': 0.0004881560541054392,
        'lr_decay': 0.3859865389852272,
        'max_fisher_updates': 256,
        'memory_size': 768,
        'momentum': 0.6637877167307819,
        'num_epochs': 22,
        'online_fisher': True,
        'partial_module_bundle': 'conv1',
        'pre_learning_rate': 0.00040664222340180853,
        'quantile': 0.5019455542829452,
        'safety_keep_frac_partial': 0.46263262582633596,
        'step_size': 61,
        'steps': 5,
        'tau': 7,
        'temperature': 1.8040851754064766,
        'train_classifier_partial': True,
        'train_scope': 'partial',
        'use_eata_reg': False,
        'use_eata_select': True,
        'use_quantile': False,
        'warmup_min': 119,
        'weight_decay': 4.529866866418434e-05,
    },
}

EEG_ACCUP_SCENARIO_OVERRIDES = {
    # Optuna best overrides for EEG (study: tta_optuna)
    # study: EEG_ACCUP_0_11 (trial 85, f1 0.4791)
    scenario(0, 11): {
        'learning_rate': 1.0133453612043214e-04,
        'pre_learning_rate': 4.958066715039774e-04,
        'tau': 25,
        'temperature': 2.914722920741117,
        'warmup_min': 205,
        'quantile': 0.2529354283324555,
        'use_quantile': True,
        'safety_keep_frac': 0.337384307833167,
        'lambda_eata': 1.9022948836924023,
        'e_margin_scale': 0.8675286976378526,
        'd_margin': 0.13809420076977189,
        'memory_size': 3584,
        'fisher_alpha': 2425.599851216968,
        'use_eata_select': True,
        'use_eata_reg': True,
        'online_fisher': False,
        'include_warmup_support': True,
        'max_fisher_updates': -1,
        'train_scope': 'bn_only',
        'freeze_bn_stats_bn_only': True,
        'filter_K': 13,
        'filter_K_bn_only': 13,
        'safety_keep_frac_bn_only': 0.337384307833167,
        'grad_clip': 0.2978930706551386,
        'grad_clip_bn_only': 0.2978930706551386,
        'grad_clip_value': 0.1,
        'grad_clip_value_bn_only': 0.1,
        'times_hidden_channels': 256,
        'times_num_layers': 4,
        'times_dropout': 0.07246761783144418,
        'times_ffn_expansion': 2.6790973501683997,
        'times_patch_base': 16,
        'times_patch_scale': 1.8665391607957145,
        'times_patch_count': 4
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
    # study: EEG_ACCUP_7_18 (trial 11, f1 0.7082)
    scenario(7, 18): {
        'learning_rate': 1.2347528062767586e-04,
        'pre_learning_rate': 1.3158557340444023e-04,
        'tau': 18,
        'temperature': 2.11361357557397,
        'warmup_min': 30,
        'quantile': 0.14167042020635323,
        'use_quantile': True,
        'safety_keep_frac': 0.17911779868354522,
        'lambda_eata': 0.566842805225137,
        'e_margin_scale': 0.6894333998182214,
        'd_margin': 0.10293557930545941,
        'memory_size': 4096,
        'fisher_alpha': 1504.328161917314,
        'use_eata_select': True,
        'use_eata_reg': False,
        'online_fisher': False,
        'include_warmup_support': False,
        'max_fisher_updates': 256,
        'train_scope': 'bn_only',
        'freeze_bn_stats_bn_only': True,
        'filter_K': 5,
        'filter_K_bn_only': 5,
        'safety_keep_frac_bn_only': 0.17911779868354522,
        'grad_clip': 0.15178133412554715,
        'grad_clip_bn_only': 0.15178133412554715,
        'grad_clip_value': None,
        'grad_clip_value_bn_only': None,
        'times_hidden_channels': 256,
        'times_num_layers': 3,
        'times_dropout': 0.23114491509075497,
        'times_ffn_expansion': 1.3038719409562929,
        'times_patch_base': 96,
        'times_patch_scale': 1.3303717842798888,
        'times_patch_count': 4
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
    # study: EEG_ACCUP_9_14 (trial 7, f1 0.6649)
    scenario(9, 14): {
        'learning_rate': 1.0122614999144783e-05,
        'pre_learning_rate': 4.827670468678035e-04,
        'tau': 24,
        'temperature': 2.590079127893502,
        'warmup_min': 43,
        'quantile': 0.2252314614805666,
        'use_quantile': False,
        'safety_keep_frac': 0.36409531702704145,
        'lambda_eata': 1.7037085189188386,
        'e_margin_scale': 0.7625502822020727,
        'd_margin': 0.018365476145257163,
        'memory_size': 1280,
        'fisher_alpha': 5609.55348418654,
        'use_eata_select': False,
        'use_eata_reg': True,
        'online_fisher': True,
        'include_warmup_support': True,
        'max_fisher_updates': -1,
        'train_scope': 'partial',
        'train_classifier_partial': False,
        'partial_module_bundle': 'conv12',
        'freeze_bn_stats_partial': False,
        'filter_K': 19,
        'filter_K_partial': 19,
        'safety_keep_frac_partial': 0.36409531702704145,
        'grad_clip': 0.7677470739699619,
        'grad_clip_partial': 0.7677470739699619,
        'grad_clip_value': 0.25,
        'grad_clip_value_partial': 0.25,
        'times_hidden_channels': 160,
        'times_num_layers': 2,
        'times_dropout': 0.2487702914866347,
        'times_ffn_expansion': 1.6822492900189818,
        'times_patch_base': 72,
        'times_patch_scale': 1.4928123091088987,
        'times_patch_count': 3
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

