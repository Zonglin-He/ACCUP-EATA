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
            'NoAdap' : {'pre_learning_rate': 0.001}
        }

class HAR():
    def __init__(self):
        super(HAR, self).__init__()
        self.train_params = {
            'num_epochs': 40,
            'batch_size': 32,
            'weight_decay': 1e-4,
            'step_size': 50,
            'lr_decay': 0.5,
            'steps': 1,
            'optim_method':'adam',
            'momentum':0.9
        }
        # 关键：加入 EATA 的开关和超参
        self.alg_hparams = {
            'ACCUP': {
                'pre_learning_rate': 0.001,
                'learning_rate': 3e-4,
                'filter_K': 5,
                'tau': 20,
                'temperature': 0.7,

                # EATA 相关
                'use_eata_select': True,
                'use_eata_reg': True,
                # e_margin = ln(num_classes) * e_margin_scale
                'e_margin_scale': 0.55,   # 建议先试 0.45（更严格）或 0.35（更宽松）
                'd_margin': 0.04,         # 密度阈值，0.03~0.08 之间试两档
                'lambda_eata': 1.0,# 正则强度，0.5~1.5 之间可扫两档
                'e_margin_scale': 0.70,
                'd_margin': 0.0,
                'memory_size': 4096,

            },
            'NoAdap': {'pre_learning_rate': 0.001}
        }

