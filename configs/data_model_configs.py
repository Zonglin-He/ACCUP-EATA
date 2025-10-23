import os
import torch

def get_dataset_class(dataset_name): # 根据给定数据集名称字符串返回对应的数据集配置类
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name)) # 如果数据集名称不在全局变量中，抛出未实现错误
    return globals()[dataset_name]

class EEG(): # 数据集 (EEG) 参数
    def __init__(self):
        super(EEG, self).__init__() # 调用父类的初始化方法
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.scenarios = [("0", "11"), ("12", "5"), ("7", "18"), ("16", "1"), ("9", "14")]
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2

        # features
        self.mid_channels = 16
        self.final_out_channels = 8
        self.features_len = 65 # for my model
        self.AR_hid_dim = 8

        # AR Discriminator
        self.disc_hid_dim = 256
        self.disc_AR_bid= False
        self.disc_AR_hid = 128
        self.disc_n_layers = 1
        self.disc_out_dim = 1

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100

        # augmentations
        self.jitter_scale_ratio = 1.0 # 1.5
        self.jitter_ration = 0.5 # 2

        # TimesNet defaults
        self.times_hidden_channels = 128
        self.times_num_layers = 3
        self.times_patch_lens = [16, 32, 64]
        self.times_dropout = 0.1
        self.times_ffn_expansion = 2


class FD(): # 数据集 (FD) 参数
    def __init__(self):
        super(FD, self).__init__()
        self.sequence_len = 5120
        self.scenarios = [("0", "1"), ("1", "2"), ("3", "1"), ("1", "0"), ("2", "3")]
        self.class_names = ['Healthy', 'D1', 'D2']
        self.num_classes = 3
        self.shuffle = True
        self.drop_last = False
        self.normalize = True

        # Model configs
        self.input_channels = 1
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.5

        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.AR_hid_dim = 128

        # augmentations
        self.jitter_scale_ratio = 1.0 # 2
        self.jitter_ratio = 0.5 # 0.1

        # TimesNet defaults
        self.times_hidden_channels = 256
        self.times_num_layers = 3
        self.times_patch_lens = [32, 64, 128]
        self.times_dropout = 0.1
        self.times_ffn_expansion = 2

class HAR(): # 数据集 (HAR) 参数
    def __init__(self):
        super(HAR, self)
        self.scenarios = [("2", "11"), ("6", "23"), ("7", "13"), ("9", "18"), ("12", "16")]

        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # CNN and RESNET features
        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1
        # self.features_len = 18 for sequential methods
        self.AR_hid_dim = 128

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128

        # augmentations.
        self.jitter_scale_ratio = 1.0 # 1.1
        self.jitter_ratio = 0.5 # 0.8

        # TimesNet defaults
        self.times_hidden_channels = 128
        self.times_num_layers = 2
        self.times_patch_lens = [4, 8, 16]
        self.times_dropout = 0.1
        self.times_ffn_expansion = 2

