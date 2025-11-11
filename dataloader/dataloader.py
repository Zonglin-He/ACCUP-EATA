import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
import os
import numpy as np

from utils.utils import safe_torch_load


class Load_Dataset(Dataset):  # 加载数据集
    def __init__(self, dataset, dataset_configs):  # dataset_configs包含数据集的配置信息
        super().__init__()  # 调用父类的初始化方法
        self.num_channels = dataset_configs.input_channels  # 获取输入通道数

        # Load samples
        x_data = dataset["samples"]

        if len(x_data.shape) == 2:  # 检查样本数据的维度
            x_data = x_data.unsqueeze(1)  # 若为(N, L)则加一维变(N, 1, L)
        elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:  # 若为(N, C, L)但C不对则转置
            x_data = x_data.transpose(0, 2, 1)  # 转为(N, L, C)再转为(N, C, L)

        if isinstance(x_data, np.ndarray):  # 将numpy转为tensor
            x_data = torch.from_numpy(x_data)

        y_data = dataset.get("labels")  # 获取标签数据
        if y_data is not None and isinstance(y_data, np.ndarray):  # 将numpy转为tensor
            y_data = torch.from_numpy(y_data)

        if dataset_configs.normalize:
            # 如果配置要求归一化，则计算每个通道的均值(data_mean)和标准差(data_std) 并创建归一化变换
            data_mean = torch.mean(x_data, dim=(0, 2))
            data_std = torch.std(x_data, dim=(0, 2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std)

        self.x_data = x_data.float()  # 将样本数据转为float类型
        self.y_data = y_data.long() if y_data is not None else None  # 将标签数据转为long类型（若存在）
        self.len = x_data.shape[0]  # 样本数量

    def __getitem__(self, index):  # 按索引获取样本和标签
        x = self.x_data[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)
            # 这里 reshape(self.num_channels, -1, 1) 将数据变为 (C, L, 1) 再Normalize，最后再reshape回原形状 (C, L)
        y = self.y_data[index] if self.y_data is not None else None  # 获取标签（若存在）
        return x, y, index

    def __len__(self):  # 返回数据集长度
        return self.len


class Load_ALL_Dataset(Dataset):  # 加载训练和测试数据集的结合体
    def __init__(self, train_dataset, test_dataset, dataset_configs):
        super().__init__()
        self.num_channels = dataset_configs.input_channels

        # Load samples
        x_train_data = train_dataset["samples"]  # 加载训练集样本
        x_test_data = test_dataset["samples"]  # 加载测试集样本
        # Load labels
        y_train_data = train_dataset.get("labels")  # 加载训练集标签
        y_test_data = test_dataset.get("labels")  # 加载测试集标签

        if isinstance(x_train_data, np.ndarray):  # 若样本是numpy数组则用np.concatenate拼接
            x_data = np.concatenate([x_train_data, x_test_data], axis=0)
            y_data = np.concatenate([y_train_data, y_test_data], axis=0)
        else:  # 否则用torch.cat拼接
            x_data = torch.cat([x_train_data, x_test_data], dim=0)
            y_data = torch.cat([y_train_data, y_test_data], dim=0)

        if len(x_data.shape) == 2:  # 检查样本数据的维度
            x_data = x_data.unsqueeze(1)  # 若为(N, L)则加一维变(N, 1, L)
        elif len(x_data.shape) == 3 and x_data.shape[1] != self.num_channels:  # 若为(N, C, L)但C不对则转置
            x_data = x_data.transpose(0, 2, 1)  # 转为(N, L, C)再转为(N, C, L)
        # 与Load_Dataset一样的逻辑

        if isinstance(x_data, np.ndarray):  # 将numpy转为tensor
            x_data = torch.from_numpy(x_data)

        if y_data is not None and isinstance(y_data, np.ndarray):  # 将numpy转为tensor
            y_data = torch.from_numpy(y_data)

        if dataset_configs.normalize:  # 如果配置要求归一化，则计算每个通道的均值(data_mean)和标准差(data_std) 并创建归一化变换
            data_mean = torch.mean(x_data, dim=(0, 2))
            data_std = torch.std(x_data, dim=(0, 2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std)

        self.x_data = x_data.float()
        self.y_data = y_data.long() if y_data is not None else None
        self.len = x_data.shape[0]

    def __getitem__(self, index):  # 按索引获取样本和标签
        x = self.x_data[index]
        if self.transform:
            x = self.transform(self.x_data[index].reshape(self.num_channels, -1, 1)).reshape(self.x_data[index].shape)
        y = self.y_data[index] if self.y_data is not None else None
        return x, y, index

    def __len__(self):  # 返回数据集长度
        return self.len


def data_generator(data_path, domain_id, dataset_configs, hparams, dtype):
    # 按给定的数据根目录、域 ID、数据集类型（train/val/test）和批大小，构建并返回一个 DataLoader
    dataset_file = safe_torch_load(os.path.join(data_path, f"{dtype}_{domain_id}.pt"))  # 定位并加载文件

    dataset = Load_Dataset(dataset_file, dataset_configs)  # 使用 Load_Dataset 类加载数据集

    if dtype == "test":  # 测试集不打乱，是否丢弃末批由超参控制
        shuffle = False
        drop_last = hparams.get("drop_last_test", True)
    else:  # 训练/验证集根据配置决定是否打乱和丢弃最后一个不满批次
        shuffle = dataset_configs.shuffle
        drop_last = dataset_configs.drop_last

    # Dataloaders
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=hparams["batch_size"],
                                              shuffle=shuffle,
                                              drop_last=drop_last,
                                              num_workers=0)  # 创建 DataLoader

    return data_loader


def whole_targe_data_generator(data_path, domain_id, dataset_configs, hparams):
    # 将目标域 (source=train, target=test) 全部数据整合，用于计算整体指标
    train_dataset_file = safe_torch_load(os.path.join(data_path, f"{'train'}_{domain_id}.pt"))  # 加载训练集
    test_dataset_file = safe_torch_load(os.path.join(data_path, f"{'test'}_{domain_id}.pt"))  # 加载测试集

    # Loading datasets
    whole_dataset = Load_ALL_Dataset(train_dataset_file, test_dataset_file, dataset_configs)  # 整合数据集

    shuffle = False  # 整体数据不打乱且不丢弃最后一个不满批次
    drop_last = hparams.get("drop_last_eval",
                            hparams.get("drop_last_test", True))

    # Dataloaders
    data_loader = torch.utils.data.DataLoader(dataset=whole_dataset,
                                              batch_size=hparams["batch_size"],
                                              shuffle=shuffle,
                                              drop_last=drop_last,
                                              num_workers=0)  # 创建 DataLoader

    return data_loader
