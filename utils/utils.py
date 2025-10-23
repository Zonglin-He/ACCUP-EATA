import torch
import torch.nn.functional as F
from torch import nn as nn

import pickle
import random
import os
import sys
import logging
import numpy as np
import pandas as pd
from shutil import copy
from datetime import datetime
from einops import rearrange

from skorch import NeuralNetClassifier  # for DIV Risk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

try:
    import torch.serialization as _torch_serialization

    _torch_serialization.add_safe_globals([
        np.core.multiarray._reconstruct,
        np.ndarray,
    ])
except Exception:
    pass

_torch_original_load = torch.load


def _torch_load_with_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    kwargs.setdefault("pickle_module", pickle)
    return _torch_original_load(*args, **kwargs)


torch.load = _torch_load_with_compat


def safe_torch_load(*args, **kwargs):
    """Compat loader kept for clarity; delegates to the patched torch.load."""
    return torch.load(*args, **kwargs)


class AverageMeter(object): #计算并存储平均值和当前值

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def fix_randomness(SEED): #设置随机种子以确保结果可复现
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _logger(logger_name, level=logging.DEBUG): #创建日志记录器，返回日志
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = "%(message)s"
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


def starting_logs(data_type,da_method, exp_log_dir, src_id, tgt_id, run_id): #初始化日志记录器和日志目录
    log_dir = os.path.join(exp_log_dir, src_id + "_to_" + tgt_id + "_run_" + str(run_id))
    os.makedirs(log_dir, exist_ok=True)
    log_file_name = os.path.join(log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    return logger, log_dir


def save_checkpoint(home_path, algorithm, selected_scenarios, dataset_configs, log_dir, hparams): #保存模型检查点
    save_dict = {
        "x-domains": selected_scenarios,
        "configs": dataset_configs.__dict__,
        "hparams":  dict(hparams),
        "model_dict": algorithm.state_dict()
    }
    # save classification report
    save_path = os.path.join(home_path, log_dir, "checkpoint.pt")

    torch.save(save_dict, save_path)


def weights_init(m): #初始化神经网络层的权重
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.1)
        m.bias.data.fill_(0)

def _calc_metrics(pred_labels, true_labels, log_dir, home_path, target_names): #计算分类指标并保存分类报告
    pred_labels = np.array(pred_labels).astype(int)
    true_labels = np.array(true_labels).astype(int)

    r = classification_report(true_labels, pred_labels, target_names=target_names, digits=6, output_dict=True)

    df = pd.DataFrame(r)
    accuracy = accuracy_score(true_labels, pred_labels)
    df["accuracy"] = accuracy
    df = df * 100

    # save classification report
    file_name = "classification_report.xlsx"
    report_Save_path = os.path.join(home_path, log_dir, file_name)
    df.to_excel(report_Save_path)

    return accuracy * 100, r["macro avg"]["f1-score"] * 100

def _calc_metrics_pretrain(src_pred, src_true,trg_pred, trg_true,  log_dir, home_path, target_names):  #计算预训练模型在源域和目标域上的分类指标

    src_pred_labels = np.array(src_pred).astype(int)
    src_true_labels = np.array(src_true).astype(int)
    trg_pred_labels = np.array(trg_pred).astype(int)
    trg_true_labels = np.array(trg_true).astype(int)

    src_rep = classification_report(src_true_labels, src_pred_labels, target_names=target_names, digits=6, output_dict=True)
    trg_rep = classification_report(trg_true_labels, trg_pred_labels, target_names=target_names, digits=6, output_dict=True)

    src_df = pd.DataFrame(src_rep)
    trg_df = pd.DataFrame(trg_rep)

    src_acc = accuracy_score(src_true_labels, src_pred_labels)
    trg_acc = accuracy_score(trg_true_labels, trg_pred_labels)

    return src_acc * 100, src_df["macro avg"]["f1-score"] * 100, trg_acc *100,  trg_df["macro avg"]["f1-score"] *100

import collections
def to_device(input, device): #将输入数据移动到指定设备（CPU或GPU）
    if torch.is_tensor(input):
        return input.to(device=device)
    elif isinstance(input, str):
        return input
    elif isinstance(input, collections.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError("Input must contain tensor, dict or list, found {type(input)}")

def copy_Files(destination): #备份关键代码文件到指定目录
    destination_dir = os.path.join(destination, "MODEL_BACKUP_FILES")
    os.makedirs(destination_dir, exist_ok=True)
    copy("main.py", os.path.join(destination_dir, "main.py"))
    copy("algorithms/algorithms.py", os.path.join(destination_dir, "algorithms.py"))
    copy(f"configs/data_model_configs.py", os.path.join(destination_dir, f"data_model_configs.py"))
    copy(f"configs/hparams.py", os.path.join(destination_dir, f"hparams.py"))
    copy(f"configs/sweep_params.py", os.path.join(destination_dir, f"sweep_params.py"))
    copy("utils.py", os.path.join(destination_dir, "utils.py"))

def get_iwcv_value(weight, error): #计算加权交叉验证风险
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, 'dimension mismatch!'
    weighted_error = weight * error
    return np.mean(weighted_error)

def get_dev_value(weight, error): #计算加权验证风险
    """
    :param weight: shape [N, 1], the importance weight for N source samples in the validation set
    :param error: shape [N, 1], the error value for each source sample in the validation set
    (typically 0 for correct classification and 1 for wrong classification)
    """
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, 'dimension mismatch!'
    weighted_error = weight * error
    cov = np.cov(np.concatenate((weighted_error, weight), axis=1), rowvar=False)[0][1]
    var_w = np.var(weight, ddof=1)
    eta = - cov / var_w
    return np.mean(weighted_error) + eta * np.mean(weight) - eta

class simple_MLP(nn.Module): #简单的多层感知机，用于域分类器
    def __init__(self, inp_units, out_units=2):
        super(simple_MLP, self).__init__()

        self.dense0 = nn.Linear(inp_units, inp_units//2)
        self.nonlin = nn.ReLU()
        self.output = nn.Linear(inp_units//2, out_units)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, **kwargs):
        x = self.nonlin(self.dense0(x))
        x = self.softmax(self.output(x))
        return x

def get_weight_gpu(source_feature, target_feature, validation_feature, configs, device): #计算源域和目标域样本的权重
    """
    :param source_feature: shape [N_tr, d], features from training set
    :param target_feature: shape [N_te, d], features from test set
    :param validation_feature: shape [N_v, d], features from validation set
    :return:
    """
    import copy
    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape
    source_feature = copy.deepcopy(source_feature.detach().cpu()) #source_feature.clone()
    target_feature = copy.deepcopy(target_feature.detach().cpu()) #target_feature.clone()
    source_feature = source_feature.to(device)
    target_feature = target_feature.to(device)
    all_feature = torch.cat((source_feature, target_feature), dim=0)
    all_label = torch.from_numpy(np.asarray([1] * N_s + [0] * N_t, dtype=np.int32)).long()

    feature_for_train, feature_for_test, label_for_train, label_for_test = train_test_split(all_feature, all_label,
                                                                                            train_size=0.8)
    learning_rates = [1e-1, 5e-2, 1e-2]
    val_acc = []
    domain_classifiers = []

    for lr in learning_rates:
        domain_classifier = NeuralNetClassifier(
            simple_MLP,
            module__inp_units = configs.final_out_channels * configs.features_len,
            max_epochs=30,
            lr=lr,
            device=device,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            callbacks="disable"
        )
        domain_classifier.fit(feature_for_train.float(), label_for_train.long())
        output = domain_classifier.predict(feature_for_test)
        acc = np.mean((label_for_test.numpy() == output).astype(np.float32))
        val_acc.append(acc)
        domain_classifiers.append(domain_classifier)

    index = val_acc.index(max(val_acc))
    domain_classifier = domain_classifiers[index]

    domain_out = domain_classifier.predict_proba(validation_feature.to(device).float())
    return domain_out[:, :1] / domain_out[:, 1:] * N_s * 1.0 / N_t


def calc_dev_risk(target_model, src_train_dl, tgt_train_dl, src_valid_dl, configs, device): #计算验证风险
    src_train_feats, _ = target_model.feature_extractor(src_train_dl.dataset.x_data.float().to(device))
    tgt_train_feats, _ = target_model.feature_extractor(tgt_train_dl.dataset.x_data.float().to(device))
    src_valid_feats, _ = target_model.feature_extractor(src_valid_dl.dataset.x_data.float().to(device))
    src_valid_pred = target_model.classifier(src_valid_feats)

    dev_weights = get_weight_gpu(src_train_feats.to(device), tgt_train_feats.to(device),
                                 src_valid_feats.to(device), configs, device)
    dev_error = F.cross_entropy(src_valid_pred, src_valid_dl.dataset.y_data.long().to(device), reduction='none')
    dev_risk = get_dev_value(dev_weights, dev_error.unsqueeze(1).detach().cpu().numpy())
    # iwcv_risk = get_iwcv_value(dev_weights, dev_error.unsqueeze(1).detach().cpu().numpy())
    return dev_risk


def calculate_risk(target_model, risk_dataloader, device): #计算给定数据集上的分类损失作为风险

    x_data = risk_dataloader.dataset.x_data
    y_data = risk_dataloader.dataset.y_data

    feat, _ = target_model.feature_extractor(x_data.float().to(device))
    pred = target_model.classifier(feat)
    cls_loss = F.cross_entropy(pred, y_data.long().to(device))
    return cls_loss.item()

def get_named_submodule(model, sub_name: str): #根据子模块名称获取子模块
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value): #根据子模块名称设置子模块的值
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)

# 可放到 trainers/eata_utils.py（或你项目中合适的位置）
# trainers/eata_utils.py
import math
import torch
import torch.nn.functional as F
from typing import Optional, Dict

class EATAMemory:
    def __init__(self, maxlen: int = 4096, device: str = "cpu"):
        self.maxlen = maxlen
        self.device = torch.device(device)
        self._feats = None    # [M, D], unit-normalized
        self._probs = None    # [M, C]

    def __len__(self):
        return 0 if self._feats is None else int(self._feats.size(0))

    def to(self, device: torch.device):
        """切换记忆库所使用的设备，并同步已缓存的数据。"""
        device = torch.device(device)
        if self.device == device:
            return self
        if self._feats is not None:
            self._feats = self._feats.to(device)
        if self._probs is not None:
            self._probs = self._probs.to(device)
        self.device = device
        return self

    @torch.no_grad()
    def push(self, feats: torch.Tensor, probs: torch.Tensor):
        if feats.device != self.device:
            self.to(feats.device)
        feats = F.normalize(feats.detach(), dim=1).to(self.device)
        probs = probs.detach().to(self.device)
        if self._feats is None:
            self._feats = feats[-self.maxlen:]
            self._probs = probs[-self.maxlen:]
            return
        self._feats = torch.cat([self._feats, feats], dim=0)[-self.maxlen:]
        self._probs = torch.cat([self._probs, probs], dim=0)[-self.maxlen:]

    @torch.no_grad()
    def density(self, feats: torch.Tensor, K: int = 5):
        """用记忆库计算每个当前样本的‘冗余度’（对记忆库特征的平均TopK余弦相似度）。
           值越大=越像过去见过的=越冗余；越小=越新颖。"""
        if self._feats is None or len(self) == 0:
            dens = torch.full((feats.size(0),), float("nan"), device=feats.device)
            return dens, {"mean": float("nan"), "med": float("nan"), "p90": float("nan")}, 0

        feats = F.normalize(feats, dim=1)
        mem_feats = self._feats.to(feats.device)
        sims = feats @ mem_feats.T                       # [B, M]
        k_eff = min(K, sims.size(1))
        topk, _ = torch.topk(sims, k=k_eff, dim=1)       # [B, k_eff]
        dens = topk.mean(dim=1)                          # 越大越“像”记忆库
        stats = {
            "mean": float(torch.nanmean(dens).item()),
            "med" : float(torch.nanmedian(dens).item()),
            "p90" : float(torch.nanquantile(dens, 0.90).item())
        }
        return dens, stats, k_eff

    @torch.no_grad()
    def mean_probs(self) -> Optional[torch.Tensor]:
        return None if self._probs is None else self._probs.mean(dim=0)


def softmax_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    return -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=1)


import math
import torch

@torch.no_grad()
def select_eata_indices(
    logits: torch.Tensor,                   # [B, C]
    feats: torch.Tensor,                    # [B, D]
    num_classes: int,
    memory,                                  # EATAMemory，需提供 .density(feats, K)
    # —— 原有超参 —— #
    e_margin_scale: float = 0.85,
    d_margin: float = 0.0,
    K: int = 5,
    temperature: float = 0.7,
    warmup_min: int = 128,
    use_quantile: bool = True,
    quantile: float = 0.70,
    safety_keep_frac: float = 0.125,
    # —— 新增更稳控制项 —— #
    e_margin_min: float = 0.25,            # 熵阈的“地板”（避免被分位数拉得太低）
    balance_per_class: bool = True,        # 是否做类均衡选样
    min_per_class: int = 2,                # 每类保底样本
    fuse_density: bool = True,             # 是否与密度融合排序
    density_high_is_good: bool = False     # dens 越小越近(=距离)；若是“相似度”，设为 True
):
    """
    返回:
        idx: 选中样本下标 [<=B]
        log: 文本日志
    依赖:
        - softmax_entropy_from_logits(logits)
        - memory.density(feats, K) -> (dens[B], stats{mean/med/p90}, k_eff)
    """

    device = logits.device
    B = int(logits.size(0))
    C = int(num_classes)

    # ---------- 小工具：安全 z-score ----------
    def _zsafe(x: torch.Tensor) -> torch.Tensor:
        # 避免 numel<2 时产生 dof 警告
        if x.numel() < 2:
            return torch.zeros_like(x)
        return (x - x.mean()) / (x.std(correction=0) + 1e-6)

    # ---------- 首轮/暖启动 ----------
    is_first_round = (len(memory) == 0)
    in_warmup = (len(memory) < warmup_min)
    # 首轮更保守，不要一下子强回填太多
    keep_frac_eff = min(safety_keep_frac, 0.10) if is_first_round else safety_keep_frac
    n_min = max(1, int(B * keep_frac_eff))

    # ---------- 熵（温度缩放可选） ----------
    if temperature is not None and temperature != 1.0:
        ent = softmax_entropy_from_logits(logits / temperature)  # [B]
    else:
        ent = softmax_entropy_from_logits(logits)

    # ---------- 低熵阈值（先上界，再地板） ----------
    H = math.log(max(2, C))                # ln(C)
    base = H * e_margin_scale              # 上界
    if use_quantile and B > 1:
        qth = ent.quantile(quantile).item()
        e_th = min(base, qth)
    else:
        qth = float("nan")
        e_th = base
    e_th = max(e_th, e_margin_min)         # 关键：不让阈值掉穿地板

    # Step 1: 低熵候选
    cand = torch.where(ent < e_th)[0]

    # ---------- 密度过滤（去冗余/取“更近”） ----------
    if hasattr(memory, "to"):
        memory.to(feats.device)
    dens, stats, k_eff = memory.density(feats, K=K)
    if k_eff > 0 and cand.numel() > 0:
        dens_c = dens[cand]
        # 用候选的中位数更稳；退化到全局统计
        if torch.isfinite(dens_c).any():
            med = torch.nanmedian(dens_c).item()
        else:
            med = float(stats.get("med", float("nan")))
        if math.isnan(med):
            med = float(stats.get("mean", 0.0))

        if not density_high_is_good:
            # dens 是距离：越小越近，保留 <= (med - d_margin)
            keep = (torch.isnan(dens_c) | (dens_c <= (med - d_margin)))
        else:
            # dens 是相似度：越大越近，保留 >= (med + d_margin)
            keep = (torch.isnan(dens_c) | (dens_c >= (med + d_margin)))
        cand = cand[keep]

    # ---------- 全局预算（随 quantile 自动缩放） ----------
    # 大致是 (1 - quantile) * B；暖启动/首轮更保守
    budget = max(1, int(B * (1.0 - quantile)))
    if is_first_round or (k_eff == 0) or in_warmup:
        budget = max(C * max(1, min_per_class), budget // 2)

    # cand 为空则用低熵 top-k 兜底，但不超过预算和 n_min
    if cand.numel() == 0:
        k = max(1, min(budget, n_min))
        cand = torch.topk(-ent, k).indices

    # ---------- 类均衡选样 + 融合排序 ----------
    if balance_per_class:
        pred = logits.argmax(dim=1)
        selected = []

        # 每类先取保底
        per_take = max(1, min_per_class)
        for c in range(C):
            idx_c = cand[pred[cand] == c]
            if idx_c.numel() == 0:
                continue

            if fuse_density and (k_eff > 0):
                d = dens[idx_c]
                s_ent = -ent[idx_c]                              # 低熵更好
                s_den = (-d if not density_high_is_good else d)  # 更近更好
                score = _zsafe(s_ent) + _zsafe(s_den)
                order = torch.argsort(-score)
            else:
                order = torch.argsort(ent[idx_c])                # 仅按熵（升序）

            take = min(per_take, idx_c.numel())
            selected.append(idx_c[order[:take]])

        if len(selected) > 0:
            selected = torch.cat(selected, dim=0)
        else:
            selected = cand[:0]  # 空张量

        # 还没到预算则做全局补齐
        if selected.numel() < budget:
            # 用 “B维布尔表” 排除已选，效率高且不依赖 CPU set()
            already = torch.zeros(B, dtype=torch.bool, device=device)
            if selected.numel() > 0:
                already[selected] = True
            remain = cand[~already[cand]]

            if remain.numel() > 0:
                if fuse_density and (k_eff > 0):
                    d = dens[remain]
                    s_ent = -ent[remain]
                    s_den = (-d if not density_high_is_good else d)
                    score = _zsafe(s_ent) + _zsafe(s_den)
                    order = torch.argsort(-score)
                else:
                    order = torch.argsort(ent[remain])

                need = min(budget - selected.numel(), remain.numel())
                selected = torch.cat([selected, remain[order[:need]]], dim=0)

        idx = selected[:budget]

    else:
        # 不做类均衡：按融合分数/熵直接截取到 budget
        if cand.numel() > budget:
            if fuse_density and (k_eff > 0):
                d = dens[cand]
                s_ent = -ent[cand]
                s_den = (-d if not density_high_is_good else d)
                score = _zsafe(s_ent) + _zsafe(s_den)
                order = torch.argsort(-score)
            else:
                order = torch.argsort(ent[cand])
            idx = cand[order[:budget]]
        else:
            idx = cand

    # ---------- 少样本兜底（不超过预算） ----------
    if idx.numel() < n_min:
        need = min(B, n_min)
        extra = torch.topk(-ent, k=need).indices
        merged = torch.unique(torch.cat([idx, extra], dim=0))
        if merged.numel() > budget:
            merged = merged[:budget]
        idx = merged

    # ---------- 日志 ----------
    mean_ = float(stats.get('mean', float('nan'))) if isinstance(stats, dict) else float('nan')
    med_  = float(stats.get('med',  float('nan'))) if isinstance(stats, dict) else float('nan')
    p90_  = float(stats.get('p90',  float('nan'))) if isinstance(stats, dict) else float('nan')
    log = (
        f"[EATA] selected {int(idx.numel())}/{B} | "
        f"K={int(k_eff)} dens[mean/med/p90]={mean_:.3f}/{med_:.3f}/{p90_:.3f} | "
        f"e_th={e_th:.3f} (base={base:.3f}, q{int(quantile*100)}={qth:.3f})"
    )

    return idx, log




# ======= 集成示例（放到你的 ACCUP/TTA 训练循环里）=======
# 1) 初始化一次（如在 trainer.__init__）
#   self.eata_memory = EATAMemory(maxlen=4096, device=self.device)

# 2) 每个 test batch 内，在拿到 logits、feats 后调用：
#   idx = select_eata_indices(
#       logits=logits, feats=feats
#       num_classes=self.num_classes,
#       e_margin_scale=self.hparams.get('e_margin_scale', 0.55),
#       d_margin=self.hparams.get('d_margin', 0.04),
#       K=self.hparams.get('filter_K', 10),
#       memory=self.eata_memory,
#       temperature=self.hparams.get('temperature', 1.0),
#   )
#   # 用 idx 对 x / pseudo_y / loss 做后续 ACCUP+EATA 的自适应更新
