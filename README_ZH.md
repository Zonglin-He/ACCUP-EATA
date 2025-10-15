# ACCUP+EATA（中文 README）

**Augmented Contrastive Clustering with Uncertainty-Aware Prototyping + Entropy-based Adversarial Test-time Adaptation（PyTorch）**

> 面向 EEG / FD / HAR 等时序分类任务的**无标签测试时自适应（TTA）**。在 ACCUP 的“原型 + 多视图对比”框架上，集成 **EATA** 的样本选择与遗忘正则，提升稳健性并减小坏伪标签带来的退化。

---

## 目录

* [方法总览](#方法总览)
* [关键亮点](#关键亮点)
* [仓库结构](#仓库结构)
* [环境与安装](#环境与安装)
* [数据准备](#数据准备)
* [快速上手](#快速上手)

  * [A. 源域预训练](#a-源域预训练)
  * [B. 测试时自适应（ACCUP+EATA）](#b-测试时自适应accupeata)
  * [C. 命令行一键运行](#c-命令行一键运行)
* [重要超参数](#重要超参数)
* [输出与结果复现](#输出与结果复现)
* [常见问题](#常见问题)
* [引用](#引用)
* [许可证](#许可证)
* [贡献](#贡献)

---

## 方法总览

**ACCUP**：

1. 生成多视图（raw/aug），进行**特征或 logit 集成**；
2. 维护**每类低熵支持样本**，构建**不确定性感知原型**；
3. 对每个目标样本比较“**原型预测熵** vs **增强集成预测熵**”，选更低者生成伪标签；
4. 用**多视图监督对比损失（SupCon）**做轻量在线更新（主要更新 BN 与少量卷积层）。

**EATA**（与 ACCUP 无缝组合）：

* **选择（select）**：仅让**低熵且密度合理**的样本参与更新（分位数阈 + Top-K 筛选）。
* **正则（reg）**：Fisher/近似 Fisher 约束，抑制对源域能力的灾难性遗忘。
* **梯度裁剪**：与选择/正则配合，稳定在线优化。

---

## 关键亮点

* **双熵择优伪标签**：原型 vs 集成，动态选择更可靠的一侧。
* **不确定性感知原型**：每类维护 **Top-K 低熵**支持样本。
* **EATA 选择+正则**：更稳健、更抗噪伪标签。
* **低侵入更新**：仅 BN + 少量卷积层参数，支持**逐批在线**适配。
* **即插即用**：开启 `use_eata` 即得 ACCUP+EATA，也可关闭做纯 ACCUP 对比。

---

## 仓库结构

```
.
├─ algorithms/
│  ├─ accup.py                  # ACCUP 主算法
│  ├─ eata_utils.py             # EATA 选择 & 正则组件（本项目集成）
│  └─ base_tta_algorithm.py     # TTA 基类/通用工具
├─ trainers/
│  └─ tta_trainer.py            # 训练/测试时适配脚本
├─ configs/
│  ├─ data_model_configs.py     # 数据/模型配置（EEG/FD/HAR）
│  └─ tta_hparams_new.py        # 超参数（ACCUP & EATA）
├─ dataloader/
│  ├─ dataloader.py             # 原始 DataLoader
│  └─ demo_dataloader.py        # 增强 DataLoader（返回 raw, aug[, aug2]）
├─ loss/
│  └─ sup_contrast_loss.py      # 监督对比损失
├─ pre_train_model/
│  └─ pre_train_model.py        # 源域预训练封装
├─ results/
│  └─ *.csv / 日志 / 可视化
└─ README.md / requirements.txt / LICENSE / ...
```

> 你的工程名可能略有不同，按需调整 import 路径。

---

## 环境与安装

* Python ≥ 3.9
* PyTorch ≥ 1.12（CUDA 对应版本；CPU 亦可）
* numpy / scipy / scikit-learn / torchvision / pandas

```bash
pip install -r requirements.txt
# 或
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn pandas
```

---

## 数据准备

目录示例：

```
data/<DATASET>/<SCENARIO>/
  ├─ train_<domain_id>.pt
  └─ test_<domain_id>.pt
```

`.pt` 至少含：

* `"samples"`：内部标准化为 `(N, C, L)`
* `"labels"`：仅评估用（TTA 无需标签）

标准化策略：

* `dataloader.py` 对 **raw** 做每通道 z-score；
* `demo_dataloader.py` 对 **raw 与 aug 分别**标准化，避免泄漏。

---

## 快速上手

### A. 源域预训练

```python
from configs.data_model_configs import HAR as DatasetCfg
from configs.tta_hparams_new import HAR as HpCfg
from dataloader.dataloader import data_generator
from models.da_models import get_backbone_class
from pre_train_model.pre_train_model import PreTrainModel
from your_module.pretrain import pre_train_model

dataset_cfg = DatasetCfg()
hp_all = HpCfg()
train_hp = hp_all.train_params
accup_hp = hp_all.alg_hparams['ACCUP']

backbone = get_backbone_class('CNN')
src_dl = data_generator(
    data_path="data/HAR/2_10_11", domain_id="2",
    dataset_configs=dataset_cfg, hparams=train_hp, dtype="train"
)

src_state_dict, pretrained_model = pre_train_model(
    backbone=backbone,
    configs=dataset_cfg,
    hparams={**train_hp, **accup_hp},
    src_dataloader=src_dl,
    avg_meter=None, logger=None, device="cuda:0"
)
```

### B. 测试时自适应（ACCUP+EATA）

```python
import torch, torch.optim as optim
from dataloader.demo_dataloader import data_generator_demo
from algorithms.accup import ACCUP

dataset_cfg = DatasetCfg()
hp_all = HpCfg()
train_hp = hp_all.train_params
accup_hp = hp_all.alg_hparams['ACCUP']
eata_hp  = hp_all.alg_hparams['EATA']   # ★ EATA 配置

trg_dl = data_generator_demo(
    data_path="data/HAR/2_10_11", domain_id="11",
    dataset_configs=dataset_cfg, hparams=train_hp, dtype="test"
)

tta = ACCUP(
    configs=dataset_cfg,
    hparams={**train_hp, **accup_hp, **eata_hp, "use_eata": True},  # ★ 开启 EATA
    model=pretrained_model,
    optimizer=lambda p: optim.Adam(p, lr=accup_hp['learning_rate'])
).to("cuda:0")

tta.model.train()  # TTA 期间需保持 train 以更新 BN
for (raw, aug, _), _, _ in trg_dl:
    raw, aug = raw.float().cuda(), aug.float().cuda()
    _ = tta.forward((raw, aug))
```

### C. 命令行一键运行

Windows 示例（与你日志一致）：

```bash
python -m trainers.tta_trainer ^
  --data_path .\data\Dataset ^
  --dataset HAR ^
  --save_dir .\results\tta_experiments_logs ^
  --device cpu ^
  --use_eata 1
```

Linux/macOS：

```bash
python -m trainers.tta_trainer \
  --data_path ./data/Dataset \
  --dataset HAR \
  --save_dir ./results/tta_experiments_logs \
  --device cuda:0 \
  --use_eata 1
```

---

## 重要超参数

**ACCUP**

* `filter_K`：每类保留的低熵支持数（`-1`=全保留；常用 10）
* `tau`：原型相似度温度
* `temperature`：SupCon 温度
* `learning_rate`：TTA 学习率
* `steps`：每次 `forward` 的 TTA 更新步数（通常 1）
* `bn_only`：是否仅更新 BN（搭配少量卷积层可更稳）

**EATA**（与日志字段一致的典型设置）

* `select=True, reg=True`
* `use_quantile=True, quantile=0.9`
* `e_margin_scale=0.7`, `d_margin=0.05`
* `filter_K=10`
* `warmup_min=12`
* `safety_keep_frac=0.65`
* 配合 `grad_clip=0.1` 稳定优化

---

## 输出与结果复现

* `--save_dir` 下自动生成场景级 **CSV**（列：`scenario, run, acc, f1_score, auroc`），底部含 **mean/std** 汇总。
* 控制台打印包括：`Pretraining stage...`、`[GradClip] ...`、`[EATA] selected ...`、`Average current f1_scores` 等。
* 你可以把结果 CSV 汇总成表或直接粘贴 Excel 截图到 README。

---

## 常见问题

* **`torch.load(..., weights_only=False)` 警告**：若文件不完全可信，建议显式设为 `weights_only=True` 并用 `torch.serialization.add_safe_globals` 进行白名单；仅本地自管文件可忽略。
* **小 batch 的 BN 不稳定**：可调小 LR、仅 BN 更新、启用更严格的 EATA 选择，或提高 `grad_clip` 约束。
* **原型内样本是否无限增长**：默认 Top-K 低熵保留；长流式可加滑动窗口/衰减策略。

---

## 引用

```bibtex
@article{Gong2025ACCUParXiv,
  author  = {Peiliang Gong and Mohamed Ragab and Min Wu and Zhenghua Chen and Yongyi Su and Xiaoli Li and Daoqiang Zhang},
  title   = {Augmented Contrastive Clustering with Uncertainty-Aware Prototyping for Time Series Test Time Adaptation},
  journal = {arXiv preprint arXiv:2501.01472},
  year    = {2025}
}

@inproceedings{EATA,
  title   = {Efficient Test-Time Adaptation via Entropy Regularization},
  author  = {…},
  booktitle = {…},
  year    = {…}
}
```

> 请按你的论文/实现替换 EATA 完整条目。

---

## 许可证

推荐 MIT 或 Apache-2.0（详见 `LICENSE`）。

---

## 贡献

欢迎 PR：新的时序 backbone（TCN/ResNet/LSTM/iTransformer）、更多增强与集成策略、可视化（t-SNE/UMAP）、更丰富的数据集模板等。


