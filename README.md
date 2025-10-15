# ACCUP-EATA

# ACCUP+EATA (English README)

**Augmented Contrastive Clustering with Uncertainty-Aware Prototyping + Entropy-based Adversarial Test-time Adaptation (PyTorch)**

> Unlabeled **test-time adaptation (TTA)** for time-series classification (EEG/FD/HAR). We integrate **EATA** (selection + forgetting regularization) into the ACCUP framework to improve robustness and reduce degradation from noisy pseudo labels.

---

## Table of Contents

* [Overview](#overview)
* [Highlights](#highlights)
* [Repository Layout](#repository-layout)
* [Environment & Installation](#environment--installation)
* [Data Preparation](#data-preparation)
* [Quick Start](#quick-start)

  * [A. Source Pretraining](#a-source-pretraining)
  * [B. Test-time Adaptation (ACCUP+EATA)](#b-testtime-adaptation-accupeata)
  * [C. Command-line Runner](#c-commandline-runner)
* [Key Hyperparameters](#key-hyperparameters)
* [Outputs & Reproducibility](#outputs--reproducibility)
* [FAQ](#faq)
* [Citation](#citation)
* [License](#license)
* [Contributing](#contributing)

---

## Overview

**ACCUP**

1. Build **multi-view** (raw/aug) features and **ensemble** them;
2. Maintain **per-class low-entropy supports** to form **uncertainty-aware prototypes**;
3. For each target batch, compare **prototype entropy** vs **ensemble entropy** and choose the **lower-entropy** side to derive pseudo labels;
4. Update with **multi-view SupCon** while only tuning BN and a few conv layers.

**EATA** (plugged into ACCUP):

* **Selection**: only update on **low-entropy & density-reasonable** samples (quantile threshold + Top-K).
* **Regularization**: Fisher-style forgetting constraint to protect source discrimination.
* **Grad clipping** stabilizes the online optimization.

---

## Highlights

* **Entropy-wise pseudo-label switch** (prototype vs ensemble).
* **Uncertainty-aware prototypes** with **Top-K low-entropy** supports.
* **EATA selection + regularization** for robustness.
* **Low-intrusion updates** (BN + a few conv layers), **online per-batch** adaptation.
* **Toggleable**: set `use_eata` to enable ACCUP+EATA, or disable for ACCUP-only baseline.

---

## Repository Layout

```
.
├─ algorithms/
│  ├─ accup.py                  # ACCUP core
│  ├─ eata_utils.py             # EATA selection & regularization (integrated)
│  └─ base_tta_algorithm.py     # TTA base utilities
├─ trainers/
│  └─ tta_trainer.py            # runner for pretrain/TTA
├─ configs/
│  ├─ data_model_configs.py     # dataset/model configs (EEG/FD/HAR)
│  └─ tta_hparams_new.py        # hyperparameters (ACCUP & EATA)
├─ dataloader/
│  ├─ dataloader.py             # raw loader
│  └─ demo_dataloader.py        # augmented loader (raw, aug[, aug2])
├─ loss/
│  └─ sup_contrast_loss.py      # SupCon
├─ pre_train_model/
│  └─ pre_train_model.py        # source pretrain wrapper
├─ results/
│  └─ *.csv / logs / plots
└─ README.md / requirements.txt / LICENSE / ...
```

> Names may slightly differ in your repo—adjust imports accordingly.

---

## Environment & Installation

* Python ≥ 3.9
* PyTorch ≥ 1.12 (CUDA-matched; CPU works)
* numpy / scipy / scikit-learn / torchvision / pandas

```bash
pip install -r requirements.txt
# Or:
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn pandas
```

---

## Data Preparation

Expected layout:

```
data/<DATASET>/<SCENARIO>/
  ├─ train_<domain_id>.pt
  └─ test_<domain_id>.pt
```

Each `.pt` includes:

* `"samples"`: normalized internally to `(N, C, L)`
* `"labels"`: used for evaluation only (TTA itself is unlabeled)

Normalization:

* `dataloader.py` z-scores **raw** per channel;
* `demo_dataloader.py` z-scores **raw and aug separately** to avoid leakage.

---

## Quick Start

### A. Source Pretraining

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

### B. Test-time Adaptation (ACCUP+EATA)

```python
import torch, torch.optim as optim
from dataloader.demo_dataloader import data_generator_demo
from algorithms.accup import ACCUP

dataset_cfg = DatasetCfg()
hp_all = HpCfg()
train_hp = hp_all.train_params
accup_hp = hp_all.alg_hparams['ACCUP']
eata_hp  = hp_all.alg_hparams['EATA']   # ★ EATA settings

trg_dl = data_generator_demo(
    data_path="data/HAR/2_10_11", domain_id="11",
    dataset_configs=dataset_cfg, hparams=train_hp, dtype="test"
)

tta = ACCUP(
    configs=dataset_cfg,
    hparams={**train_hp, **accup_hp, **eata_hp, "use_eata": True},  # ★ enable EATA
    model=pretrained_model,
    optimizer=lambda p: optim.Adam(p, lr=accup_hp['learning_rate'])
).to("cuda:0")

tta.model.train()
for (raw, aug, _), _, _ in trg_dl:
    raw, aug = raw.float().cuda(), aug.float().cuda()
    _ = tta.forward((raw, aug))
```

### C. Command-line Runner

Windows:

```bash
python -m trainers.tta_trainer ^
  --data_path .\data\Dataset ^
  --dataset HAR ^
  --save_dir .\results\tta_experiments_logs ^
  --device cpu ^
  --use_eata 1
```

Linux/macOS:

```bash
python -m trainers.tta_trainer \
  --data_path ./data/Dataset \
  --dataset HAR \
  --save_dir ./results/tta_experiments_logs \
  --device cuda:0 \
  --use_eata 1
```

---

## Key Hyperparameters

**ACCUP**

* `filter_K`: max low-entropy supports per class (`-1` = keep all; 10 is common)
* `tau`: prototype similarity temperature
* `temperature`: SupCon temperature
* `learning_rate`: TTA LR
* `steps`: TTA steps per `forward` (usually 1)
* `bn_only`: update BN only (optionally plus a few conv layers)

**EATA** (matches log fields)

* `select=True, reg=True`
* `use_quantile=True, quantile=0.9`
* `e_margin_scale=0.7`, `d_margin=0.05`
* `filter_K=10`
* `warmup_min=12`
* `safety_keep_frac=0.65`
* `grad_clip=0.1` recommended for stability

---

## Outputs & Reproducibility

* Per-scenario **CSV** in `--save_dir` (columns: `scenario, run, acc, f1_score, auroc`), with **mean/std** at bottom.
* Console logs show `Pretraining stage...`, `[GradClip] ...`, `[EATA] selected ...`, `Average current f1_scores`, etc.
* You can aggregate all CSVs into a summary table or paste your spreadsheet screenshot into the README.

---

## FAQ

* **FutureWarning from `torch.load(..., weights_only=False)`**: if files are not fully trusted, set `weights_only=True` and whitelist custom objects via `torch.serialization.add_safe_globals`. If everything is local and trusted, you can ignore it.
* **Small BN batch instability**: reduce LR, switch to BN-only updates, tighten EATA selection, and/or increase `grad_clip`.
* **Prototype memory growth**: we keep Top-K low-entropy supports; for long streams, add a sliding window/decay.

---

## Citation

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

> Replace the EATA entry with the exact reference you use.

---

## License

MIT or Apache-2.0 recommended (see `LICENSE`).

---

## Contributing

PRs welcome! Ideas: new time-series backbones (TCN/ResNet/LSTM/iTransformer), more augmentations/ensembles, visualization (t-SNE/UMAP), and dataset templates.
