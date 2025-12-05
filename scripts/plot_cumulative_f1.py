
import os
import sys
from pathlib import Path

import torch
import pandas as pd
from argparse import Namespace
from sklearn.metrics import f1_score
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainers.tta_trainer import TTATrainer
from optim.optimizer import build_optimizer
from utils.utils import AverageMeter
import collections

DATASETS = ["HAR", "EEG", "FD"]
SEEDS = [41, 42, 43]
DATA_PATH = ROOT / "data" / "Dataset"
USE_CACHE = True
CACHE_DIR = ROOT / "results" / "pretrain_cache"
DA_METHOD = "ACCUP"
BACKBONE = "CNN"

# 输出 CSV/PNG 放在 results/tta_experiments_logs/cumulative_f1 下
OUT_DIR = ROOT / "results" / "tta_experiments_logs" / "cumulative_f1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class _NullLogger:
    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass


def run_dataset(dataset: str, seed: int):
    # 如果数据目录不存在，跳过该数据集
    data_dir = DATA_PATH / dataset
    if not data_dir.exists():
        print(f"[Skip] Dataset path not found: {data_dir}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = Namespace(
        save_dir="results/tta_experiments_logs",
        exp_name=f"{dataset}_cumf1_seed{seed}",
        da_method=DA_METHOD,
        data_path=str(DATA_PATH),
        dataset=dataset,
        backbone=BACKBONE,
        num_runs=1,
        device=device,
        seed=seed,
        scenario=None,
        pretrain_cache_dir=CACHE_DIR if USE_CACHE else None,
        disable_pretrain_cache=not USE_CACHE,
    )
    trainer = TTATrainer(args)
    trainer.logger = _NullLogger()
    trainer.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
    # 遍历该数据集的所有场景
    for src_id, trg_id in trainer.dataset_configs.scenarios:
        trainer.set_scenario_hparams(src_id, trg_id)
        trainer._current_scenario = (str(src_id), str(trg_id))
        trainer.load_data_demo(src_id, trg_id, trainer.seed)

        # 预训练 + 初始化 TTA 模型
        _, pre_trained_model = trainer.pre_train()
        optimizer = build_optimizer(trainer.hparams)
        tta_cls = trainer.get_tta_model_class()
        tta_model = tta_cls(trainer.dataset_configs, trainer.hparams, pre_trained_model, optimizer).to(trainer.device)

        rows = []
        seen = 0
        all_preds, all_labels = [], []

        for batch in trainer.trg_whole_dl:
            data, labels, trg_idx = batch
            if isinstance(data, list):
                data = [x.float().to(trainer.device) for x in data]
            else:
                data = data.float().to(trainer.device)
            labels = labels.long()

            outputs = tta_model.forward_and_adapt(data, tta_model.model, tta_model.optimizer)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            seen += labels.size(0)
            cum_f1 = f1_score(all_labels, all_preds, average="macro")
            rows.append({"samples_seen": seen, "f1": cum_f1})

        out_csv = OUT_DIR / f"cum_f1_{dataset}_{src_id}to{trg_id}_seed{seed}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")
        # Plot and save PNG
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["samples_seen"] for r in rows], [r["f1"] for r in rows], marker="o")
        ax.set_xlabel("Number of samples seen")
        ax.set_ylabel("Cumulative F1")
        ax.set_title(f"{dataset} {src_id}->{trg_id} seed {seed}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_png = out_csv.with_suffix(".png")
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"Saved {out_png}")


def main():
    for dataset in DATASETS:
        for seed in SEEDS:
            run_dataset(dataset, seed)


if __name__ == "__main__":
    main()
