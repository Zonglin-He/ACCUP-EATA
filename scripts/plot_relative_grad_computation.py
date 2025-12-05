import os
import sys
import collections
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import Namespace

# 项目根目录放入 sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trainers.tta_trainer import TTATrainer
from optim.optimizer import build_optimizer
from utils.utils import AverageMeter

# -----------------------------
# 可配置区域
# -----------------------------
DATASETS = ["HAR", "EEG", "FD"]
SEEDS = [41, 42, 43]
DATA_PATH = ROOT / "data" / "Dataset"
USE_CACHE = True
CACHE_DIR = ROOT / "results" / "pretrain_cache"
DA_METHOD = "ACCUP"  # 假设 NuSTAR 的实现基于 ACCUP 的选样逻辑
BACKBONE = "CNN"

OUT_DIR = ROOT / "results" / "tta_experiments_logs" / "relative_grad"
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


def run_one(dataset: str, seed: int):
    """运行一个数据集 + 种子，返回 (selected_sum, total_sum)。"""
    data_dir = DATA_PATH / dataset
    if not data_dir.exists():
        print(f"[Skip] Dataset path not found: {data_dir}")
        return 0, 0

    device = "cuda"

    args = Namespace(
        save_dir="results/tta_experiments_logs",
        exp_name=f"{dataset}_relgrad_seed{seed}",
        da_method=DA_METHOD,
        data_path=str(DATA_PATH),
        dataset=dataset,
        backbone=BACKBONE,
        num_runs=1,
        device=device,
        seed=seed,
        scenario=None,
        pretrain_cache_dir=str(CACHE_DIR) if USE_CACHE else None,
        disable_pretrain_cache=not USE_CACHE,
    )
    trainer = TTATrainer(args)
    trainer.logger = _NullLogger()
    trainer.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

    selected_sum = 0
    total_sum = 0

    # 遍历该数据集的所有场景
    for src_id, trg_id in trainer.dataset_configs.scenarios:
        trainer.set_scenario_hparams(src_id, trg_id)
        trainer._current_scenario = (str(src_id), str(trg_id))
        trainer.load_data_demo(src_id, trg_id, trainer.seed)

        _, pre_trained_model = trainer.pre_train()
        optimizer = build_optimizer(trainer.hparams)
        tta_cls = trainer.get_tta_model_class()
        tta_model = tta_cls(trainer.dataset_configs, trainer.hparams, pre_trained_model, optimizer).to(trainer.device)

        # 统计初始化
        tta_model._total_samples = len(trainer.trg_whole_dl.dataset)
        tta_model._selected_counter = 0

        # 完整跑一遍目标集
        for batch in trainer.trg_whole_dl:
            data, labels, trg_idx = batch
            if isinstance(data, list):
                data = [x.float().to(trainer.device) for x in data]
            else:
                data = data.float().to(trainer.device)
            _ = tta_model.forward_and_adapt(data, tta_model.model, tta_model.optimizer)

        selected_sum += getattr(tta_model, "_selected_counter", 0)
        total_sum += getattr(tta_model, "_total_samples", 0)

    return selected_sum, total_sum


def main():
    results = {}
    for ds in DATASETS:
        ratios = []
        for seed in SEEDS:
            sel, tot = run_one(ds, seed)
            if tot > 0:
                ratios.append(sel / tot * 100.0)
        # 取平均相对计算量
        avg_cost = sum(ratios) / len(ratios) if ratios else 0.0
        results[ds] = avg_cost

    # 绘图
    datasets = list(results.keys())
    accup_cost = [100.0] * len(datasets)
    nustar_cost = [results[ds] for ds in datasets]
    speedups = [100.0 / c if c > 0 else 0.0 for c in nustar_cost]

    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    x = range(len(datasets))
    width = 0.35
    color_accup = "0.6"
    color_nustar = "#1f77b4"

    plt.bar(
        [i - width / 2 for i in x],
        accup_cost,
        width,
        label="ACCUP (Baseline)",
        edgecolor=color_accup,
        facecolor="white",
        linewidth=2,
        hatch="//",
    )
    plt.bar(
        [i + width / 2 for i in x],
        nustar_cost,
        width,
        label="NuSTAR (Yours)",
        color=color_nustar,
    )

    for xc, cost, spd in zip(x, nustar_cost, speedups):
        plt.text(
            xc + width / 2,
            cost + 2,
            f"{spd:.1f}x",
            ha="center",
            va="bottom",
            fontsize=14,
            color="black",
        )

    plt.xticks(x, datasets, fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0, 110)
    plt.ylabel("Relative Gradient Computation (%)", fontsize=16)
    plt.xlabel("Dataset", fontsize=16)
    plt.title("Relative Gradient Computation Comparison", fontsize=18)
    plt.legend(fontsize=12)
    plt.tight_layout()

    out_pdf = OUT_DIR / "relative_grad_computation.pdf"
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    main()
