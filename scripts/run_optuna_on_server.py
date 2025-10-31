"""Launch Optuna hyperparameter searches directly on the FunHPC server.

This script removes the Google Colab specific glue code by constructing the
argument namespace that `optuna_tuner.py` expects and running the study for
each (src, trg) pair. Edit the paths and configuration constants below to match
your environment before executing.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import sys

# ------------------------------------------------------------------------------
# Server-specific paths --------------------------------------------------------
# ------------------------------------------------------------------------------
REPO_ROOT = Path("/data/coding/accup-eata")  # Repository checkout on FunHPC

# Ensure repository modules are importable when running from scripts/.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optuna_tuner

# ------------------------------------------------------------------------------
DATA_ROOT = Path("/data/coding/accup-eata/data/Dataset")  # Folder containing EEG data
SAVE_ROOT = REPO_ROOT / "results" / "tta_experiments_logs"
PRETRAIN_CACHE = REPO_ROOT / "results" / "pretrain_cache"
STUDY_DB = REPO_ROOT / "optuna.db"

# ------------------------------------------------------------------------------
# Search scenarios and shared options -----------------------------------------
# ------------------------------------------------------------------------------
PAIRS: List[Dict[str, int]] = [
    {"src": 0, "trg": 11},
    {"src": 7, "trg": 18},
    {"src": 9, "trg": 14},
]

N_TRIALS = 40
RESUME_STUDY = True

DA_METHOD = "ACCUP"
DATASET = "EEG"
BACKBONE = "TimesNet"
NUM_RUNS = 1
DEVICE = "cuda"
SEED = 42


def ensure_directories() -> None:
    """Make sure result folders exist before launching trials."""
    for path in (SAVE_ROOT, PRETRAIN_CACHE, STUDY_DB.parent):
        path.mkdir(parents=True, exist_ok=True)


def build_args(src: int, trg: int) -> SimpleNamespace:
    """Construct the Namespace that optuna_tuner.main() expects."""
    study_name = f"tta_optuna_s{src}_t{trg}"
    return SimpleNamespace(
        save_dir=str(SAVE_ROOT),
        exp_name="optuna",
        da_method=DA_METHOD,
        data_path=str(DATA_ROOT),
        dataset=DATASET,
        backbone=BACKBONE,
        num_runs=NUM_RUNS,
        device=DEVICE,
        seed=SEED,
        src_id=str(src),
        trg_id=str(trg),
        study_name=study_name,
        storage=f"sqlite:///{STUDY_DB}",
        direction="maximize",
        n_trials=N_TRIALS,
        pruner="none",
        resume=RESUME_STUDY,
        tune_train_params=False,
        pretrain_cache_dir=str(PRETRAIN_CACHE),
        disable_pretrain_cache=False,
        viz_dir=None,
        best_summary_path=None,
        write_overrides=False,
        overrides_config="configs/tta_hparams_new.py",
    )


def run_pair(config: Dict[str, int]) -> None:
    """Run Optuna for a single source/target domain pair."""
    src = int(config["src"])
    trg = int(config["trg"])
    args = build_args(src, trg)

    print(f"\n[Optuna] {args.study_name}: {src} -> {trg}")
    original_parse_args = optuna_tuner.parse_args
    try:
        optuna_tuner.parse_args = lambda: args  # type: ignore[assignment]
        optuna_tuner.main()
    finally:
        optuna_tuner.parse_args = original_parse_args


def main(pairs: Iterable[Dict[str, int]]) -> None:
    ensure_directories()
    for pair in pairs:
        run_pair(pair)


if __name__ == "__main__":
    main(PAIRS)
