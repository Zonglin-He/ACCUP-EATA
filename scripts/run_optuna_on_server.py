"""Launch Optuna hyperparameter searches directly on the FunHPC server.

This script removes the Google Colab specific glue code by constructing the
argument namespace that `optuna_tuner.py` expects and running the study for
each (src, trg) pair. Edit the paths and configuration constants below to match
your environment before executing.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import sys

# ------------------------------------------------------------------------------
# Server-specific paths --------------------------------------------------------
# ------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure repository modules are importable when running from scripts/.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import optuna_tuner

# ------------------------------------------------------------------------------
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "Dataset"
DATA_ROOT = Path(os.environ.get("TTA_DATA_ROOT", DEFAULT_DATA_ROOT))
SAVE_ROOT = REPO_ROOT / "results" / "tta_experiments_logs"
PRETRAIN_CACHE = REPO_ROOT / "results" / "pretrain_cache"
STUDY_DB = REPO_ROOT / "optuna.db"

# ------------------------------------------------------------------------------
# Search scenarios and shared options -----------------------------------------
# ------------------------------------------------------------------------------
PAIRS: List[Dict[str, int]] = [
    {"src": 7, "trg": 13},
    {"src": 9, "trg": 18},
]

DEFAULT_N_TRIALS = 70
DEFAULT_RESUME = True

DA_METHOD = "ACCUP"
DATASET = "HAR"
BACKBONE = "CNN"
NUM_RUNS = 3
DEVICE = "cuda"
SEED = 42


def ensure_directories() -> None:
    """Make sure result folders exist before launching trials."""
    for path in (SAVE_ROOT, PRETRAIN_CACHE, STUDY_DB.parent):
        path.mkdir(parents=True, exist_ok=True)


def build_args(src: int, trg: int, *, n_trials: int, resume: bool) -> SimpleNamespace:
    """Construct the Namespace that optuna_tuner.main() expects."""
    tag = f"{DATASET.lower()}_{BACKBONE.lower()}"
    study_name = f"{tag}_s{src}_t{trg}"
    return SimpleNamespace(
        save_dir=str(SAVE_ROOT),
        exp_name=tag,
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
        n_trials=n_trials,
        pruner="none",
        resume=resume,
        tune_train_params=True,
        pretrain_cache_dir=str(PRETRAIN_CACHE),
        disable_pretrain_cache=False,
        viz_dir=None,
        best_summary_path=None,
        write_overrides=True,
        overrides_config=str(REPO_ROOT / "configs" / "tta_hparams_new.py"),
    )


def run_pair(config: Dict[str, int], *, n_trials: int, resume: bool) -> None:
    """Run Optuna for a single source/target domain pair."""
    src = int(config["src"])
    trg = int(config["trg"])
    args = build_args(src, trg, n_trials=n_trials, resume=resume)

    print(f"\n[Optuna] {args.study_name}: {src} -> {trg}")
    original_parse_args = optuna_tuner.parse_args
    try:
        optuna_tuner.parse_args = lambda: args  # type: ignore[assignment]
        optuna_tuner.main()
    finally:
        optuna_tuner.parse_args = original_parse_args


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Optuna sweeps for specific HAR scenarios.")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"Number of Optuna trials per scenario (default: {DEFAULT_N_TRIALS}).",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume the existing Optuna study (default).",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Start a fresh Optuna study, ignoring previous results.",
    )
    parser.set_defaults(resume=DEFAULT_RESUME)
    return parser.parse_args()


def main(pairs: Iterable[Dict[str, int]], *, n_trials: int, resume: bool) -> None:
    ensure_directories()
    for pair in pairs:
        run_pair(pair, n_trials=n_trials, resume=resume)


if __name__ == "__main__":
    cli_args = parse_cli_args()
    main(PAIRS, n_trials=cli_args.n_trials, resume=cli_args.resume)
