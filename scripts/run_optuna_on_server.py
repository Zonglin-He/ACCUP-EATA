# /data/coding/accup-eata/scripts/run_optuna_on_server.py
from __future__ import annotations
from pathlib import Path
import subprocess

# 仓库根目录 & 数据根目录（你已有 data/Dataset 的软链接）
REPO_ROOT       = Path("/data/coding/accup-eata")
DATA_ROOT       = REPO_ROOT / "data" / "Dataset"     # 不要再细到 EEG，--dataset 会指明
SAVE_ROOT       = REPO_ROOT / "results" / "tta_experiments_logs"
PRETRAIN_CACHE  = REPO_ROOT / "results" / "pretrain_cache"
STUDY_DB        = REPO_ROOT / "optuna.db"

# 要跑的 src->trg 组合（按需改）
PAIRS = [
    {"src": 0, "trg": 11},
    {"src": 7, "trg": 18},
    {"src": 9, "trg": 14},
]

# 共享的 Optuna 参数（按需改）
N_TRIALS     = 40
RESUME_STUDY = True
EXTRA_ARGS = [
    "--da-method", "ACCUP",
    "--dataset", "EEG",
    "--backbone", "TimesNet",
    "--num-runs", "1",
    "--device", "cuda",
]

# 使用你项目里的 venv
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"
if not PYTHON_BIN.exists():
    PYTHON_BIN = Path("python")   # 兜底

def ensure_dirs():
    for p in (SAVE_ROOT, PRETRAIN_CACHE, STUDY_DB.parent):
        p.mkdir(parents=True, exist_ok=True)

def run_pair(src: int, trg: int):
    study_name = f"tta_optuna_s{src}_t{trg}"
    cmd = [
        str(PYTHON_BIN),
        "optuna_tuner.py",
        # 注意：下面两个参数名要和 optuna_tuner.py 的 argparse 完全一致
        "--data-path",        str(DATA_ROOT),         # 如果脚本用 --data-path，请把下划线改成连字符
        "--save-dir",         str(SAVE_ROOT),         # 同理：--save-dir / --save_dir 取决于你的脚本
        "--pretrain-cache-dir", str(PRETRAIN_CACHE),
        "--study-name",       study_name,
        "--storage",          f"sqlite:///{STUDY_DB}",
        "--src-id",           str(src),
        "--trg-id",           str(trg),
        "--n-trials",         str(N_TRIALS),
    ] + EXTRA_ARGS
    if RESUME_STUDY:
        cmd.append("--resume")

    print(f"\n[Optuna] {study_name}: {src} -> {trg}")
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)

def main():
    ensure_dirs()
    for pair in PAIRS:
        run_pair(int(pair["src"]), int(pair["trg"]))

if __name__ == "__main__":
    main()
