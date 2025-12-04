import glob
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 场景级 Baseline (ACCUP) F1，百分比
BASELINE_PER_SCENARIO = {
    "EEG": {"0to11": 37.92, "12to5": 64.86, "7to18": 70.80, "16to1": 62.68, "9to14": 69.98},
    "HAR": {"2to11": 99.67, "6to23": 95.25, "7to13": 99.15, "9to18": 84.74, "12to16": 67.89},
    "FD":  {"0to1": 99.64, "1to2": 89.69, "3to1": 100.0, "1to0": 88.87, "2to3": 99.78},
}

DATASET_MAP = {
    "EEG": "Sleep Stage Classification (EEG)",
    "HAR": "Human Activity Recognition (HAR)",
    "FD":  "Machine Fault Diagnosis (MFD)",
}


def plot_convergence(data_dir=None):
    root = Path(__file__).resolve().parents[1]
    data_dir = Path(data_dir) if data_dir else root / "results" / "tta_experiments_logs" / "cumulative_f1"
    out_dir = root / "results" / "tta_experiments_logs" / "convergence"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"未找到目录：{data_dir}")
        return

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams['font.family'] = 'serif'

    files = glob.glob(str(data_dir / "*.csv"))
    if not files:
        print(f"未找到 CSV 文件！请检查路径：{data_dir}")
        return

    data_list = []
    for f in files:
        parts = Path(f).stem.split("_")  # cum_f1_EEG_0to11_seed41
        if len(parts) < 5:
            continue
        dataset, scenario, seed = parts[2], parts[3], parts[4].replace("seed", "")
        df = pd.read_csv(f)
        if df["f1"].max() <= 1.0:
            df["f1"] = df["f1"] * 100.0
        df["dataset"] = dataset
        df["scenario"] = scenario
        df["seed"] = seed
        data_list.append(df)

    if not data_list:
        print("数据解析失败。")
        return

    full_df = pd.concat(data_list, ignore_index=True)

    for ds in full_df["dataset"].unique():
        ds_df = full_df[full_df["dataset"] == ds]
        scenarios = ds_df["scenario"].unique()
        palette = sns.color_palette("bright", n_colors=len(scenarios))
        color_map = dict(zip(scenarios, palette))

        plt.figure(figsize=(10, 7))
        sns.lineplot(
            data=ds_df,
            x="samples_seen",
            y="f1",
            hue="scenario",
            palette=color_map,
            linewidth=3,
            errorbar="sd",
        )

        ax = plt.gca()
        if ds in BASELINE_PER_SCENARIO:
            base_dict = BASELINE_PER_SCENARIO[ds]
            for sc in scenarios:
                if sc in base_dict:
                    base_val = base_dict[sc]
                    color = color_map[sc]
                    ax.axhline(y=base_val, color=color, linestyle="--", linewidth=1.5, alpha=0.8)
                    ax.text(
                        ds_df["samples_seen"].max(),
                        base_val,
                        f"{base_val:.1f}",
                        color=color,
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                    )

        plt.title(f"{DATASET_MAP.get(ds, ds)}: Convergence Speedup", fontsize=18, fontweight="bold", pad=15)
        plt.xlabel("Samples Seen (TTA Steps)", fontsize=16)
        plt.ylabel("Test Macro F1 (%)", fontsize=16)
        plt.legend(title="NuSTAR Scenarios", loc="lower right", bbox_to_anchor=(1, 0))
        plt.tight_layout()

        save_path = out_dir / f"convergence_v2_{ds}.pdf"
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"✅ 图表已保存: {save_path}")


if __name__ == "__main__":
    plot_convergence()
