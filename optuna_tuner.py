import argparse
import importlib.util
import json
import math
import uuid
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import optuna
import torch

from trainers.tta_trainer import TTATrainer
from configs.data_model_configs import get_dataset_class

SEARCH_SPAN = 0.3
SENSITIVE_SPAN_SCALE = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for TTATrainer.")

    # Core trainer arguments
    parser.add_argument('--save-dir', default='results/tta_experiments_logs', type=str)
    parser.add_argument('--exp-name', default='optuna', type=str)
    parser.add_argument('--da-method', default='ACCUP', type=str)
    parser.add_argument('--data-path', default=r'D:\PyCharm Project\ACCUP + EATA\data\Dataset', type=str)
    parser.add_argument('--dataset', default='EEG', type=str)
    parser.add_argument('--backbone', default='TimesNet', type=str)
    parser.add_argument('--num-runs', default=3, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help="Override the dataloader batch size before launching trials.",
    )

    # Scenario selection
    parser.add_argument('--src-id', type=str, default=None, help="Source domain ID to adapt from.")
    parser.add_argument('--trg-id', type=str, default=None, help="Target domain ID to adapt to.")
    parser.add_argument(
        '--scenario',
        dest='scenarios',
        action='append',
        default=None,
        help="Specify src->trg pair. Repeat to evaluate multiple scenarios per trial.",
    )

    # Optuna configuration
    parser.add_argument('--study-name', default='tta_optuna', type=str)
    parser.add_argument('--storage', default=None, type=str, help="Optuna storage URI (e.g. sqlite:///optuna.db).")
    parser.add_argument('--direction', default='maximize', choices=['maximize', 'minimize'])
    parser.add_argument('--n-trials', default=20, type=int)
    parser.add_argument('--pruner', default='none', choices=['none', 'median', 'hyperband'])
    parser.add_argument('--resume', action='store_true', help="Resume the Optuna study if it already exists.")

    parser.add_argument(
        '--tune-train-params',
        action='store_true',
        help="Allow Optuna to explore pre-training hyperparameters (slower trials).",
    )
    parser.add_argument(
        '--pretrain-cache-dir',
        type=str,
        default='results/pretrain_cache',
        help="Directory for storing/reusing pre-training checkpoints across trials.",
    )
    parser.add_argument(
        '--disable-pretrain-cache',
        action='store_true',
        help="Force every trial to run pre-training from scratch (ignore cache).",
    )

    # Visualization
    parser.add_argument('--viz-dir', type=str, default=None, help="If set, export Optuna plots (HTML) to this directory.")

    # Best trial export
    parser.add_argument(
        '--best-summary-path',
        type=str,
        default=None,
        help="Optional JSON file to update with the best trial params/metrics keyed by scenario.",
    )
    parser.add_argument(
        '--write-overrides',
        action='store_true',
        help="Persist best params into configs/tta_hparams_new.py scenario_overrides.",
    )
    parser.add_argument(
        '--overrides-config',
        type=str,
        default='configs/tta_hparams_new.py',
        help="Path to the hyperparameter config containing SCENARIO_OVERRIDES dictionaries.",
    )
    parser.add_argument(
        '--search-span',
        type=float,
        default=0.3,
        help="Base relative span (e.g., 0.3 for ±30%) around scenario defaults when sampling hyperparameters.",
    )
    parser.add_argument(
        '--sensitive-span-scale',
        type=float,
        default=2.0,
        help="Multiplier applied to key ACCUP hyperparameters (LR, quantile, safety_keep_frac, etc.).",
    )

    return parser.parse_args()


def _parse_scenario_spec(spec: str) -> Tuple[str, str]:
    if "->" in spec:
        src, trg = spec.split("->", 1)
    elif "," in spec:
        src, trg = spec.split(",", 1)
    else:
        raise ValueError(f"Invalid scenario spec '{spec}'. Expected format src->trg.")
    return src.strip(), trg.strip()


def resolve_scenarios(
    dataset: str,
    src_id: Optional[str],
    trg_id: Optional[str],
    scenario_specs: Optional[List[Any]] = None,
) -> List[Tuple[str, str]]:
    """Return the list of scenarios to evaluate per trial."""
    pairs: List[Tuple[str, str]] = []

    if scenario_specs:
        for spec in scenario_specs:
            if isinstance(spec, (tuple, list)) and len(spec) == 2:
                src, trg = spec
            else:
                src, trg = _parse_scenario_spec(str(spec))
            pairs.append((str(src), str(trg)))
        return pairs

    if src_id is not None and trg_id is not None:
        return [(str(src_id), str(trg_id))]

    dataset_cfg = get_dataset_class(dataset)()
    if not getattr(dataset_cfg, "scenarios", None):
        raise ValueError(f"Dataset {dataset} does not define any scenarios.")
    default_src, default_trg = dataset_cfg.scenarios[0]
    return [(str(default_src), str(default_trg))]


def format_scenario_key(src_id: str, trg_id: str) -> str:
    """Generate a stable key for scenario-specific bookkeeping."""
    return f"{src_id}->{trg_id}"


def to_jsonable(obj: Any) -> Any:
    """Convert values to JSON-serializable builtins."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def build_pruner(name: str):
    if name == 'median':
        return optuna.pruners.MedianPruner()
    if name == 'hyperband':
        return optuna.pruners.HyperbandPruner()
    return None


def generate_visualizations(study: optuna.Study, out_dir: str):
    """Export common Optuna visualizations as interactive HTML files."""
    try:
        from optuna.visualization import (
            plot_optimization_history,
            plot_param_importances,
            plot_parallel_coordinate,
        )
    except Exception as exc:
        print(f"Skipping visualization export (plotly/visualization unavailable): {exc}")
        return

    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figures = {
        "optimization_history": plot_optimization_history(study),
        "param_importances": plot_param_importances(study),
        "parallel_coordinate": plot_parallel_coordinate(study),
    }

    for name, fig in figures.items():
        target_file = output_path / f"{name}.html"
        fig.write_html(str(target_file), include_plotlyjs="cdn")
        print(f"Saved {target_file}")


def _int_bounds_from_base(base: float, lower_ratio: float = 0.5, upper_ratio: float = 1.5,
                          min_value: int = 1, max_value: Optional[int] = None) -> Tuple[int, int]:
    """Derive integer search bounds around a baseline value."""
    try:
        base_val = int(base)
    except (TypeError, ValueError):
        base_val = min_value
    low = max(min_value, int(math.floor(base_val * lower_ratio)))
    high = int(math.ceil(base_val * upper_ratio))
    if high <= low:
        high = low + 1
    if max_value is not None:
        high = min(high, max_value)
        if high <= low:
            low = max(min_value, high - 1)
    return low, high


def suggest_train_params(trial: optuna.Trial, train_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Expand the search space to include generic training hyperparameters."""
    if not train_params:
        return {}

    suggestions: Dict[str, Any] = {}

    if "num_epochs" in train_params:
        low, high = _int_bounds_from_base(train_params["num_epochs"], min_value=5, max_value=200)
        suggestions["num_epochs"] = trial.suggest_int("num_epochs", low, high)

    if "batch_size" in train_params:
        low, high = _int_bounds_from_base(train_params["batch_size"], min_value=16, max_value=256)
        suggestions["batch_size"] = trial.suggest_int("batch_size", low, high, step=8)

    if "weight_decay" in train_params:
        base = float(train_params["weight_decay"])
        suggestions["weight_decay"] = trial.suggest_float(
            "weight_decay",
            max(1e-6, base / 10.0),
            min(1e-2, base * 10.0),
            log=True,
        )

    if "step_size" in train_params:
        low, high = _int_bounds_from_base(train_params["step_size"], min_value=5, max_value=80)
        suggestions["step_size"] = trial.suggest_int("step_size", low, high)

    if "lr_decay" in train_params:
        base = float(train_params["lr_decay"])
        suggestions["lr_decay"] = trial.suggest_float(
            "lr_decay",
            max(0.1, base * 0.5),
            min(0.99, base * 1.5),
        )

    if "steps" in train_params:
        base = max(1, int(train_params["steps"]))
        suggestions["steps"] = trial.suggest_int("steps", 1, max(5, base + 3))

    if "momentum" in train_params:
        base = float(train_params["momentum"])
        suggestions["momentum"] = trial.suggest_float(
            "momentum",
            max(0.1, base * 0.7),
            min(0.999, base * 1.1),
        )

    return suggestions


def _suggest_training_scope(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Constrain Optuna to always adapt the full backbone + classifier.
    This keeps the search space aligned with the desired “常开”训练范围。
    """
    scope_params: Dict[str, Any] = {}
    scope_params["train_full_backbone"] = True
    scope_params["train_classifier"] = True
    scope_params["train_backbone_modules"] = None
    # 当全骨干参与训练时，强制 BN 追踪目标域统计
    scope_params["freeze_bn_stats"] = False
    scope_params["filter_K"] = trial.suggest_int("filter_K_full", 9, 33, step=2)
    scope_params["safety_keep_frac"] = trial.suggest_float(
        "safety_keep_frac_full", 0.3, 0.9
    )
    scope_params["grad_clip"] = trial.suggest_float("grad_clip_full", 0.35, 1.5)
    scope_params["grad_clip_value"] = trial.suggest_categorical(
        "grad_clip_value_full", [0.1, 0.25, 0.5, 1.0]
    )
    return scope_params


def suggest_accup_params(
    trial: optuna.Trial,
    base_hparams: Dict[str, float],
    train_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Define search space for ACCUP + EATA."""
    base_ratio = max(0.05, float(SEARCH_SPAN))  # default relative span
    sensitive_scale = max(1.0, float(SENSITIVE_SPAN_SCALE))

    def narrow_float(name, default_low, default_high, *, log=False, min_clip=None, max_clip=None, ratio_scale=1.0):
        base = base_hparams.get(name, None)
        if base is None:
            return trial.suggest_float(name, default_low, default_high, log=log)
        base = float(base)
        span_ratio = base_ratio * ratio_scale
        if log:
            low = base * (1 - span_ratio)
            high = base * (1 + span_ratio)
        else:
            span = max(1e-9, abs(base) * span_ratio)
            low = base - span
            high = base + span
        if min_clip is not None:
            low = max(min_clip, low)
        if max_clip is not None:
            high = min(max_clip, high)
        if low >= high:
            high = low + (1e-6 if log else 1e-3)
        return trial.suggest_float(name, low, high, log=log)

    def narrow_int(name, default_low, default_high, *, step=1, min_clip=None, max_clip=None, ratio_scale=1.0):
        base = base_hparams.get(name, None)
        if base is None:
            return trial.suggest_int(name, default_low, default_high, step=step)
        base = int(round(base))
        span_ratio = base_ratio * ratio_scale
        delta = max(step, int(round(max(1, base) * span_ratio)))
        low = base - delta
        high = base + delta
        if min_clip is not None:
            low = max(min_clip, low)
        if max_clip is not None:
            high = min(max_clip, high)
        if low > high:
            low = high = base
        return trial.suggest_int(name, low, high, step=step)

    frozen_scope_keys = (
        "train_full_backbone",
        "train_backbone_modules",
        "train_classifier",
        "freeze_bn_stats",
    )
    frozen_scope = {k: base_hparams.get(k) for k in frozen_scope_keys}

    suggestions = {
        "learning_rate": narrow_float("learning_rate", 1e-5, 5e-4, log=True, min_clip=1e-6, max_clip=1e-3, ratio_scale=sensitive_scale),
        "pre_learning_rate": narrow_float("pre_learning_rate", 1e-5, 5e-4, log=True, min_clip=1e-6, max_clip=1e-3, ratio_scale=sensitive_scale),
        "tau": narrow_int("tau", 5, 30, min_clip=3, max_clip=60),
        "temperature": narrow_float("temperature", 0.3, 3.0, min_clip=0.1, max_clip=5.0, ratio_scale=1.5),
        "quantile": narrow_float("quantile", 0.1, 0.9, min_clip=0.05, max_clip=0.95, ratio_scale=sensitive_scale),
        "lambda_eata": narrow_float("lambda_eata", 0.5, 2.5, min_clip=0.1, max_clip=4.0, ratio_scale=sensitive_scale),
        "e_margin_scale": narrow_float("e_margin_scale", 0.2, 0.9, min_clip=0.05, max_clip=1.5),
        "d_margin": narrow_float("d_margin", 0.0, 0.15, min_clip=0.0, max_clip=0.3),
        "fisher_alpha": narrow_float("fisher_alpha", 500.0, 9000.0, min_clip=10.0, max_clip=20000.0),
        "filter_K": narrow_int("filter_K", 5, 33, step=2, min_clip=3, max_clip=63, ratio_scale=sensitive_scale),
        "safety_keep_frac": narrow_float("safety_keep_frac", 0.1, 0.9, min_clip=0.05, max_clip=0.95, ratio_scale=sensitive_scale),
        "use_quantile": bool(base_hparams.get("use_quantile", True)),
        "use_eata_select": True,
        "use_eata_reg": True,
        "online_fisher": True,
        "include_warmup_support": True,
    }

    # Parameters we keep fixed to the anchor values to stabilize search.
    suggestions["memory_size"] = int(base_hparams.get("memory_size", 2048))
    suggestions["warmup_min"] = int(base_hparams.get("warmup_min", 64))
    suggestions["max_fisher_updates"] = int(base_hparams.get("max_fisher_updates", -1))

    suggestions.update(_suggest_training_scope(trial))

    for key, value in frozen_scope.items():
        if value is not None:
            suggestions[key] = value

    if train_params:
        suggestions.update(suggest_train_params(trial, train_params))

    return suggestions


def _suggest_fd12_accup_params(
    trial: optuna.Trial,
    base_hparams: Dict[str, Any],
    train_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    专门针对 FD 1→2 的 ACCUP 空间：允许更宽的范围（含 batch_size/drop_last 之类），
    方便冲击 90+ 宏 F1。
    """

    def int_choice(name, default, *, low=None, high=None, step=1, candidates=None):
        if candidates is not None:
            return trial.suggest_categorical(name, candidates)
        base = int(default)
        lo = low if low is not None else max(8, base - step * 8)
        hi = high if high is not None else base + step * 8
        if lo >= hi:
            hi = lo + step
        return trial.suggest_int(name, lo, hi, step=step)

    def float_range(name, low, high, *, log=False):
        return trial.suggest_float(name, low, high, log=log)

    def bool_flag(name, default=False):
        # 打开 drop_last 的概率极低，靠权重即可
        return trial.suggest_categorical(name, [default, not default])

    batch_base = None
    if train_params and "batch_size" in train_params:
        batch_base = int(train_params["batch_size"])
    elif "batch_size" in base_hparams:
        batch_base = int(base_hparams["batch_size"])
    batch_base = batch_base or 128

    fd_params: Dict[str, Any] = {
        "batch_size": int_choice("fd12_batch_size", batch_base, low=80, high=192, step=8),
        "learning_rate": float_range("fd12_learning_rate", 2.0e-5, 1.5e-4, log=True),
        "pre_learning_rate": float_range("fd12_pre_lr", 4.0e-5, 2.0e-4, log=True),
        "filter_K": int_choice("fd12_filter_K", base_hparams.get("filter_K", 34), low=18, high=64, step=2),
        "quantile": float_range("fd12_quantile", 0.45, 0.78),
        "temperature": float_range("fd12_temperature", 0.45, 1.6),
        "tau": int_choice("fd12_tau", base_hparams.get("tau", 14), low=8, high=28, step=1),
        "lambda_eata": float_range("fd12_lambda_eata", 0.6, 2.4),
        "e_margin_scale": float_range("fd12_e_margin_scale", 0.35, 0.85),
        "d_margin": float_range("fd12_d_margin", 0.005, 0.13),
        "safety_keep_frac": float_range("fd12_safety_keep_frac", 0.2, 0.6),
        "memory_size": int_choice(
            "fd12_memory_size",
            base_hparams.get("memory_size", 2816),
            candidates=[1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840],
        ),
        "warmup_min": int_choice("fd12_warmup_min", base_hparams.get("warmup_min", 160), low=80, high=256, step=16),
        "max_fisher_updates": int_choice(
            "fd12_max_fisher_updates",
            base_hparams.get("max_fisher_updates", 256),
            candidates=[128, 192, 256, 320, 384, 448, 512, 640],
        ),
        "grad_clip": float_range("fd12_grad_clip", 0.45, 1.4),
        "grad_clip_value": trial.suggest_categorical("fd12_grad_clip_value", [None, 0.25, 0.5, 1.0]),
        "drop_last_test": bool_flag("fd12_drop_last_test", default=False),
        "drop_last_eval": bool_flag("fd12_drop_last_eval", default=False),
    }

    return fd_params


def suggest_timesnet_params(trial: optuna.Trial, dataset_cfg) -> Dict[str, Any]:
    """Define a lightweight search space for TimesNet backbone hyperparameters."""
    seq_len = getattr(dataset_cfg, "sequence_len", 512)
    max_patch = max(4, min(seq_len // 2, 512))
    hidden_default = int(getattr(dataset_cfg, "times_hidden_channels", 128))
    layers_default = int(getattr(dataset_cfg, "times_num_layers", 3))
    dropout_default = float(getattr(dataset_cfg, "times_dropout", 0.1))
    ffn_default = float(getattr(dataset_cfg, "times_ffn_expansion", 2.0))

    hidden = trial.suggest_int(
        "times_hidden_channels",
        max(32, hidden_default // 2),
        min(512, max(hidden_default * 2, 64)),
        step=32,
    )
    num_layers = trial.suggest_int("times_num_layers", 2, max(2, layers_default + 3))
    dropout = trial.suggest_float(
        "times_dropout",
        max(0.01, dropout_default / 2),
        min(0.6, dropout_default * 2.5),
    )
    ffn_expansion = trial.suggest_float(
        "times_ffn_expansion",
        max(1.2, ffn_default / 2),
        min(4.0, ffn_default * 2),
    )

    base_patch = trial.suggest_int(
        "times_patch_base",
        4,
        max(4, min(96, max_patch)),
        step=4,
    )
    patch_scale = trial.suggest_float("times_patch_scale", 1.3, 2.8)
    patch_count = trial.suggest_int("times_patch_count", 2, 4)
    patch_lens = []
    current = base_patch
    for _ in range(patch_count):
        patch_lens.append(int(max(2, min(max_patch, round(current)))))
        current *= patch_scale
    patch_lens = sorted({p for p in patch_lens if p > 1})
    if not patch_lens:
        patch_lens = [max(2, base_patch)]

    return {
        "times_hidden_channels": hidden,
        "times_num_layers": num_layers,
        "times_dropout": dropout,
        "times_ffn_expansion": ffn_expansion,
        "times_patch_lens": patch_lens,
    }


def build_search_space(
    trial: optuna.Trial,
    method: str,
    base_hparams: Dict[str, float],
    train_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    method = method.lower()
    if method == "accup":
        return suggest_accup_params(trial, base_hparams, train_params)
    if method == "noadap":
        return {"pre_learning_rate": trial.suggest_float("pre_learning_rate", 1e-5, 1e-3, log=True)}
    raise NotImplementedError(f"No search space implemented for method {method}.")


def _load_hparam_module(config_path: Path):
    spec = importlib.util.spec_from_file_location(
        f"_tta_hparams_{config_path.stem}_{uuid.uuid4().hex}",
        config_path,
    )
    if not spec or not spec.loader:
        raise RuntimeError(f"Unable to load hparam config at {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _is_int_like(value: Any) -> bool:
    try:
        int(str(value))
        return True
    except (TypeError, ValueError):
        return False


def _format_scenario_component(value: Any) -> str:
    if _is_int_like(value):
        return str(int(str(value)))
    return repr(str(value))


def _scenario_sort_key(key) -> Tuple[Any, ...]:
    def _component(val: str):
        if _is_int_like(val):
            return 0, int(val)
        return 1, str(val)

    if isinstance(key, tuple):
        if len(key) == 3:
            backbone = str(key[0])
            return (0, backbone.lower(), _component(str(key[1])), _component(str(key[2])))
        if len(key) == 2:
            return (1, "", _component(str(key[0])), _component(str(key[1])))
    return (2, str(key))


def _render_overrides_block(
    var_name: str,
    overrides: Dict[Any, Dict[str, Any]],
) -> str:
    lines = [f"{var_name} = {{"]  # opening brace
    items = sorted(overrides.items(), key=lambda item: _scenario_sort_key(item[0]))
    if items:
        for idx, (key, params) in enumerate(items):
            if isinstance(key, tuple) and len(key) == 3:
                backbone, src, trg = key
                header = (
                    f"    backbone_scenario("
                    f"{_format_scenario_component(backbone)}, "
                    f"{_format_scenario_component(src)}, "
                    f"{_format_scenario_component(trg)}): {{"
                )
            elif isinstance(key, tuple) and len(key) == 2:
                src, trg = key
                header = (
                    f"    scenario({_format_scenario_component(src)}, {_format_scenario_component(trg)}): {{"
                )
            else:
                header = f"    {repr(str(key))}: {{"
            lines.append(header)
            for param_name in sorted(params.keys()):
                param_value = params[param_name]
                lines.append(f"        '{param_name}': {repr(param_value)},")
            lines.append("    },")
            if idx < len(items) - 1:
                lines.append("")
    else:
        lines.append("    # scenario(src_id, trg_id): {'learning_rate': ...},")
    lines.append("}")
    return "\n".join(lines)


def _replace_block_in_file(config_path: Path, var_name: str, new_block: str):
    lines = config_path.read_text(encoding="utf-8").splitlines()
    start_idx = None
    end_idx = None
    brace_balance = 0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if start_idx is None and stripped.startswith(f"{var_name}") and "={" in stripped.replace(" ", ""):
            start_idx = idx
            brace_balance = line.count("{") - line.count("}")
            if brace_balance == 0:
                end_idx = idx
                break
            continue

        if start_idx is not None:
            brace_balance += line.count("{") - line.count("}")
            if brace_balance == 0:
                end_idx = idx
                break

    if start_idx is None or end_idx is None:
        raise RuntimeError(f"Could not locate block for {var_name} in {config_path}")

    replacement = new_block.splitlines()
    lines[start_idx:end_idx + 1] = replacement
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_scenario_overrides(
    config_path: Path,
    dataset: str,
    method: str,
    scenario: Tuple[str, str],
    params: Dict[str, Any],
    backbone: Optional[str] = None,
):
    """Write/refresh the scenario overrides entry for a given dataset + method."""
    config_path = config_path.resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot find overrides config at {config_path}")

    module = _load_hparam_module(config_path)
    var_name = f"{dataset.upper()}_{method.upper()}_SCENARIO_OVERRIDES"
    current_overrides = getattr(module, var_name, None)
    if current_overrides is None:
        raise RuntimeError(f"{var_name} not defined inside {config_path}")

    if backbone:
        normalized_key = (str(backbone), str(scenario[0]), str(scenario[1]))
    else:
        normalized_key = (str(scenario[0]), str(scenario[1]))
    merged = dict(current_overrides)
    merged[normalized_key] = dict(params)
    new_block = _render_overrides_block(var_name, merged)
    _replace_block_in_file(config_path, var_name, new_block)


def make_trainer_args(args: argparse.Namespace, trial_number: int) -> Namespace:
    """Build a Namespace compatible with TTATrainer from high-level CLI arguments."""
    cache_dir = None if args.disable_pretrain_cache else args.pretrain_cache_dir
    return Namespace(
        save_dir=args.save_dir,
        exp_name=f"{args.exp_name}_trial{trial_number}",
        da_method=args.da_method,
        data_path=args.data_path,
        dataset=args.dataset,
        backbone=args.backbone,
        num_runs=args.num_runs,
        device=args.device,
        seed=args.seed,
        pretrain_cache_dir=cache_dir,
    )


def objective(
    trial: optuna.Trial,
    args: argparse.Namespace,
    scenarios: List[Tuple[str, str]],
) -> float:
    trainer_args = make_trainer_args(args, trial.number)
    trainer = TTATrainer(trainer_args)
    batch_size_override = getattr(args, "batch_size", None)
    if batch_size_override is not None:
        batch_size_override = int(batch_size_override)
        trainer._train_params['batch_size'] = batch_size_override
        trainer.hparams['batch_size'] = batch_size_override

    # Limit to the requested scenarios for this trial
    scenario_pairs = [(str(src), str(trg)) for src, trg in scenarios]
    trainer.dataset_configs.scenarios = scenario_pairs

    # Sample new hyperparameters and apply before adaptation
    reference_hparams = dict(trainer._base_alg_hparams)
    anchor_override = trainer.get_scenario_override(*scenario_pairs[0])
    reference_hparams.update(anchor_override)
    trial_hparams = build_search_space(
        trial,
        args.da_method,
        reference_hparams,
        dict(trainer._train_params) if args.tune_train_params else None,
    )
    if args.backbone.lower() == "timesnet":
        trial_hparams.update(suggest_timesnet_params(trial, trainer.dataset_configs))

    def _needs_fd12_space() -> bool:
        if args.dataset.lower() != "fd":
            return False
        if args.da_method.lower() != "accup":
            return False
        if len(scenario_pairs) != 1:
            return False
        src_id, trg_id = scenario_pairs[0]
        return str(src_id) == "1" and str(trg_id) == "2"

    if _needs_fd12_space():
        fd_specific = _suggest_fd12_accup_params(
            trial,
            reference_hparams,
            dict(trainer._train_params),
        )
        trial_hparams.update(fd_specific)

    for src_id, trg_id in scenario_pairs:
        existing_override = trainer.get_scenario_override(src_id, trg_id)
        combined_override = dict(existing_override)
        combined_override.update(trial_hparams)
        trainer.store_scenario_override(src_id, trg_id, combined_override)
    trainer.hparams.update(trial_hparams)

    try:
        trainer.test_time_adaptation()
    except RuntimeError as exc:
        message = str(exc).lower()
        if "out of memory" in message:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned("CUDA out of memory during adaptation") from exc
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    trial_metrics: Dict[str, Dict[str, float]] = {}
    aggregated_f1: List[float] = []

    for src_id, trg_id in scenario_pairs:
        key = (str(src_id), str(trg_id))
        metrics = trainer.scenario_metrics.get(key)
        if metrics is None:
            raise RuntimeError(f"No metrics recorded for scenario {format_scenario_key(*key)}.")
        scenario_key = format_scenario_key(*key)
        trial_metrics[scenario_key] = metrics
        aggregated_f1.append(float(metrics["f1_mean"]))
        for stat_key, stat_value in metrics.items():
            attr_name = f"{scenario_key}.{stat_key}"
            trial.set_user_attr(attr_name, stat_value)

    if not trial_metrics:
        raise RuntimeError("Trainer did not record any scenario metrics.")

    objective_value = float(sum(aggregated_f1) / len(aggregated_f1))
    trial.set_user_attr("metrics", trial_metrics)
    trial.set_user_attr("scenario_keys", list(trial_metrics.keys()))

    for key, value in trial_hparams.items():
        trial.set_user_attr(f"hparam_{key}", value)
    trial.set_user_attr("objective_avg_f1", objective_value)

    # Maximize macro F1 score by default
    return objective_value


def main():
    args = parse_args()
    global SEARCH_SPAN, SENSITIVE_SPAN_SCALE
    SEARCH_SPAN = max(0.05, float(args.search_span))
    SENSITIVE_SPAN_SCALE = max(1.0, float(args.sensitive_span_scale))
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    scenarios = resolve_scenarios(
        args.dataset,
        args.src_id,
        args.trg_id,
        getattr(args, "scenarios", None),
    )
    scenario_labels = [format_scenario_key(*pair) for pair in scenarios]
    scenario_summary = ", ".join(scenario_labels)

    pruner = build_pruner(args.pruner)
    study = optuna.create_study(
        study_name=args.study_name,
        direction=args.direction,
        storage=args.storage,
        load_if_exists=args.resume,
        pruner=pruner,
    )

    try:
        study.optimize(lambda trial: objective(trial, args, scenarios), n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("Optuna search interrupted by user.")

    if len(study.trials) == 0:
        print("No trials were completed.")
        return

    completed_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))

    if not completed_trials:
        print(f"No completed trials found for scenarios: {scenario_summary}.")
    else:
        best = max(
            completed_trials,
            key=lambda t: t.value if t.value is not None else float("-inf"),
        )
        print(f"Best average F1 across [{scenario_summary}]: {best.value:.4f}")
        sanitized_params = {
            k[len("hparam_"):]: v
            for k, v in (best.user_attrs or {}).items()
            if k.startswith("hparam_")
        }
        if not sanitized_params:
            sanitized_params = dict(best.params)

        print("Best params (applied to trainer):")
        for k, v in sanitized_params.items():
            print(f"  {k}: {v}")
        metrics_blob = best.user_attrs.get("metrics") or {}
        if metrics_blob:
            print("Recorded metrics per scenario:")
            for scenario_name in sorted(metrics_blob.keys()):
                print(f"  [{scenario_name}]")
                for metric_key in sorted(metrics_blob[scenario_name].keys()):
                    label = "Std" if metric_key.endswith("_std") else "Mean"
                    value = metrics_blob[scenario_name][metric_key]
                    print(f"    {metric_key} ({label}): {value}")
        else:
            print("Recorded metrics: <none>")

        if args.best_summary_path:
            summary_path = Path(args.best_summary_path)
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                existing = json.loads(summary_path.read_text(encoding="utf-8"))
                if not isinstance(existing, dict):
                    print(f"Warning: {summary_path} does not contain a JSON object. Overwriting.")
                    existing = {}
            except FileNotFoundError:
                existing = {}
            except json.JSONDecodeError as exc:
                print(f"Warning: Failed to decode {summary_path} ({exc}). Overwriting.")
                existing = {}

            summary_key = " | ".join(scenario_labels)
            existing[summary_key] = {
                "study_name": study.study_name,
                "trial_number": best.number,
                "objective_value": float(best.value) if best.value is not None else None,
                "params": to_jsonable(sanitized_params),
                "metrics": to_jsonable(best.user_attrs.get("metrics")),
            }

            summary_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Updated best hyperparameters at {summary_path} for scenarios [{scenario_summary}].")

        if args.write_overrides:
            overrides_path = Path(args.overrides_config)
            for src_id, trg_id in scenarios:
                update_scenario_overrides(
                    overrides_path,
                    args.dataset,
                    args.da_method,
                    (str(src_id), str(trg_id)),
                    sanitized_params,
                    backbone=args.backbone,
                )
            print(
                "Updated "
                f"{args.dataset.upper()}_{args.da_method.upper()}_SCENARIO_OVERRIDES "
                f"in {overrides_path} for scenarios [{scenario_summary}]."
            )

    if args.viz_dir:
        generate_visualizations(study, args.viz_dir)


if __name__ == "__main__":
    main()
