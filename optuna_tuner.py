import argparse
import importlib.util
import json
import math
import uuid
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

import optuna

from trainers.tta_trainer import TTATrainer
from configs.data_model_configs import get_dataset_class


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for TTATrainer.")

    # Core trainer arguments
    parser.add_argument('--save-dir', default='results/tta_experiments_logs', type=str)
    parser.add_argument('--exp-name', default='optuna', type=str)
    parser.add_argument('--da-method', default='ACCUP', type=str)
    parser.add_argument('--data-path', default=r'E:\Dataset', type=str)
    parser.add_argument('--dataset', default='EEG', type=str)
    parser.add_argument('--backbone', default='TimesNet', type=str)
    parser.add_argument('--num-runs', default=1, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int)

    # Scenario selection
    parser.add_argument('--src-id', type=str, default=None, help="Source domain ID to adapt from.")
    parser.add_argument('--trg-id', type=str, default=None, help="Target domain ID to adapt to.")

    # Optuna configuration
    parser.add_argument('--study-name', default='tta_optuna', type=str)
    parser.add_argument('--storage', default=None, type=str, help="Optuna storage URI (e.g. sqlite:///optuna.db).")
    parser.add_argument('--direction', default='maximize', choices=['maximize', 'minimize'])
    parser.add_argument('--n-trials', default=20, type=int)
    parser.add_argument('--pruner', default='none', choices=['none', 'median', 'hyperband'])
    parser.add_argument('--resume', action='store_true', help="Resume the Optuna study if it already exists.")

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

    return parser.parse_args()


def resolve_scenario(dataset: str, src_id: Optional[str], trg_id: Optional[str]) -> Tuple[str, str]:
    """Fall back to the first configured scenario if user does not specify one."""
    if src_id is not None and trg_id is not None:
        return str(src_id), str(trg_id)

    dataset_cfg = get_dataset_class(dataset)()
    if not getattr(dataset_cfg, "scenarios", None):
        raise ValueError(f"Dataset {dataset} does not define any scenarios.")
    default_src, default_trg = dataset_cfg.scenarios[0]
    resolved_src = str(src_id) if src_id is not None else str(default_src)
    resolved_trg = str(trg_id) if trg_id is not None else str(default_trg)
    return resolved_src, resolved_trg


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


def suggest_accup_params(
    trial: optuna.Trial,
    base_hparams: Dict[str, float],
    train_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Define search space for ACCUP + EATA."""
    suggestions = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
        "pre_learning_rate": trial.suggest_float("pre_learning_rate", 1e-5, 5e-4, log=True),
        "filter_K": trial.suggest_int("filter_K", 5, 33, step=2),
        "tau": trial.suggest_int("tau", 5, 30),
        "temperature": trial.suggest_float("temperature", 0.3, 3.0),
        "warmup_min": trial.suggest_int("warmup_min", 8, 320),
        "quantile": trial.suggest_float("quantile", 0.1, 0.9),
        "use_quantile": trial.suggest_categorical("use_quantile", [True, False]),
        "safety_keep_frac": trial.suggest_float("safety_keep_frac", 0.05, 0.9),
        "lambda_eata": trial.suggest_float("lambda_eata", 0.5, 2.5),
        "e_margin_scale": trial.suggest_float("e_margin_scale", 0.2, 0.9),
        "d_margin": trial.suggest_float("d_margin", 0.0, 0.15),
        "memory_size": trial.suggest_int("memory_size", 512, 4096, step=256),
        "fisher_alpha": trial.suggest_float("fisher_alpha", 500.0, 9000.0),
        "use_eata_select": trial.suggest_categorical("use_eata_select", [True, False]),
        "use_eata_reg": trial.suggest_categorical("use_eata_reg", [True, False]),
        "online_fisher": trial.suggest_categorical("online_fisher", [True, False]),
        "include_warmup_support": trial.suggest_categorical("include_warmup_support", [True, False]),
        "max_fisher_updates": trial.suggest_categorical(
            "max_fisher_updates", [-1, 32, 64, 128, 256, 512, 1024]
        ),
        "train_full_backbone": trial.suggest_categorical("train_full_backbone", [True, False]),
        "train_classifier": trial.suggest_categorical("train_classifier", [True, False]),
        "freeze_bn_stats": trial.suggest_categorical("freeze_bn_stats", [True, False]),
        "grad_clip": trial.suggest_float("grad_clip", 0.1, 1.5),
        "grad_clip_value": trial.suggest_categorical(
            "grad_clip_value", [None, 0.05, 0.1, 0.25, 0.5, 1.0]
        ),
    }

    # Keep values within sensible bounds around the defaults if available.
    if "temperature" in base_hparams:
        default_temp = float(base_hparams["temperature"])
        lo = max(0.1, default_temp * 0.5)
        hi = default_temp * 2.5
        suggestions["temperature"] = trial.suggest_float("temperature", lo, hi)

    if train_params:
        suggestions.update(suggest_train_params(trial, train_params))

    return suggestions


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


def _scenario_sort_key(key: Tuple[str, str]) -> Tuple[Any, Any]:
    def _component(val: str):
        if _is_int_like(val):
            return 0, int(val)
        return 1, str(val)

    return _component(key[0]), _component(key[1])


def _render_overrides_block(
    var_name: str,
    overrides: Dict[Tuple[str, str], Dict[str, Any]],
) -> str:
    lines = [f"{var_name} = {{"]  # opening brace
    items = sorted(overrides.items(), key=lambda item: _scenario_sort_key(item[0]))
    if items:
        for idx, ((src, trg), params) in enumerate(items):
            lines.append(
                f"    scenario({_format_scenario_component(src)}, {_format_scenario_component(trg)}): {{"
            )
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

    normalized_key = (str(scenario[0]), str(scenario[1]))
    merged = dict(current_overrides)
    merged[normalized_key] = dict(params)
    new_block = _render_overrides_block(var_name, merged)
    _replace_block_in_file(config_path, var_name, new_block)


def make_trainer_args(args: argparse.Namespace, trial_number: int) -> Namespace:
    """Build a Namespace compatible with TTATrainer from high-level CLI arguments."""
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
    )


def objective(trial: optuna.Trial, args: argparse.Namespace, scenario: Tuple[str, str]) -> float:
    trainer_args = make_trainer_args(args, trial.number)
    trainer = TTATrainer(trainer_args)

    # Limit to the requested scenario
    src_id, trg_id = scenario
    trainer.dataset_configs.scenarios = [(str(src_id), str(trg_id))]

    # Sample new hyperparameters and apply before adaptation
    base_hparams = dict(trainer._base_alg_hparams)
    trial_hparams = build_search_space(
        trial,
        args.da_method,
        base_hparams,
        dict(trainer._train_params),
    )

    scenario_key = (str(src_id), str(trg_id))
    overrides = dict(trainer._scenario_hparam_overrides.get(scenario_key, {}))
    overrides.update(trial_hparams)
    trainer._scenario_hparam_overrides[scenario_key] = overrides
    trainer.hparams.update(trial_hparams)

    trainer.test_time_adaptation()

    metrics = trainer.scenario_metrics.get(scenario_key)
    if metrics is None:
        raise RuntimeError(f"No metrics recorded for scenario {scenario_key}.")

    trial.set_user_attr("metrics", metrics)
    for key, value in trial_hparams.items():
        trial.set_user_attr(f"hparam_{key}", value)
    trial.set_user_attr("scenario_key", format_scenario_key(*scenario_key))

    # Maximize macro F1 score by default
    return float(metrics["f1_mean"])


def main():
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    scenario = resolve_scenario(args.dataset, args.src_id, args.trg_id)

    pruner = build_pruner(args.pruner)
    study = optuna.create_study(
        study_name=args.study_name,
        direction=args.direction,
        storage=args.storage,
        load_if_exists=args.resume,
        pruner=pruner,
    )

    try:
        study.optimize(lambda trial: objective(trial, args, scenario), n_trials=args.n_trials)
    except KeyboardInterrupt:
        print("Optuna search interrupted by user.")

    if len(study.trials) == 0:
        print("No trials were completed.")
        return

    scenario_key = format_scenario_key(*scenario)
    completed_trials = [
        t for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if t.user_attrs.get("scenario_key") == scenario_key
    ]

    if not completed_trials:
        print(f"No completed trials found for scenario {scenario_key}.")
    else:
        best = max(
            completed_trials,
            key=lambda t: t.value if t.value is not None else float("-inf"),
        )
        print(f"Best F1 for scenario {scenario_key}: {best.value:.4f}")
        print("Best params:")
        for k, v in best.params.items():
            print(f"  {k}: {v}")
        print("Recorded metrics:")
        for k, v in (best.user_attrs.get("metrics") or {}).items():
            print(f"  {k}: {v}")

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

            existing[scenario_key] = {
                "study_name": study.study_name,
                "trial_number": best.number,
                "objective_value": float(best.value) if best.value is not None else None,
                "params": to_jsonable(best.params),
                "metrics": to_jsonable(best.user_attrs.get("metrics")),
            }

            summary_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Updated best hyperparameters at {summary_path} for scenario {scenario_key}.")

        if args.write_overrides:
            overrides_path = Path(args.overrides_config)
            update_scenario_overrides(
                overrides_path,
                args.dataset,
                args.da_method,
                scenario,
                best.params,
            )
            print(
                "Updated "
                f"{args.dataset.upper()}_{args.da_method.upper()}_SCENARIO_OVERRIDES "
                f"in {overrides_path} for scenario {scenario_key}."
            )

    if args.viz_dir:
        generate_visualizations(study, args.viz_dir)


if __name__ == "__main__":
    main()
