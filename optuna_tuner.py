import argparse
import json
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


def suggest_accup_params(trial: optuna.Trial, base_hparams: Dict[str, float]) -> Dict[str, float]:
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
    }

    # Keep values within sensible bounds around the defaults if available.
    if "temperature" in base_hparams:
        default_temp = float(base_hparams["temperature"])
        lo = max(0.1, default_temp * 0.5)
        hi = default_temp * 2.5
        suggestions["temperature"] = trial.suggest_float("temperature", lo, hi)

    return suggestions


def build_search_space(trial: optuna.Trial, method: str, base_hparams: Dict[str, float]) -> Dict[str, float]:
    method = method.lower()
    if method == "accup":
        return suggest_accup_params(trial, base_hparams)
    if method == "noadap":
        return {"pre_learning_rate": trial.suggest_float("pre_learning_rate", 1e-5, 1e-3, log=True)}
    raise NotImplementedError(f"No search space implemented for method {method}.")


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
    trial_hparams = build_search_space(trial, args.da_method, base_hparams)

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

    if args.viz_dir:
        generate_visualizations(study, args.viz_dir)


if __name__ == "__main__":
    main()
