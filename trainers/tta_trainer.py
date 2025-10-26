import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

ADATIME_PATH = os.path.abspath(os.path.join(PROJECT_ROOT, 'ADATIME'))
if ADATIME_PATH not in sys.path:
    sys.path.append(ADATIME_PATH)

import pandas as pd
import torch
import collections
import argparse
import warnings
import sklearn.exceptions
from datetime import datetime
import numpy as np

from utils.utils import fix_randomness, starting_logs, AverageMeter
from trainers.tta_abstract_trainer import TTAAbstractTrainer
from optim.optimizer import build_optimizer
from utils.utils import EATAMemory, select_eata_indices

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


class TTATrainer(TTAAbstractTrainer):
    """Main training loop for our test-time adaptation methods."""

    def __init__(self, args):
        super(TTATrainer, self).__init__(args)
        self.seed = getattr(args, "seed", 42)
        fix_randomness(self.seed)
        self.exp_log_dir = os.path.join(
            self.home_path,
            self.save_dir,
            self.experiment_description,
            f"{self.run_description}",
        )
        self.load_pretrained_checkpoint = os.path.join(
            self.home_path,
            self.save_dir,
            self.experiment_description,
            "NoAdap_All_Trg",
        )
        os.makedirs(self.exp_log_dir, exist_ok=True)
        self.summary_f1_scores = open(
            os.path.join(self.exp_log_dir, 'summary_f1_scores.txt'), 'w'
        )

        # Initialize EATA memory (length is configurable via hparams)
        mem_len = int(self.hparams.get('memory_size', 4096)) if hasattr(self, 'hparams') else 4096
        self.eata_memory = EATAMemory(maxlen=mem_len, device=self.device)

    def test_time_adaptation(self):
        """Entry point for running test-time adaptation."""
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]
        table_results = pd.DataFrame(columns=results_columns)
        risks_columns = ["scenario", "run", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)

        # Reset caches so repeated calls do not leak state
        self.scenario_metrics = {}
        self.last_table_results = None
        self.last_table_risks = None

        for src_id, trg_id in self.dataset_configs.scenarios:
            self.set_scenario_hparams(src_id, trg_id)
            scenario = f"{src_id}_to_{trg_id}"
            cur_scenario_f1_ret = []
            cur_scenario_metrics = []

            for run_id in range(self.num_runs):
                self.run_id = run_id
                fix_randomness(self.seed)
                print(run_id)
                self.logger, self.scenario_log_dir = starting_logs(
                    self.dataset, self.da_method, self.exp_log_dir, src_id, trg_id, run_id
                )
                self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # Refresh memory buffer for each run to avoid cross-scenario leakage
                mem_len = int(self.hparams.get('memory_size', 4096)) if hasattr(self, 'hparams') else 4096
                self.eata_memory = EATAMemory(maxlen=mem_len, device=self.device)

                if self.da_method == "NoAdap":
                    self.load_data(src_id, trg_id)
                else:
                    self.load_data_demo(src_id, trg_id, run_id)

                print('Total test datasize:', len(self.trg_whole_dl.dataset))
                all_labels = torch.zeros(self.dataset_configs.num_classes)
                for _, (_, target, _) in enumerate(self.trg_whole_dl):
                    for idx in range(target.shape[0]):
                        all_labels[target[idx]] += 1
                print('trg whole labels:', all_labels)

                non_adapted_model_state, pre_trained_model = self.pre_train()
                self.save_checkpoint(self.home_path, self.scenario_log_dir, non_adapted_model_state)

                optimizer = build_optimizer(self.hparams)
                if self.da_method == 'NoAdap':
                    tta_model = pre_trained_model
                    tta_model.eval()
                else:
                    tta_model_class = self.get_tta_model_class()
                    tta_model = tta_model_class(self.dataset_configs, self.hparams, pre_trained_model, optimizer)

                    if hasattr(tta_model, "set_eata_memory"):
                        tta_model.set_eata_memory(self.eata_memory)
                    else:
                        tta_model.eata_memory = self.eata_memory

                    tta_model.select_eata_indices = select_eata_indices

                tta_model.to(self.device)
                pre_trained_model.eval()

                metrics = self.calculate_metrics(tta_model)
                cur_scenario_metrics.append(metrics)
                cur_scenario_f1_ret.append(metrics[1])
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics[:3])
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, metrics[-1])

            if cur_scenario_metrics:
                metrics_array = np.array(cur_scenario_metrics)
                avg_metrics = metrics_array.mean(axis=0)
                std_metrics = metrics_array.std(axis=0)
                cur_avg_f1_raw = float(avg_metrics[1])
                cur_std_f1_raw = float(std_metrics[1])
                cur_avg_f1_scores = 100.0 * cur_avg_f1_raw
                cur_std_f1_scores = 100.0 * cur_std_f1_raw
            else:
                avg_metrics = np.full(4, np.nan)
                std_metrics = np.full(4, np.nan)
                cur_avg_f1_raw = float('nan')
                cur_std_f1_raw = float('nan')
                cur_avg_f1_scores = float('nan')
                cur_std_f1_scores = float('nan')

            print('Average current f1_scores::', cur_avg_f1_scores, 'Std:', cur_std_f1_scores)
            print(
                scenario,
                ' : ',
                np.around(cur_avg_f1_scores, 2),
                '/',
                np.around(cur_std_f1_scores, 2),
                sep='',
                file=self.summary_f1_scores,
            )

            scenario_key = (str(src_id), str(trg_id))
            self.scenario_metrics[scenario_key] = {
                "acc_mean": float(avg_metrics[0]),
                "f1_mean": cur_avg_f1_raw,
                "auroc_mean": float(avg_metrics[2]),
                "trg_risk_mean": float(avg_metrics[3]),
                "acc_std": float(std_metrics[0]),
                "f1_std": cur_std_f1_raw,
                "auroc_std": float(std_metrics[2]),
                "trg_risk_std": float(std_metrics[3]),
            }

        table_results = self.add_mean_std_table(table_results, results_columns)
        table_risks = self.add_mean_std_table(table_risks, risks_columns)
        self.last_table_results = table_results
        self.last_table_risks = table_risks
        self.save_tables_to_file(table_results, datetime.now().strftime('%d_%m_%Y_%H_%M_%S') + '_results')
        self.save_tables_to_file(table_risks, datetime.now().strftime('%d_%m_%Y_%H_%M_%S') + '_risks')

        self.summary_f1_scores.close()


if __name__ == "__main__":
    # ========  Experiments Name ================
    parser.add_argument('--save_dir', default='results/tta_experiments_logs', type=str, help='Directory containing all experiments')
    parser.add_argument('--exp_name', default='All_Trg', type=str, help='experiment name')
    # ========= Select the DA methods ============
    parser.add_argument('--da_method', default='ACCUP', type=str, help='ACCUP, NoAdap')
    # ========= Select the DATASET ==============
    parser.add_argument('--data_path', default=r'E:\Dataset', type=str, help='Path containing dataset')
    parser.add_argument('--dataset', default='EEG', type=str, help='Dataset of choice: (WISDM - EEG - HAR - HHAR_SA)')
    # ========= Select the BACKBONE ==============
    parser.add_argument('--backbone', default='TimesNet', type=str, help='Backbone of choice: (CNN - RESNET18 - TCN)')
    # ========= Experiment settings ===============
    parser.add_argument('--num_runs', default=3, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')
    parser.add_argument(
        '--scenario',
        action='append',
        default=None,
        help=(
            "Optional src->trg scenario filter. "
            "Example: --scenario 7->18 --scenario 16->1. "
            "If omitted, all dataset-defined scenarios will be evaluated."
        ),
)

    args = parser.parse_args()
    trainer = TTATrainer(args)
    if args.scenario:
        selected_pairs = []
        for entry in args.scenario:
            if '->' in entry:
                src, trg = entry.split('->', 1)
            elif ',' in entry:
                src, trg = entry.split(',', 1)
            else:
                raise ValueError(f"Invalid scenario format '{entry}'. Expected 'src->trg'.")
            selected_pairs.append((src.strip(), trg.strip()))
        trainer.dataset_configs.scenarios = selected_pairs

    trainer.test_time_adaptation()
