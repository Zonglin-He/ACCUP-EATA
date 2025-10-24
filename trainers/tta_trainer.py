import sys
sys.path.append('../ADATIME')

import os
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

# >>> NEW: 引入记忆库与选择函数
from utils.utils import EATAMemory, select_eata_indices

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
parser = argparse.ArgumentParser()


class TTATrainer(TTAAbstractTrainer):
    """
    This class contain the main training functions for our method.
    训练方法
    """
    def __init__(self, args):  # TTATrainer 初始化
        super(TTATrainer, self).__init__(args)  # 调用父类的初始化方法
        self.seed = getattr(args, "seed", 42)
        fix_randomness(self.seed)
        self.exp_log_dir = os.path.join(self.home_path, self.save_dir, self.experiment_description, f"{self.run_description}")  # 实验日志目录
        self.load_pretrained_checkpoint = os.path.join(self.home_path, self.save_dir, self.experiment_description, f"{'NoAdap'}_{'All_Trg'}")  # 预训练检查点路径
        os.makedirs(self.exp_log_dir, exist_ok=True)  # 创建实验日志目录
        self.summary_f1_scores = open(self.exp_log_dir + '/summary_f1_scores.txt', 'w')  # 用于记录汇总F1分数的文件

        # >>> NEW: 初始化 EATAMemory（可根据 hparams 配置长度）
        mem_len = int(self.hparams.get('memory_size', 4096)) if hasattr(self, 'hparams') else 4096
        self.eata_memory = EATAMemory(maxlen=mem_len, device=self.device)

    def test_time_adaptation(self):  # 测试时间适应主函数
        results_columns = ["scenario", "run", "acc", "f1_score", "auroc"]  # 初始化结果表和风险表
        table_results = pd.DataFrame(columns=results_columns)
        risks_columns = ["scenario", "run", "trg_risk"]
        table_risks = pd.DataFrame(columns=risks_columns)

        for src_id, trg_id in self.dataset_configs.scenarios:  # 遍历所有源目标域对
            self.set_scenario_hparams(src_id, trg_id)  # 每个场景刷新对应超参数
            cur_scenario_f1_ret = []
            for run_id in range(self.num_runs):  # 多次运行以计算平均性能
                self.run_id = run_id
                fix_randomness(self.seed)
                print(run_id)
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir, src_id, trg_id, run_id)
                self.pre_loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.loss_avg_meters = collections.defaultdict(lambda: AverageMeter())

                # >>> NEW: 每个场景/每次 run 重新“清空”记忆库（避免跨场景污染）
                mem_len = int(self.hparams.get('memory_size', 4096)) if hasattr(self, 'hparams') else 4096
                self.eata_memory = EATAMemory(maxlen=mem_len, device=self.device)

                if self.da_method == "NoAdap":
                    self.load_data(src_id, trg_id)
                else:
                    self.load_data_demo(src_id, trg_id, run_id)

                print('Total test datasize:', len(self.trg_whole_dl.dataset))  # 打印目标域数据集的总大小
                # 统计目标域数据集中的每个类别的样本数量
                all_labels = torch.zeros(self.dataset_configs.num_classes)
                for batch_idx, (inputs, target, _) in enumerate(self.trg_whole_dl):
                    for id in range(target.shape[0]):
                        all_labels[target[id]] += 1
                print('trg whole labels:', all_labels)

                non_adapted_model_state, pre_trained_model = self.pre_train()  # 预训练源模型
                self.save_checkpoint(self.home_path, self.scenario_log_dir, non_adapted_model_state)  # 保存预训练模型检查点

                optimizer = build_optimizer(self.hparams)  # 构建优化器
                if self.da_method == 'NoAdap':  # 如果不进行适应
                    tta_model = pre_trained_model
                    tta_model.eval()
                else:  # 否则，构建TTA模型并进行适应
                    tta_model_class = self.get_tta_model_class()
                    tta_model = tta_model_class(self.dataset_configs, self.hparams, pre_trained_model, optimizer)

                    # >>> NEW: 把记忆库“塞”到 TTA 模型里，后续选择时用
                    if hasattr(tta_model, "set_eata_memory"):
                        tta_model.set_eata_memory(self.eata_memory)
                    else:
                        # 没有 setter 就直接挂属性
                        tta_model.eata_memory = self.eata_memory

                    # （可选）也把选择函数塞进去，便于在模型里直接用
                    tta_model.select_eata_indices = select_eata_indices

                tta_model.to(self.device)  # 将模型移动到指定设备（CPU或GPU）
                pre_trained_model.eval()   # 将预训练模型设置为评估模式

                metrics = self.calculate_metrics(tta_model)  # 计算适应后模型在整个目标域数据上的指标
                cur_scenario_f1_ret.append(metrics[1])  # 记录当前场景的F1分数
                scenario = f"{src_id}_to_{trg_id}"  # 场景名称
                table_results = self.append_results_to_tables(table_results, scenario, run_id, metrics[:3])  # 结果表
                table_risks = self.append_results_to_tables(table_risks, scenario, run_id, metrics[-1])     # 风险表

            cur_avg_f1_scores, cur_std_f1_scores = 100. * np.mean(cur_scenario_f1_ret), 100. * np.std(cur_scenario_f1_ret)
            print('Average current f1_scores::', cur_avg_f1_scores, 'Std:', cur_std_f1_scores)
            print(scenario, ' : ', np.around(cur_avg_f1_scores, 2), '/', np.around(cur_std_f1_scores, 2), sep='', file=self.summary_f1_scores)

        # 将均值和方差加入表中
        table_results = self.add_mean_std_table(table_results, results_columns)
        table_risks = self.add_mean_std_table(table_risks, risks_columns)
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
    parser.add_argument('--num_runs', default=1, type=int, help='Number of consecutive run with different seeds')
    parser.add_argument('--device', default="cuda", type=str, help='cpu or cuda')

    args = parser.parse_args()  # 解析命令行参数
    trainer = TTATrainer(args)  # 创建TTATrainer实例
    trainer.test_time_adaptation()  # 运行测试时间适应
