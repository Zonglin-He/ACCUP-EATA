# algorithms/accup.py
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_tta_algorithm import BaseTestTimeAlgorithm, softmax_entropy
from loss.sup_contrast_loss import domain_contrastive_loss
from utils.utils import EATAMemory, select_eata_indices, softmax_entropy_from_logits


class ACCUP(BaseTestTimeAlgorithm):
    """
    ACCUP + EATA: 在 ACCUP 原流程中缝入 EATA 的“先选后学 + 防遗忘”。
    - 保留：双视图（raw/aug）、原型选择、原型/模型熵比较、对比损失、返回 select_pred（兼容你原 calculate_metrics）
    - 新增：EATA 低熵筛选 + 概率均值去冗余（批内子集） + 熵最小化项 + Fisher/L2-SP
    - 更新范围：仍由 configure_model 决定（默认 BN 仿射 + 指定 Conv1d），EATA 只筛“样本”
    """

    def __init__(self, configs, hparams, model, optimizer):
        super(ACCUP, self).__init__(configs, hparams, model, optimizer)

        self.memory = EATAMemory(
            maxlen=int(hparams.get("memory_size", 4096)),
            device=hparams.get("device", "cpu")
        )

        self.hparams.setdefault("e_margin_scale", 0.85)
        self.hparams.setdefault("d_margin", 0.0)
        self.hparams.setdefault("filter_K", 5)
        self.hparams.setdefault("warmup_min", 128)
        self.hparams.setdefault("temperature", 0.7)
        self.hparams.setdefault("quantile", 0.70)
        self.hparams.setdefault("safety_keep_frac", 0.125)

        self.use_eata_select = bool(hparams.get("use_eata_select", True))  # 是否启用“低熵+去冗余”筛样本
        self.use_eata_reg    = bool(hparams.get("use_eata_reg", True))     # 是否启用 Fisher/L2-SP 正则

        # ---- 原 ACCUP 成员 ----
        self.featurizer = model.feature_extractor
        self.classifier = model.classifier
        self.filter_K = hparams['filter_K']
        self.tau = hparams['tau']
        self.temperature = hparams['temperature']
        self.num_classes = configs.num_classes

        warmup_supports = self.classifier.logits.weight.data.detach()
        self.warmup_supports = warmup_supports
        warmup_prob = self.classifier(self.warmup_supports)
        self.warmup_labels = F.one_hot(warmup_prob.argmax(1), num_classes=self.num_classes).float()
        self.warmup_ent = softmax_entropy(warmup_prob, warmup_prob)
        self.warmup_cls_scores = F.softmax(warmup_prob, 1)

        self.supports = self.warmup_supports.data
        self.labels = self.warmup_labels.data
        self.ents = self.warmup_ent.data
        self.cls_scores = self.warmup_cls_scores.data

        # ---- EATA 状态与超参（可在 hparams 覆盖）----
        self.e_margin = float(hparams.get("e_margin", math.log(self.num_classes) * 0.40))  # 低熵阈
        self.d_margin = float(hparams.get("d_margin", 0.05))                               # 去冗余阈
        self.fisher_alpha = float(hparams.get("fisher_alpha", 2000.0))                     # Fisher 权重
        self.lambda_eata = float(hparams.get("lambda_eata", 1.0))                          # 熵项系数

        self._eata_trainable_names = None  # 可训练参数名集合

        # Fisher 可直接给 dict 或路径；没有则回退 L2-SP
        self.fishers = hparams.get("fisher_state", None)
        if self.fishers is None and "fisher_path" in hparams and os.path.exists(hparams["fisher_path"]):
            self.fishers = torch.load(hparams["fisher_path"], map_location="cpu")

    # ----------------- ACCUP 原逻辑 -----------------
    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        raw_data, aug_data = batch_data[0], batch_data[1]

        # 前向两视图
        r_feat, r_seq_feat = model.feature_extractor(raw_data)
        r_output = model.classifier(r_feat)
        a_feat, a_seq_feat = model.feature_extractor(aug_data)
        a_output = model.classifier(a_feat)

        # 集成特征/预测
        z = (r_feat + a_feat) / 2.0
        p = (r_output + a_output) / 2.0
        yhat = F.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p, p)
        cls_scores = F.softmax(p, 1)

        # 记忆库追加（不反传）
        with torch.no_grad():
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ents = self.ents.to(z.device)
            self.cls_scores = self.cls_scores.to(z.device)
            self.supports = torch.cat([self.supports, z])
            self.labels = torch.cat([self.labels, yhat])
            self.ents = torch.cat([self.ents, ent])
            self.cls_scores = torch.cat([self.cls_scores, cls_scores])

        # 选最自信 supports
        supports, labels, _ = self.select_supports()

        # 原型头 logits + 熵比较（保留你原来的选择语义）
        prt_scores = self.compute_logits(z, supports, labels)
        prt_ent = softmax_entropy(prt_scores, prt_scores)
        idx = prt_ent < ent
        idx_un = idx.unsqueeze(1).expand(-1, prt_scores.shape[1])
        select_pred = torch.where(idx_un, prt_scores, cls_scores)  # 返回用

        # ----------------- 这里开始缝入 EATA 的“先选后学 + 正则” -----------------
        # 懒加载 θ0（用于 L2-SP）
        # 懒加载 θ0（用于 L2-SP）
        self._eata_snapshot_if_needed(model)

        # 用记忆库做“低熵 + 去冗余”筛选（基于 raw 视图的 logits/feats）
        sel, log_str = select_eata_indices(
            logits=r_output.detach(),
            feats=r_feat.detach(),
            num_classes=self.num_classes,
            memory=self.memory,
            e_margin_scale=float(self.hparams.get("e_margin_scale", 0.85)),
            d_margin=float(self.hparams.get("d_margin", 0.0)),
            K=int(self.hparams.get("filter_K", self.filter_K)),
            temperature=float(self.hparams.get("temperature", self.temperature)),
            warmup_min=int(self.hparams.get("warmup_min", 128)),
            use_quantile=True,
            quantile=float(self.hparams.get("quantile", 0.70)),
            safety_keep_frac=float(self.hparams.get("safety_keep_frac", 0.125)),
        )

        print(f"[EATA] select={self.use_eata_select}, reg={self.use_eata_reg}")
        if self.use_eata_select:
            print(f"[EATA] selected {int(sel.numel())}/{r_output.size(0)}")
            print(log_str)
        else:
            sel = torch.arange(r_output.size(0), device=r_output.device)
            print("[EATA] using full batch (no selection)")
        ent_raw = softmax_entropy_from_logits(r_output).detach()

        # 仅对“被选中样本”的三视图做对比损失 + EATA 熵项 + Fisher/L2-SP 正则
        loss = torch.zeros([], device=z.device)
        if sel.numel() > 0:
            B = r_output.size(0)
            cat_p = torch.cat([r_output, a_output, p], dim=0)      # [3B, C]
            cat_y = select_pred.max(1)[1].repeat(3)                # [3B]（用你最终选择后的伪标签）
            sel3 = torch.cat([sel, sel + B, sel + 2 * B], dim=0)   # 三视图中的选中索引

            # 对比损失（仅选中子集）
            loss_con = domain_contrastive_loss(cat_p[sel3], cat_y[sel3],
                                               temperature=self.temperature, device=z.device)

            # EATA 熵项（增广视图更稳）+ 样本重加权 coeff = exp(-(H - e_margin))
            coeff = torch.exp(-(ent_raw[sel].detach() - self.e_margin))
            loss_e = (softmax_entropy_from_logits(a_output[sel]) * coeff).mean()

            # 防遗忘
            # 防遗忘
            loss_reg = self._eata_regularizer(model) if self.use_eata_reg else torch.zeros([], device=z.device)

            loss = loss_con + self.lambda_eata * loss_e + loss_reg

        # 只有有选中样本时才更新；否则跳过更新（与 EATA 一致）
        optimizer.zero_grad(set_to_none=True)
        if sel.numel() > 0:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p_ for p_ in model.parameters() if p_.requires_grad),
                max_norm=5.0
            )
            optimizer.step()

        # 更新概率均值缓存（做去冗余）
        with torch.no_grad():
            self.memory.push(
                feats=r_feat.detach(),
                probs=torch.softmax(r_output.detach(), dim=1),
            )

        # 返回与原 ACCUP 保持一致（给上层 metrics 用）
        return select_pred

    # ----------------- ACCUP 原工具 -----------------
    def get_topk_neighbor(self, feature, supports, cls_scores, k_neighbor):
        feature = F.normalize(feature, dim=1)
        supports = F.normalize(supports, dim=1)
        sim_matrix = feature @ supports.T
        _, idx_near = torch.topk(sim_matrix, k_neighbor, dim=1)
        cls_score_near = cls_scores[idx_near].detach().clone()
        return cls_score_near

    def compute_logits(self, z, supports, labels):
        B, dim = z.size()
        N, dim_ = supports.size()
        assert (dim == dim_)
        temp_centroids = (labels / (labels.sum(dim=0, keepdim=True) + 1e-12)).T @ supports
        temp_z = F.normalize(z, dim=1)
        temp_centroids = F.normalize(temp_centroids, dim=1)
        logits = self.tau * temp_z @ temp_centroids.T
        return logits

    def select_supports(self):
        ent_s = self.ents
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))
        else:
            indices = []
            indices1 = torch.LongTensor(list(range(len(ent_s))))
            for i in range(self.num_classes):
                _, indices2 = torch.sort(ent_s[y_hat == i])
                indices.append(indices1[y_hat == i][indices2][:filter_K])
            indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ents = self.ents[indices]
        self.cls_scores = self.cls_scores[indices]
        return self.supports, self.labels, indices

    def configure_model(self, model):
        """
        与原 ACCUP 一致：训练模式；冻结全网；
        BN 用 batch 统计并允许仿射更新；Conv1d 的三个 block 打开训练。
        （EATA 的正则会只针对 requires_grad=True 的这些参数）
        """
        model.train()
        model.requires_grad_(False)

        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)

        for name, module in model.feature_extractor.named_children():
            if name in ('conv_block1', 'conv_block2', 'conv_block3'):
                for sub_module in module.children():
                    if isinstance(sub_module, nn.Conv1d):
                        sub_module.requires_grad_(True)
        return model

    # ----------------- EATA 小工具 -----------------
    @staticmethod
    def _softmax_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        return -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=1)

    @staticmethod
    def _update_probs_momentum(current, new_probs, m=0.9):
        if new_probs.numel() == 0:
            return current
        mean_new = new_probs.mean(dim=0)
        if current is None:
            return mean_new
        return m * current + (1.0 - m) * mean_new

    def _eata_snapshot_if_needed(self, model):
        """第一次调用时，记录可训练子集的 θ0（用于 L2-SP）。"""
        # 用 getattr 防止 AttributeError
        theta0 = getattr(self, "_eata_theta0", None)
        if theta0 is None:
            self._eata_trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
            # 记录可训练参数的初始快照
            self._eata_theta0 = {
                n: p.detach().clone()
                for n, p in model.named_parameters()
                if n in self._eata_trainable_names
            }

    def _eata_regularizer(self, model):
        """Fisher（优先）或 L2-SP（兜底），只作用于 requires_grad=True 的参数。"""
        device = next(model.parameters()).device
        reg = None
        theta0 = getattr(self, "_eata_theta0", None)

        # Fisher 优先
        if isinstance(self.fishers, dict) and len(self.fishers) > 0:
            for n, p in model.named_parameters():
                if p.requires_grad and (n in self.fishers):
                    item = self.fishers[n]
                    if isinstance(item, (list, tuple)):
                        diag, theta_prev = item[0].to(device), item[1].to(device)
                    else:
                        if theta0 is None or (n not in theta0):
                            continue
                        diag, theta_prev = item.to(device), theta0[n].to(device)
                    term = (diag * (p - theta_prev) ** 2).sum()
                    reg = term if reg is None else (reg + term)
            if reg is not None:
                return self.fisher_alpha * reg

        # L2-SP 兜底
        if theta0 is None:
            return torch.zeros([], device=device)

        for n, p in model.named_parameters():
            if p.requires_grad and (n in theta0):
                term = ((p - theta0[n].to(device)) ** 2).sum()
                reg = term if reg is None else (reg + term)
        return reg if reg is not None else torch.zeros([], device=device)

