# algorithms/eata_accup.py
import os
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_tta_algorithm import BaseTestTimeAlgorithm  # 路径与 ACCUP 一致


def _softmax_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=1)
    return -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=1)


def _update_probs_momentum(current: Optional[torch.Tensor],
                           new_probs: torch.Tensor,
                           momentum: float = 0.9) -> Optional[torch.Tensor]:
    if new_probs.numel() == 0:
        return current
    mean_new = new_probs.mean(dim=0)
    if current is None:
        return mean_new
    return momentum * current + (1.0 - momentum) * mean_new


class EATA(BaseTestTimeAlgorithm):
    """
    EATA(ICML'22) — 最小可用实现，接口对齐 ACCUP：
    - forward_and_adapt(batch_data, model, optimizer) → [B, C] 概率
    - 仅更新 classifier/adapter（可在 hparams 里改关键字）
    - 低熵筛选 + 与概率均值的余弦去冗余 + 重加权熵最小化
    - Fisher 抗遗忘（若未提供则自动回退 L2-SP）
    """
    def __init__(self, configs, hparams, model, optimizer):
        super(EATA, self).__init__(configs, hparams, model, optimizer)

        self.featurizer = model.feature_extractor
        self.classifier = model.classifier
        self.num_classes = configs.num_classes

        # —— 超参（hparams 可覆盖）
        self.e_margin = float(hparams.get("e_margin", math.log(self.num_classes) * 0.40))
        self.d_margin = float(hparams.get("d_margin", 0.05))
        self.fisher_alpha = float(hparams.get("fisher_alpha", 2000.0))
        self.grad_clip = float(hparams.get("grad_clip", 5.0))
        self.adapt_keywords = tuple(hparams.get("adapt_keywords", ("classifier", "adapter")))

        # —— 概率均值缓存（做“去冗余”）
        self.current_model_probs: Optional[torch.Tensor] = None

        # —— 统计量（仅做日志用）
        self.num_samples_update_1 = 0  # 低熵后
        self.num_samples_update_2 = 0  # 低熵+去冗余后

        # —— 只解冻头部参数；并建立 θ0（L2-SP 的锚点）
        self._freeze_all_and_unfreeze_keywords(self.model, self.adapt_keywords)
        self.theta0 = {n: p.detach().clone() for n, p in self._iter_trainable_named_params(self.model)}

        # —— Fisher：优先从 hparams["fisher_state"] 或 hparams["fisher_path"] 读取
        self.fishers: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None
        if "fisher_state" in hparams and isinstance(hparams["fisher_state"], dict):
            self.fishers = {}
            for k, v in hparams["fisher_state"].items():
                diag, theta = (v[0], v[1]) if isinstance(v, (list, tuple)) else (v, self.theta0.get(k, None))
                if theta is None:
                    continue
                self.fishers[k] = (diag.detach().clone(), theta.detach().clone())
        elif "fisher_path" in hparams and isinstance(hparams["fisher_path"], str) and os.path.exists(hparams["fisher_path"]):
            raw = torch.load(hparams["fisher_path"], map_location="cpu")
            self.fishers = {}
            for k, v in raw.items():
                if isinstance(v, (list, tuple)):
                    diag, theta = v[0], v[1]
                else:
                    diag, theta = v, self.theta0.get(k, None)
                if theta is None:
                    continue
                self.fishers[k] = (diag.detach().clone(), theta.detach().clone())

    # ============== 必要接口：配置可训练参数（LN-only 也能用） ==============
    def configure_model(self, model):
        model.train()
        model.requires_grad_(False)
        # 若模型含 BN：使用 batch 统计，不更新 running stats
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        # 仅解冻关键词匹配的参数（默认 classifier/adapter）
        for name, p in model.named_parameters():
            if any(k in name for k in self.adapt_keywords):
                p.requires_grad_(True)
        return model

    # ========================= TTA 主流程 =========================
    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        raw_data, aug_data = batch_data[0], batch_data[1]

        # 1) 前向：两视图
        r_feat, _ = model.feature_extractor(raw_data)
        r_logits = model.classifier(r_feat)        # 用它的熵筛“可靠样本”
        a_feat, _ = model.feature_extractor(aug_data)
        a_logits = model.classifier(a_feat)        # 用它参与反传更稳
        probs_raw = r_logits.softmax(dim=1)

        # 2) 低熵筛选（可靠）
        ent = _softmax_entropy_from_logits(r_logits)     # [B]
        ids1 = torch.where(ent < self.e_margin)[0]       # 可靠样本索引
        ent_sel = ent[ids1]

        # 3) 去冗余：与“概率均值”的余弦 < d_margin
        ids2 = torch.arange(ids1.numel(), device=ids1.device)
        if self.current_model_probs is not None and ids1.numel() > 0:
            cos = F.cosine_similarity(self.current_model_probs.unsqueeze(0),
                                      probs_raw[ids1], dim=1).abs()
            ids2 = torch.where(cos < self.d_margin)[0]
            ent_sel = ent_sel[ids2]

        # 4) 重加权熵最小化：coeff = exp(-(H - e_margin))
        if ent_sel.numel() > 0:
            coeff = torch.exp(-(ent_sel.detach() - self.e_margin))
            loss_ent = (_softmax_entropy_from_logits(a_logits[ids1][ids2]) * coeff).mean()
        else:
            loss_ent = r_logits.new_zeros([])

        # 5) Fisher / L2-SP 正则
        loss_reg = self._regularizer(model)
        loss = loss_ent + loss_reg

        # 6) 仅当选中样本>0 时更新
        optimizer.zero_grad(set_to_none=True)
        if ids1.numel() > 0 and ids2.numel() > 0:
            loss.backward()
            if self.grad_clip and self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self._iter_trainable_params(model), self.grad_clip)
            optimizer.step()

        # 7) 更新“概率均值”与统计
        with torch.no_grad():
            picked = probs_raw[ids1][ids2] if (ids1.numel() > 0 and ids2.numel() > 0) else probs_raw.new_zeros((0, probs_raw.size(1)))
            self.current_model_probs = _update_probs_momentum(self.current_model_probs, picked)
            self.num_samples_update_1 += int(ids1.numel())
            self.num_samples_update_2 += int(ids2.numel())

        # 8) 返回概率（给 calculate_metrics 用）
        out_probs = ((r_logits + a_logits) * 0.5).softmax(dim=1)
        return out_probs

    # ========================= 内部工具 =========================
    def _iter_trainable_named_params(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                yield n, p

    def _iter_trainable_params(self, model):
        for _, p in self._iter_trainable_named_params(model):
            yield p

    def _freeze_all_and_unfreeze_keywords(self, model, keywords):
        model.train()
        model.requires_grad_(False)
        for n, p in model.named_parameters():
            if any(k in n for k in keywords):
                p.requires_grad_(True)

    def _regularizer(self, model) -> torch.Tensor:
        """Fisher 优先；无 Fisher 时回退 L2-SP 到 θ0"""
        reg = None
        if self.fishers is not None:
            for n, p in self._iter_trainable_named_params(model):
                if n in self.fishers:
                    diag, theta_prev = self.fishers[n]
                    diag = diag.to(p.device)
                    theta_prev = theta_prev.to(p.device)
                    term = (diag * (p - theta_prev) ** 2).sum()
                    reg = term if reg is None else (reg + term)
            if reg is not None:
                return reg * self.fisher_alpha

        # L2-SP：对“可训练子集”做 ||θ - θ0||^2
        for n, p in self._iter_trainable_named_params(model):
            theta0 = self.theta0.get(n, None)
            if theta0 is None:
                continue
            term = ((p - theta0.to(p.device)) ** 2).sum()
            reg = term if reg is None else (reg + term)
        return reg if reg is not None else torch.zeros([], device=next(model.parameters()).device)
