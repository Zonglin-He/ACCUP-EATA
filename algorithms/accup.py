# algorithms/accup.py
import math
import os
from collections import deque
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.base_tta_algorithm import BaseTestTimeAlgorithm
from utils.utils import safe_torch_load, softmax_entropy_from_logits


class ACCUP(BaseTestTimeAlgorithm):
    """
    NuSTAR (implemented in ACCUP slot for compatibility).
    Core steps:
      1) Adversarial Amplitude Attack: pick per-sample perturbation that maximizes entropy.
      2) Triple-Safe Reliability Gate: statistical + semantic + consistency.
      3) Optimize entropy on adversarial views for reliable samples only.
    """

    def __init__(self, configs, hparams, model, optimizer):
        super(ACCUP, self).__init__(configs, hparams, model, optimizer)

        self.num_classes = int(configs.num_classes)
        self.featurizer = self.model.feature_extractor
        self.classifier = self.model.classifier

        self.adv_sigmas = self._build_adv_sigmas(hparams.get("adv_sigmas", [0.05, 0.1]))
        self.sem_thresh = float(hparams.get("sem_thresh", 0.5))
        self.cons_thresh = float(hparams.get("cons_thresh", 0.5))
        self.proto_momentum = float(hparams.get("proto_momentum", 0.9))

        self.stat_quantile = float(hparams.get("stat_quantile", 0.7))
        self.stat_window = int(hparams.get("stat_window", 512))
        self.stat_min_history = int(hparams.get("stat_min_history", 32))
        self.stat_min_entropy = float(hparams.get("stat_min_entropy", 0.0))
        self.entropy_history = deque(maxlen=self.stat_window)

        self.prototypes: Optional[torch.Tensor] = None
        self.proto_initialized: Optional[torch.Tensor] = None
        self._init_prototypes_from_model()

        self.lambda_reg = float(hparams.get("lambda_reg", hparams.get("fisher_alpha", 0.0)))
        self.max_fisher_updates = int(hparams.get("max_fisher_updates", -1))
        self.use_online_fisher = bool(hparams.get("online_fisher", True))
        self._online_fisher = None
        self._fisher_samples = 0
        self._fisher_updates = 0
        self.fishers = hparams.get("fisher_state", None)
        if self.fishers is None and "fisher_path" in hparams and os.path.exists(hparams["fisher_path"]):
            self.fishers = safe_torch_load(hparams["fisher_path"], map_location="cpu")
        self.theta_src = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self._selected_counter = 0

    def _build_adv_sigmas(self, sigmas: Iterable[float]) -> List[float]:
        if isinstance(sigmas, (float, int)):
            sigmas = [float(sigmas)]
        cleaned = [0.0]
        for s in sigmas:
            s = float(s)
            cleaned.append(abs(s))
            cleaned.append(-abs(s))
        # Preserve deterministic order while removing duplicates.
        seen = set()
        ordered = []
        for s in cleaned:
            if s not in seen:
                ordered.append(s)
                seen.add(s)
        return ordered

    def _init_prototypes_from_model(self):
        init_proto = None
        if hasattr(self.classifier, "logits") and hasattr(self.classifier.logits, "weight"):
            warmup_supports = self.classifier.logits.weight.data.detach()
            if warmup_supports.dim() == 2 and warmup_supports.size(0) == self.num_classes:
                init_proto = F.normalize(warmup_supports, dim=1)
        if init_proto is None:
            self.prototypes = None
            self.proto_initialized = None
        else:
            self.prototypes = init_proto
            self.proto_initialized = torch.ones(self.num_classes, dtype=torch.bool, device=init_proto.device)

    def _ensure_prototypes(self, feats: torch.Tensor):
        feat_dim = feats.size(1)
        device = feats.device
        if self.prototypes is None or self.prototypes.numel() == 0 or self.prototypes.size(1) != feat_dim:
            self.prototypes = torch.zeros(self.num_classes, feat_dim, device=device)
            self.proto_initialized = torch.zeros(self.num_classes, dtype=torch.bool, device=device)
        else:
            self.prototypes = self.prototypes.to(device)
            self.proto_initialized = self.proto_initialized.to(device)

    def configure_model(self, model):
        model.train()
        model.requires_grad_(False)

        freeze_bn_stats = bool(self.hparams.get("freeze_bn_stats", True))
        train_bn_affine = bool(self.hparams.get("train_bn_affine", True))
        train_full_backbone = bool(self.hparams.get("train_full_backbone", True))
        train_classifier = bool(self.hparams.get("train_classifier", True))

        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                if freeze_bn_stats:
                    module.track_running_stats = False
                    module.running_mean = None
                    module.running_var = None
                else:
                    module.track_running_stats = True
                if train_bn_affine:
                    if module.weight is not None:
                        module.weight.requires_grad_(True)
                    if module.bias is not None:
                        module.bias.requires_grad_(True)

        if train_full_backbone and hasattr(model, "feature_extractor"):
            for param in model.feature_extractor.parameters():
                param.requires_grad_(True)

        if train_classifier and hasattr(model, "classifier"):
            for param in model.classifier.parameters():
                param.requires_grad_(True)

        return model

    @staticmethod
    def _extract_features(model, x: torch.Tensor) -> torch.Tensor:
        feats = model.feature_extractor(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return feats

    def _entropy_threshold(self, entropy: torch.Tensor) -> torch.Tensor:
        if len(self.entropy_history) >= self.stat_min_history:
            history = torch.tensor(list(self.entropy_history), device=entropy.device, dtype=entropy.dtype)
            threshold = torch.quantile(history, self.stat_quantile)
        else:
            threshold = entropy.new_tensor(math.log(max(2, self.num_classes)))
        threshold = torch.clamp(threshold, min=self.stat_min_entropy)
        return threshold

    def _update_entropy_history(self, entropy: torch.Tensor):
        self.entropy_history.extend(entropy.detach().cpu().tolist())

    def get_adversarial_view(self, x: torch.Tensor, model) -> torch.Tensor:
        best_entropy = None
        best_view = None
        with torch.no_grad():
            for sigma in self.adv_sigmas:
                x_pert = x * (1.0 + sigma)
                feats = self._extract_features(model, x_pert)
                logits = model.classifier(feats)
                ent = softmax_entropy_from_logits(logits)
                if best_entropy is None:
                    best_entropy = ent
                    best_view = x_pert
                else:
                    pick = ent > best_entropy
                    best_entropy = torch.where(pick, ent, best_entropy)
                    while pick.dim() < x_pert.dim():
                        pick = pick.unsqueeze(-1)
                    best_view = torch.where(pick, x_pert, best_view)
        return best_view

    def update_prototypes(self, feats: torch.Tensor, labels: torch.Tensor):
        return self.update_prototypes_with_weights(feats, labels, weights=None)

    def update_prototypes_with_weights(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ):
        if feats.numel() == 0:
            return
        self._ensure_prototypes(feats)
        feats = F.normalize(feats, dim=1)
        for cls_idx in range(self.num_classes):
            mask = labels == cls_idx
            if not torch.any(mask):
                continue
            if weights is None:
                class_mean = feats[mask].mean(dim=0)
            else:
                w = weights[mask].clamp_min(0.0)
                if float(w.sum().item()) == 0.0:
                    continue
                w = w / w.sum()
                class_mean = (feats[mask] * w.unsqueeze(1)).sum(dim=0)
            if not self.proto_initialized[cls_idx]:
                self.prototypes[cls_idx] = class_mean
                self.proto_initialized[cls_idx] = True
            else:
                self.prototypes[cls_idx] = (
                    self.proto_momentum * self.prototypes[cls_idx]
                    + (1.0 - self.proto_momentum) * class_mean
                )
            self.prototypes[cls_idx] = F.normalize(self.prototypes[cls_idx], dim=0)

    def _maybe_update_online_fisher(self, model, logits: torch.Tensor):
        if not self.use_online_fisher:
            return
        if self.lambda_reg <= 0:
            return
        if logits is None or not logits.requires_grad:
            return
        if self.max_fisher_updates >= 0 and self._fisher_updates >= self.max_fisher_updates:
            return

        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not trainable:
            return

        if self._online_fisher is None:
            self._online_fisher = {n: torch.zeros_like(p) for n, p in trainable}

        names, params = zip(*trainable)
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)
        fisher_loss = -(probs * log_probs).sum(dim=1).mean()
        grads = torch.autograd.grad(fisher_loss, params, retain_graph=True, allow_unused=True)
        any_update = False
        for name, grad in zip(names, grads):
            if grad is None:
                continue
            self._online_fisher[name] = self._online_fisher[name].to(grad.device)
            self._online_fisher[name] += grad.detach() ** 2
            any_update = True
        if any_update:
            self._fisher_samples += logits.size(0)
            self._fisher_updates += 1

    def _fisher_regularizer(self, model) -> torch.Tensor:
        if self.lambda_reg <= 0:
            return torch.zeros([], device=next(model.parameters()).device)
        device = next(model.parameters()).device
        reg = None

        if isinstance(self.fishers, dict) and len(self.fishers) > 0:
            for n, p in model.named_parameters():
                if not p.requires_grad or n not in self.fishers:
                    continue
                item = self.fishers[n]
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    diag, theta_prev = item[0], item[1]
                else:
                    diag, theta_prev = item, self.theta_src.get(n, None)
                if theta_prev is None:
                    continue
                term = (diag.to(device) * (p - theta_prev.to(device)) ** 2).sum()
                reg = term if reg is None else reg + term
            if reg is not None:
                return self.lambda_reg * reg

        if self._online_fisher and self._fisher_samples > 0:
            normalizer = float(self._fisher_samples)
            for n, p in model.named_parameters():
                if not p.requires_grad or n not in self._online_fisher:
                    continue
                theta_prev = self.theta_src.get(n, p.detach()).to(device)
                diag = (self._online_fisher[n] / normalizer).to(device)
                term = (diag * (p - theta_prev) ** 2).sum()
                reg = term if reg is None else reg + term
            if reg is not None:
                return self.lambda_reg * reg

        return torch.zeros([], device=device)

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        if isinstance(batch_data, (list, tuple)):
            raw_data = batch_data[0]
        else:
            raw_data = batch_data

        x_adv = self.get_adversarial_view(raw_data, model)

        raw_feats = self._extract_features(model, raw_data)
        raw_logits = model.classifier(raw_feats)
        raw_probs = F.softmax(raw_logits, dim=1)

        adv_feats = self._extract_features(model, x_adv)
        adv_logits = model.classifier(adv_feats)
        adv_probs = F.softmax(adv_logits, dim=1)

        self._maybe_update_online_fisher(model, raw_logits)

        self._ensure_prototypes(raw_feats)

        # Gate 1: statistical low-entropy filter (dynamic quantile threshold).
        p_bar = 0.5 * (raw_probs + adv_probs)
        stat_entropy = -(p_bar * p_bar.clamp_min(1e-8).log()).sum(dim=1)
        entropy_th = self._entropy_threshold(stat_entropy.detach())
        mask_stat = stat_entropy <= entropy_th
        self._update_entropy_history(stat_entropy)

        # Gate 2: semantic alignment to class prototypes (skip when prototype is cold).
        pred_labels = raw_probs.argmax(dim=1)
        proto_ready = self.proto_initialized[pred_labels]
        proto_vecs = self.prototypes[pred_labels]
        cos_sim = F.cosine_similarity(
            F.normalize(raw_feats.detach(), dim=1),
            F.normalize(proto_vecs.detach(), dim=1),
            dim=1,
        )
        mask_sem = (~proto_ready) | (cos_sim >= self.sem_thresh)

        # Gate 3: consistency under adversarial attack.
        log_adv = (adv_probs.detach().clamp_min(1e-8)).log()
        kl_div = F.kl_div(log_adv, raw_probs.detach(), reduction="none").sum(dim=1)
        mask_cons = kl_div <= self.cons_thresh

        active_mask = mask_stat & mask_sem & mask_cons
        active_count = int(active_mask.sum().item())
        self._selected_counter += active_count

        adv_entropy = softmax_entropy_from_logits(adv_logits)
        reg_loss = self._fisher_regularizer(model)
        if active_mask.any():
            loss_adv = adv_entropy[active_mask].mean()
        else:
            loss_adv = adv_entropy.new_zeros([])
        loss = loss_adv + reg_loss

        if optimizer is not None and loss.requires_grad:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Update prototypes with reliable samples; allow cold start with low-entropy only.
        quality = 1.0 - (stat_entropy.detach() / math.log(max(2, self.num_classes)))
        quality = quality.clamp(min=0.0, max=1.0)
        if active_mask.any():
            self.update_prototypes_with_weights(
                raw_feats.detach()[active_mask],
                pred_labels.detach()[active_mask],
                quality[active_mask],
            )
        elif mask_stat.any() and (self.proto_initialized is not None) and (not torch.all(self.proto_initialized)):
            self.update_prototypes_with_weights(
                raw_feats.detach()[mask_stat],
                pred_labels.detach()[mask_stat],
                quality[mask_stat],
            )

        return raw_logits


# Optional alias for clarity if users refer to NuSTAR explicitly.
NuSTAR = ACCUP
