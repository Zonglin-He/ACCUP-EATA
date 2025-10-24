from typing import List

import torch
import torch.nn.functional as F
from torch import nn


class TimesBlock(nn.Module):
    """Lightweight TimesNet-style residual block operating on multi-scale 2D patches."""

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        patch_lens: List[int],
        dropout: float = 0.1,
        ffn_expansion: int = 2,
    ):
        super().__init__()
        processed_patch_lens = sorted({int(p) for p in patch_lens if int(p) > 1})
        if not processed_patch_lens:
            raise ValueError("TimesBlock requires at least one patch length greater than 1.")
        self.patch_lens = processed_patch_lens

        self.scale_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, hidden_channels, kernel_size=(3, 3), padding=(1, 1)),
                    nn.GELU(),
                    nn.Conv2d(hidden_channels, channels, kernel_size=(3, 3), padding=(1, 1)),
                )
                for _ in self.patch_lens
            ]
        )

        self.post_norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

        ffn_channels = max(channels * ffn_expansion, channels)
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, ffn_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(ffn_channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        B, C, L = x.shape
        aggregated = None
        valid = 0

        for patch_len, conv in zip(self.patch_lens, self.scale_convs):
            if L < 2:
                continue
            pad_len = (patch_len - (L % patch_len)) % patch_len
            if pad_len > 0:
                pad_mode = "reflect" if L > 1 else "replicate"
                x_pad = F.pad(x, (0, pad_len), mode=pad_mode)
            else:
                x_pad = x

            new_len = x_pad.shape[-1]
            num_patch = new_len // patch_len
            patches = x_pad.view(B, C, num_patch, patch_len)
            processed = conv(patches)
            processed = processed.view(B, C, new_len)
            if pad_len > 0:
                processed = processed[..., :L]

            aggregated = processed if aggregated is None else aggregated + processed
            valid += 1

        if aggregated is None or valid == 0:
            out = x
        else:
            out = aggregated / valid

        out = self.dropout(out)
        out = self.post_norm(out + x)
        out = out + self.ffn(out)
        return out


class TimesNet(nn.Module):
    """TimesNet-style feature extractor with configurable blocks."""

    def __init__(self, configs):
        super(TimesNet, self).__init__()
        hidden = getattr(configs, "times_hidden_channels", configs.final_out_channels)
        num_layers = getattr(configs, "times_num_layers", 3)
        patch_lens = getattr(configs, "times_patch_lens", [8, 16, 32])
        dropout = getattr(configs, "times_dropout", 0.1)
        ffn_expansion = getattr(configs, "times_ffn_expansion", 2)

        self.input_channels = configs.input_channels
        self.initial_proj = nn.Conv1d(self.input_channels, hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                TimesBlock(hidden, hidden, patch_lens, dropout=dropout, ffn_expansion=ffn_expansion)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden)
        self.final_proj = nn.Conv1d(hidden, configs.final_out_channels, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x: torch.Tensor):
        # Expect input in [B, C, L]
        z = self.initial_proj(x)
        for block in self.blocks:
            z = block(z)
        z = self.norm(z.transpose(1, 2)).transpose(1, 2)
        seq_feat = self.final_proj(z)
        pooled = self.pool(seq_feat)
        flat = pooled.reshape(pooled.size(0), -1)
        return flat, seq_feat
