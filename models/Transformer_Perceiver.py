# perceiver_io.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttnBlock(nn.Module):
    """Latents attend to inputs (cross-attention) + FFN"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, latents, inputs, key_padding_mask=None):
        h, _ = self.attn(latents, inputs, inputs, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(latents + h)
        y = self.ff(x)
        out = self.ln2(x + y)
        return out


class SelfAttnBlock(nn.Module):
    """Latents self-attend + FFN"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + h)
        y = self.ff(x)
        out = self.ln2(x + y)
        return out


class PerceiverClassifier(nn.Module):
    """
    Inputs: (B, 2000, 1536)
    Latents: (B, M, d_model), M << 2000
    """
    def __init__(
        self,
        in_dim=1536,
        seq_len=2000,
        num_classes=3,
        d_model=256,
        nhead=8,
        num_latents=128,
        num_layers=4,          # number of iterations
        self_attn_per_layer=1, # latent self-attn blocks per iteration
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.num_latents = num_latents

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.latents = nn.Parameter(torch.randn(1, num_latents, d_model) * 0.02)

        self.cross_blocks = nn.ModuleList([
            CrossAttnBlock(d_model, nhead, dropout=dropout) for _ in range(num_layers)
        ])
        self.self_blocks = nn.ModuleList([
            nn.ModuleList([SelfAttnBlock(d_model, nhead, dropout=dropout) for _ in range(self_attn_per_layer)])
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x, key_padding_mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, L, _ = x.shape
        if L != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {L}")

        x = self.input_proj(x)
        z = self.latents.expand(B, -1, -1)

        for i in range(len(self.cross_blocks)):
            z = self.cross_blocks[i](z, x, key_padding_mask=key_padding_mask)
            for sb in self.self_blocks[i]:
                z = sb(z)

        # pool latents (mean)
        feat = z.mean(dim=1)
        logits = self.head(feat)
        probs = F.softmax(logits, dim=-1)
        return logits, probs