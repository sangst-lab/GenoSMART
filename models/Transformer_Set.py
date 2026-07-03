# set_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAB(nn.Module):
    """Multihead Attention Block (Set Transformer)"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, q, k, key_padding_mask=None):
        # q,k: (B,L,D)
        h, _ = self.attn(q, k, k, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(q + h)
        y = self.ff(x)
        out = self.ln2(x + y)
        return out


class ISAB(nn.Module):
    """Induced Set Attention Block"""
    def __init__(self, d_model, nhead, num_inducing=32, dropout=0.1):
        super().__init__()
        self.I = nn.Parameter(torch.randn(1, num_inducing, d_model) * 0.02)
        self.mab1 = MAB(d_model, nhead, dropout=dropout)  # I attends to X
        self.mab2 = MAB(d_model, nhead, dropout=dropout)  # X attends to induced

    def forward(self, x, key_padding_mask=None):
        B = x.size(0)
        I = self.I.expand(B, -1, -1)
        H = self.mab1(I, x, key_padding_mask=key_padding_mask)
        out = self.mab2(x, H, key_padding_mask=None)  # induced set has no padding
        return out


class PMA(nn.Module):
    """Pooling by Multihead Attention"""
    def __init__(self, d_model, nhead, num_seeds=1, dropout=0.1):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, d_model) * 0.02)
        self.mab = MAB(d_model, nhead, dropout=dropout)

    def forward(self, x, key_padding_mask=None):
        B = x.size(0)
        S = self.S.expand(B, -1, -1)
        return self.mab(S, x, key_padding_mask=key_padding_mask)  # (B, num_seeds, D)


class SetTransformerClassifier(nn.Module):
    """
    Permutation-invariant Transformer for sets.
    """
    def __init__(
        self,
        in_dim=1536,
        seq_len=2000,
        num_classes=3,
        d_model=256,
        nhead=8,
        num_isab=2,
        num_inducing=64,
        dropout=0.1,
        num_seeds=1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            ISAB(d_model, nhead, num_inducing=num_inducing, dropout=dropout)
            for _ in range(num_isab)
        ])
        self.pma = PMA(d_model, nhead, num_seeds=num_seeds, dropout=dropout)

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
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)

        pooled = self.pma(x, key_padding_mask=key_padding_mask)  # (B,1,D)
        pooled = pooled[:, 0]
        logits = self.head(pooled)
        probs = F.softmax(logits, dim=-1)
        return logits, probs