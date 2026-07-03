# xformers_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from xformers.ops import memory_efficient_attention
    _HAS_XFORMERS = True
except Exception:
    _HAS_XFORMERS = False


class XFormerMHA(nn.Module):
    """
    Multi-head attention using xformers memory_efficient_attention.
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.dh = d_model // nhead
        self.dropout = dropout

        self.to_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        if not _HAS_XFORMERS:
            raise ImportError("xFormers is not installed. Please: pip install xformers")

        B, L, D = x.shape
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # (B, H, L, Dh)
        q = q.view(B, L, self.nhead, self.dh).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.dh).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.dh).transpose(1, 2)

        # xformers expects (B*H, L, Dh)
        q = q.reshape(B * self.nhead, L, self.dh)
        k = k.reshape(B * self.nhead, L, self.dh)
        v = v.reshape(B * self.nhead, L, self.dh)

        out = memory_efficient_attention(q, k, v, p=self.dropout)  # (B*H, L, Dh)
        out = out.view(B, self.nhead, L, self.dh).transpose(1, 2).contiguous().view(B, L, D)
        return self.to_out(out)


class XFormerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.attn = XFormerMHA(d_model, nhead, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ff(x))
        return x


class XFormersTransformerClassifier(nn.Module):
    def __init__(
        self,
        in_dim=1536,
        seq_len=2000,
        num_classes=3,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        use_cls_token=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.use_cls = use_cls_token

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        if use_cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)
        else:
            self.cls = None

        self.layers = nn.ModuleList([
            XFormerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, L, _ = x.shape
        if L != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {L}")

        x = self.input_proj(x)
        if self.use_cls:
            x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1)  # (B,2001,D)

        for layer in self.layers:
            x = layer(x)

        feat = x[:, 0] if self.use_cls else x.mean(dim=1)
        logits = self.head(feat)
        probs = F.softmax(logits, dim=-1)
        return logits, probs