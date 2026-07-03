# performer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, device=None, scaling=0):
    # simplified orthogonal random features
    mat = torch.randn(nb_rows, nb_columns, device=device)
    q, _ = torch.linalg.qr(mat, mode='reduced')
    if scaling == 0:
        return q
    elif scaling == 1:
        return q * math.sqrt(nb_columns)
    else:
        return q


def softmax_kernel(data, projection_matrix, is_query, eps=1e-6):
    # data: (B, H, L, Dh)
    B, H, L, Dh = data.shape
    data_normalizer = (Dh ** -0.25)
    data = data_normalizer * data

    proj = projection_matrix  # (m, Dh)
    proj = proj.unsqueeze(0).unsqueeze(0)  # (1,1,m,Dh)

    # (B,H,L,m)
    data_dash = torch.einsum('bhld,bhmd->bhlm', data, proj.expand(B, H, -1, -1))

    diag_data = (data ** 2).sum(dim=-1, keepdim=True) / 2.0  # (B,H,L,1)
    if is_query:
        # stabilize per-row
        data_dash = data_dash - diag_data
        data_dash = data_dash - data_dash.max(dim=-1, keepdim=True).values
    else:
        data_dash = data_dash - diag_data
        data_dash = data_dash - data_dash.max().detach()

    return torch.exp(data_dash) + eps  # positive features


class PerformerAttention(nn.Module):
    def __init__(self, d_model, nhead, nb_features=256, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.dh = d_model // nhead
        self.nb_features = nb_features

        self.to_qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("proj", None, persistent=False)

    def _get_proj(self, device):
        if self.proj is None or self.proj.device != device:
            self.proj = gaussian_orthogonal_random_matrix(self.nb_features, self.dh, device=device, scaling=1)
        return self.proj

    def forward(self, x):
        # x: (B,L,D)
        B, L, D = x.shape
        qkv = self.to_qkv(x)  # (B,L,3D)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape to (B,H,L,Dh)
        q = q.view(B, L, self.nhead, self.dh).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.dh).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.dh).transpose(1, 2)

        proj = self._get_proj(x.device)  # (m,Dh)

        q_prime = softmax_kernel(q, proj, is_query=True)   # (B,H,L,m)
        k_prime = softmax_kernel(k, proj, is_query=False)  # (B,H,L,m)

        # Compute linear attention:
        # (B,H,m,Dh)
        kv = torch.einsum('bhlm,bhld->bhmd', k_prime, v)
        # (B,H,L,Dh)
        z = 1.0 / (torch.einsum('bhlm,bhm->bhl', q_prime, k_prime.sum(dim=2)) + 1e-6)
        out = torch.einsum('bhlm,bhmd->bhld', q_prime, kv)
        out = out * z.unsqueeze(-1)

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.to_out(out)
        return self.dropout(out)


class PerformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, nb_features=256, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.attn = PerformerAttention(d_model, nhead, nb_features=nb_features, dropout=dropout)
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


class PerformerClassifier(nn.Module):
    def __init__(
        self,
        in_dim=1536,
        seq_len=2000,
        num_classes=3,
        d_model=256,
        nhead=8,
        num_layers=4,
        nb_features=256,
        dim_feedforward=1024,
        dropout=0.1,
        use_cls_token=True
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
        else:
            self.cls = None

        self.layers = nn.ModuleList([
            PerformerEncoderLayer(d_model, nhead, nb_features=nb_features,
                                  dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

        nn.init.trunc_normal_(self.cls, std=0.02) if self.cls is not None else None

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