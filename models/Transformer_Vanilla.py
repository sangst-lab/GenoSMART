import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    """
    True Transformer Encoder for gene-token sequences.

    Input:
      x: (B, 2000, 1536) or (2000, 1536)
    Output:
      logits: (B, 3)
      probs:  (B, 3)
    """

    def __init__(
        self,
        in_dim: int = 1536,
        seq_len: int = 2000,
        num_classes: int = 3,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        use_pos_embed: bool = False,   # gene是集合，默认不加pos
    ):
        super().__init__()

        self.seq_len = seq_len
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token
        self.use_pos_embed = use_pos_embed

        # 1536 -> d_model
        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # CLS
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            pos_len = seq_len + 1
        else:
            self.cls_token = None
            pos_len = seq_len

        # 可选 pos embedding
        if use_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_len, d_model))
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(dropout)

        # ✅ 真正的 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        # allow single sample
        if x.dim() == 2:
            x = x.unsqueeze(0)

        B, L, C = x.shape
        if L != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got L={L}")

        x = self.input_proj(x)  # (B, L, d_model)

        if self.use_cls_token:
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)  # (B, 1+L, d_model)

            # mask对齐
            if key_padding_mask is not None:
                cls_mask = torch.zeros((B, 1), dtype=torch.bool, device=key_padding_mask.device)
                key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.pos_drop(x)

        # ✅ Transformer Encoder
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # pooling
        if self.use_cls_token:
            feat = x[:, 0]
        else:
            feat = x.mean(dim=1)

        feat = self.norm(feat)
        logits = self.head(feat)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


def main():
    # 1) device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")
    if device == "cuda":
        print(f"[INFO] gpu    = {torch.cuda.get_device_name(0)}")

    # 2) config
    B = 2
    seq_len = 2000
    in_dim = 1536
    num_classes = 3

    # 3) model
    model = TransformerEncoder(
        in_dim=in_dim,
        seq_len=seq_len,
        num_classes=num_classes,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        use_cls_token=True,
        use_pos_embed=False,
    ).to(device)

    model.eval()

    # 4) fake input
    x = torch.randn(B, seq_len, in_dim, device=device)

    # 5) optional key_padding_mask
    #    True 表示该 token 是 padding（会被忽略）；False 表示有效
    #    这里做个示例：第0个样本最后 300 个 token 视为 padding
    key_padding_mask = torch.zeros((B, seq_len), dtype=torch.bool, device=device)
    key_padding_mask[0, -300:] = True

    # 6) forward test (with mask)
    with torch.no_grad():
        logits, probs = model(x, key_padding_mask=key_padding_mask)

    print("[INFO] logits shape:", tuple(logits.shape))
    print("[INFO] probs  shape:", tuple(probs.shape))
    print("[INFO] probs row sum (should be ~1):", probs.sum(dim=1))

    # 7) sanity checks
    assert logits.shape == (B, num_classes), f"Bad logits shape: {logits.shape}"
    assert probs.shape == (B, num_classes), f"Bad probs shape: {probs.shape}"
    assert torch.allclose(probs.sum(dim=1), torch.ones(B, device=device), atol=1e-5), "Softmax not summing to 1"

    # 8) test single sample input (2D)
    x_single = torch.randn(seq_len, in_dim, device=device)
    with torch.no_grad():
        logits1, probs1 = model(x_single)  # should auto unsqueeze
    print("[INFO] single sample logits shape:", tuple(logits1.shape))
    print("[INFO] single sample probs  shape:", tuple(probs1.shape))

    print("\n✅ Forward 测试通过：模型可以正常跑通。")

import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer_AttentionPool(nn.Module):
    """
    输入: x (B, 2000, 1536)
    输出: logits (B, C), probs (B, C)
    """
    def __init__(
        self,
        in_dim=1536,
        seq_len=2000,
        num_classes=3,
        d_model=256,
        nhead=8,
        num_layers=2,          # ✅ 建议先降到 2，更稳
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len

        self.input_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,     # ✅ 训练更稳
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # ✅ 关键：attention pooling（learnable query）
        self.pool_q = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.pool_q, std=0.02)

        self.pool_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, key_padding_mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, L, _ = x.shape
        if L != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {L}")

        x = self.input_proj(x)                 # (B, L, d)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        # pool: q attends to tokens
        q = self.pool_q.expand(B, -1, -1)      # (B, 1, d)
        pooled, _ = self.pool_attn(q, x, x, key_padding_mask=key_padding_mask)  # (B,1,d)
        feat = pooled[:, 0]                    # (B, d)

        feat = self.norm(feat)
        logits = self.head(feat)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

if __name__ == "__main__":
    main()