import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenStem(nn.Module):
    """
    对每个 token 做:
      1) RMSNorm(x)
      2) Linear -> d_model
      3) 显式加入 weight embedding
      4) 加 branch/type embedding
    """
    def __init__(self, in_dim=1536, d_model=256, dropout=0.1, weight_scale=1000.0):
        super().__init__()
        self.norm = nn.RMSNorm(in_dim)
        self.proj = nn.Linear(in_dim, d_model)
        self.weighted_proj = nn.Linear(in_dim, d_model)
        self.weight_scale = float(weight_scale)

        self.weight_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

        self.out = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, w, branch_embed):
        """
        x: (B, L, in_dim), stored as expression_weight * gene_embedding
        w: (B, L)   raw weights
        branch_embed: (1, 1, d_model)
        """
        # Recover the gene embedding direction before normalization. Applying
        # RMSNorm directly to expression_weight * embedding erases most of the
        # expression magnitude information.
        w_safe = torch.clamp(w, min=1e-8).unsqueeze(-1)
        has_expr = (w > 0).unsqueeze(-1)
        x_gene = torch.where(has_expr, x / w_safe, torch.zeros_like(x))

        w_feat = torch.log1p(torch.clamp(w, min=0.0) * self.weight_scale).unsqueeze(-1)  # (B, L, 1)
        weight_feature = self.weight_mlp(w_feat)
        gate = self.gate_mlp(w_feat)

        gene_feature = self.proj(self.norm(x_gene))
        weighted_feature = self.weighted_proj(x)
        x = gene_feature * (1.0 + gate) + weighted_feature + weight_feature + branch_embed
        x = self.out(x)
        return x


class CrossAttentionBlock(nn.Module):
    """
    latent queries 对 input tokens 做 cross-attention
    """
    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q, kv, kv_mask=None):
        """
        q:  (B, M, d_model)     latent
        kv: (B, N, d_model)     tokens
        kv_mask: (B, N) bool, True=ignore
        """
        q2 = self.q_norm(q)
        kv2 = self.kv_norm(kv)
        attn_out, _ = self.attn(q2, kv2, kv2, key_padding_mask=kv_mask)
        q = q + attn_out
        q = q + self.ffn(self.ffn_norm(q))
        return q


class LatentSelfBlock(nn.Module):
    """
    latent 上做 self-attention
    """
    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super().__init__()
        self.block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x):
        return self.block(x)


class AttentionPool(nn.Module):
    """
    类似 Set Transformer 的 PMA 思想:
    用 learnable query 从 latent 中聚合一个全局表示
    """
    def __init__(self, d_model=256, nhead=8, dropout=0.1):
        super().__init__()
        self.pool_q = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

        nn.init.trunc_normal_(self.pool_q, std=0.02)

    def forward(self, x):
        """
        x: (B, M, d_model)
        return: (B, d_model)
        """
        B = x.size(0)
        q = self.pool_q.expand(B, -1, -1)  # (B,1,d)
        pooled, _ = self.attn(q, x, x)
        feat = self.norm(pooled[:, 0])
        return feat


class GenoSmart(nn.Module):
    """
    输入:
      x_xgb:  (B, 2000, 1536)
      x_case: (B, 2000, 1536)
      w_xgb:  (B, 2000)
      w_case: (B, 2000)

    输出:
      logits: (B, 3)
      probs:  (B, 3)
    """
    def __init__(
        self,
        in_dim=1536,
        seq_len_xgb=2000,
        seq_len_case=2000,
        num_classes=3,
        d_model=64,
        nhead=4,
        branch_layers=2,
        latent_len=128,
        latent_layers=4,
        dropout=0.1,
    ):
        super().__init__()

        self.seq_len_xgb = seq_len_xgb
        self.seq_len_case = seq_len_case
        self.d_model = d_model
        self.register_buffer("expr_mean", torch.zeros(1, seq_len_xgb + seq_len_case))
        self.register_buffer("expr_std", torch.ones(1, seq_len_xgb + seq_len_case))

        # 两个 branch 的 type embedding
        self.xgb_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.case_type = nn.Parameter(torch.zeros(1, 1, d_model))

        # token stem
        self.xgb_stem = TokenStem(in_dim=in_dim, d_model=d_model, dropout=dropout)
        self.case_stem = TokenStem(in_dim=in_dim, d_model=d_model, dropout=dropout)
        self.xgb_pos_embed = nn.Parameter(torch.zeros(1, seq_len_xgb, d_model))
        self.case_pos_embed = nn.Parameter(torch.zeros(1, seq_len_case, d_model))

        # 分支内 self-attention
        xgb_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        case_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.xgb_encoder = nn.TransformerEncoder(xgb_layer, num_layers=branch_layers)
        self.case_encoder = nn.TransformerEncoder(case_layer, num_layers=branch_layers)

        # Perceiver-style latent bottleneck
        self.latents = nn.Parameter(torch.zeros(1, latent_len, d_model))
        self.cross_block = CrossAttentionBlock(d_model=d_model, nhead=nhead, dropout=dropout)
        self.latent_blocks = nn.ModuleList([
            LatentSelfBlock(d_model=d_model, nhead=nhead, dropout=dropout)
            for _ in range(latent_layers)
        ])

        # attention pooling
        self.pool = AttentionPool(d_model=d_model, nhead=nhead, dropout=dropout)

        summary_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.branch_summary_fusion = nn.TransformerEncoder(summary_layer, num_layers=1)

        # Small expression bottleneck. It helps the model keep patient-level
        # expression signal, but it does not produce logits directly.
        self.expr_bottleneck = nn.Sequential(
            nn.LayerNorm(seq_len_xgb + seq_len_case),
            nn.Linear(seq_len_xgb + seq_len_case, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Classifier over fused Transformer and expression-bottleneck features.
        self.head = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def set_expr_normalization(self, mean, std):
        mean = torch.as_tensor(mean, dtype=self.expr_mean.dtype, device=self.expr_mean.device).view(1, -1)
        std = torch.as_tensor(std, dtype=self.expr_std.dtype, device=self.expr_std.device).view(1, -1)
        if mean.shape != self.expr_mean.shape:
            raise ValueError(f"Expected expr mean shape {self.expr_mean.shape}, got {mean.shape}")
        if std.shape != self.expr_std.shape:
            raise ValueError(f"Expected expr std shape {self.expr_std.shape}, got {std.shape}")
        self.expr_mean.copy_(mean)
        self.expr_std.copy_(torch.clamp(std, min=1e-6))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.trunc_normal_(self.xgb_type, std=0.02)
        nn.init.trunc_normal_(self.case_type, std=0.02)
        nn.init.trunc_normal_(self.xgb_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.case_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.latents, std=0.02)

    @staticmethod
    def _weighted_mean(tokens, weights, eps=1e-8):
        weights = torch.clamp(weights, min=0.0)
        denom = torch.clamp(weights.sum(dim=1, keepdim=True), min=eps)
        norm_weights = (weights / denom).unsqueeze(-1)
        return torch.sum(tokens * norm_weights, dim=1)

    def forward(self, x_xgb, x_case, w_xgb, w_case):
        if x_xgb.dim() == 2:
            x_xgb = x_xgb.unsqueeze(0)
        if x_case.dim() == 2:
            x_case = x_case.unsqueeze(0)
        if w_xgb.dim() == 1:
            w_xgb = w_xgb.unsqueeze(0)
        if w_case.dim() == 1:
            w_case = w_case.unsqueeze(0)

        B, L1, C1 = x_xgb.shape
        B2, L2, C2 = x_case.shape

        if B != B2:
            raise ValueError("Batch size mismatch between x_xgb and x_case")
        if L1 != self.seq_len_xgb:
            raise ValueError(f"Expected x_xgb length={self.seq_len_xgb}, got {L1}")
        if L2 != self.seq_len_case:
            raise ValueError(f"Expected x_case length={self.seq_len_case}, got {L2}")

        # 可以把零权重 token 视为 padding
        mask_xgb = (w_xgb <= 0)
        mask_case = (w_case <= 0)

        # 1) stem
        tx = self.xgb_stem(x_xgb, w_xgb, self.xgb_type)    # (B, L1, d)
        tc = self.case_stem(x_case, w_case, self.case_type) # (B, L2, d)
        tx = tx + self.xgb_pos_embed
        tc = tc + self.case_pos_embed

        # 2) branch encoders
        tx = self.xgb_encoder(tx, src_key_padding_mask=mask_xgb)
        tc = self.case_encoder(tc, src_key_padding_mask=mask_case)

        # 2.5) two branch-level vectors: XGB vector and patient-specific X vector
        xgb_vec = self._weighted_mean(tx, w_xgb)
        case_vec = self._weighted_mean(tc, w_case)
        branch_vectors = torch.stack([xgb_vec, case_vec], dim=1)
        branch_vectors = self.branch_summary_fusion(branch_vectors)
        branch_feat = branch_vectors.reshape(B, 2 * self.d_model)

        # 3) concat both branches
        tokens = torch.cat([tx, tc], dim=1)  # (B, L1+L2, d)
        token_mask = torch.cat([mask_xgb, mask_case], dim=1)  # (B, L1+L2)

        # 4) latent bottleneck cross-attention
        latents = self.latents.expand(B, -1, -1)  # (B, M, d)
        latents = self.cross_block(latents, tokens, kv_mask=token_mask)

        # 5) latent self-attention
        for blk in self.latent_blocks:
            latents = blk(latents)

        # 6) attention pooling
        expr_weights = torch.log1p(
            torch.cat([torch.clamp(w_xgb, min=0.0), torch.clamp(w_case, min=0.0)], dim=1) * 1000.0
        )
        expr_weights = (expr_weights - self.expr_mean) / self.expr_std
        expr_feat = self.expr_bottleneck(expr_weights)
        feat = torch.cat([self.pool(latents), branch_feat, expr_feat], dim=1)

        # 7) classifier
        logits = self.head(feat)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


class GenoSmartLite(nn.Module):
    """
    Smaller attentive-pooling variant for low-sample regimes.

    It keeps the same input contract as GenoSmart, but replaces the heavy
    branch Transformer + latent Transformer stack with token-level attention
    pooling. Each gene token can contribute both to a branch representation and
    directly to class logits through an attention-weighted token scorer.
    """
    def __init__(
        self,
        in_dim=1536,
        seq_len_xgb=2000,
        seq_len_case=2000,
        num_classes=3,
        d_model=64,
        nhead=4,
        branch_layers=0,
        latent_len=0,
        latent_layers=0,
        dropout=0.2,
    ):
        super().__init__()
        del nhead, branch_layers, latent_len, latent_layers

        self.seq_len_xgb = seq_len_xgb
        self.seq_len_case = seq_len_case
        self.d_model = d_model
        self.num_classes = num_classes
        self.register_buffer("expr_mean", torch.zeros(1, seq_len_xgb + seq_len_case))
        self.register_buffer("expr_std", torch.ones(1, seq_len_xgb + seq_len_case))

        self.xgb_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.case_type = nn.Parameter(torch.zeros(1, 1, d_model))
        self.xgb_pos_embed = nn.Parameter(torch.zeros(1, seq_len_xgb, d_model))
        self.case_pos_embed = nn.Parameter(torch.zeros(1, seq_len_case, d_model))

        self.xgb_stem = TokenStem(in_dim=in_dim, d_model=d_model, dropout=dropout)
        self.case_stem = TokenStem(in_dim=in_dim, d_model=d_model, dropout=dropout)

        self.token_norm = nn.LayerNorm(d_model)
        self.attn_score = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.token_logits = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
        self.token_logit_scale = nn.Parameter(torch.tensor(0.1))

        self.expr_bottleneck = nn.Sequential(
            nn.LayerNorm(seq_len_xgb + seq_len_case),
            nn.Linear(seq_len_xgb + seq_len_case, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(5 * d_model),
            nn.Linear(5 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, num_classes),
        )

        self._init_weights()

    def set_expr_normalization(self, mean, std):
        mean = torch.as_tensor(mean, dtype=self.expr_mean.dtype, device=self.expr_mean.device).view(1, -1)
        std = torch.as_tensor(std, dtype=self.expr_std.dtype, device=self.expr_std.device).view(1, -1)
        if mean.shape != self.expr_mean.shape:
            raise ValueError(f"Expected expr mean shape {self.expr_mean.shape}, got {mean.shape}")
        if std.shape != self.expr_std.shape:
            raise ValueError(f"Expected expr std shape {self.expr_std.shape}, got {std.shape}")
        self.expr_mean.copy_(mean)
        self.expr_std.copy_(torch.clamp(std, min=1e-6))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.trunc_normal_(self.xgb_type, std=0.02)
        nn.init.trunc_normal_(self.case_type, std=0.02)
        nn.init.trunc_normal_(self.xgb_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.case_pos_embed, std=0.02)

    def _pool_branch(self, tokens, weights):
        token_mask = weights <= 0
        norm_tokens = self.token_norm(tokens)
        scores = self.attn_score(norm_tokens).squeeze(-1)
        scores = scores.masked_fill(token_mask, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=1)
        attn = torch.where(token_mask, torch.zeros_like(attn), attn)
        attn = attn / torch.clamp(attn.sum(dim=1, keepdim=True), min=1e-8)

        vec = torch.sum(tokens * attn.unsqueeze(-1), dim=1)
        logits = torch.sum(self.token_logits(norm_tokens) * attn.unsqueeze(-1), dim=1)
        return vec, logits

    def forward(self, x_xgb, x_case, w_xgb, w_case):
        if x_xgb.dim() == 2:
            x_xgb = x_xgb.unsqueeze(0)
        if x_case.dim() == 2:
            x_case = x_case.unsqueeze(0)
        if w_xgb.dim() == 1:
            w_xgb = w_xgb.unsqueeze(0)
        if w_case.dim() == 1:
            w_case = w_case.unsqueeze(0)

        B, L1, _ = x_xgb.shape
        B2, L2, _ = x_case.shape
        if B != B2:
            raise ValueError("Batch size mismatch between x_xgb and x_case")
        if L1 != self.seq_len_xgb:
            raise ValueError(f"Expected x_xgb length={self.seq_len_xgb}, got {L1}")
        if L2 != self.seq_len_case:
            raise ValueError(f"Expected x_case length={self.seq_len_case}, got {L2}")

        tx = self.xgb_stem(x_xgb, w_xgb, self.xgb_type) + self.xgb_pos_embed
        tc = self.case_stem(x_case, w_case, self.case_type) + self.case_pos_embed

        xgb_vec, xgb_logits = self._pool_branch(tx, w_xgb)
        case_vec, case_logits = self._pool_branch(tc, w_case)

        expr_weights = torch.log1p(
            torch.cat([torch.clamp(w_xgb, min=0.0), torch.clamp(w_case, min=0.0)], dim=1) * 1000.0
        )
        expr_weights = (expr_weights - self.expr_mean) / self.expr_std
        expr_feat = self.expr_bottleneck(expr_weights)

        fused = torch.cat([
            xgb_vec,
            case_vec,
            torch.abs(xgb_vec - case_vec),
            xgb_vec * case_vec,
            expr_feat,
        ], dim=1)
        logits = self.head(fused) + self.token_logit_scale * (xgb_logits + case_logits)
        probs = F.softmax(logits, dim=-1)
        return logits, probs


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GenoSmart(
        in_dim=1536,
        seq_len_xgb=2000,
        seq_len_case=2000,
        num_classes=3,
        d_model=256,
        nhead=8,
        branch_layers=2,
        latent_len=64,
        latent_layers=2,
        dropout=0.1,
    ).to(device)

    # batch from your dataloader
    # x_xgb:  (B,2000,1536)
    # x_case: (B,2000,1536)
    # y:      (B,3)
    # w_xgb:  (B,2000)
    # w_case: (B,2000)

    logits, probs = model(x_xgb.to(device), x_case.to(device), w_xgb.to(device), w_case.to(device))
