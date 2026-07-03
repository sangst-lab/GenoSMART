import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import csv
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

try:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
except Exception:
    accuracy_score = None
    f1_score = None
    roc_auc_score = None

# ===== 你自己的模型 =====
from models.Transformer import GenoSmart

# ===== 你自己的数据集 =====
from data.hcc_dataset import HCCDatasetDual, HCCTransform, load_weights_from_meta

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def label_to_class_index(y: Any) -> torch.Tensor:
    """
    把各种形式的 label 统一转换成类别 id（LongTensor）
    支持：
      - python int
      - numpy scalar / ndarray
      - torch scalar / shape(1,) / one-hot / prob vector
    返回：
      - torch.LongTensor shape (B,) 或 ()（标量也行）
    """
    if isinstance(y, (int, np.integer)):
        return torch.tensor(y, dtype=torch.long)

    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    if torch.is_tensor(y):
        if y.numel() == 1:
            return y.long().view(-1)

        if y.dim() == 1:
            return torch.argmax(y).long().view(-1)
        elif y.dim() == 2:
            return torch.argmax(y, dim=1).long().view(-1)

        return torch.argmax(y.view(y.shape[0], -1), dim=1).long().view(-1)

    raise TypeError(f"Unsupported label type: {type(y)}")


def extract_batch_dual(batch, device: str):
    """
    适配 HCCDatasetDual 返回：
      x_xgb, x_case, y, w_xgb, w_case, case_id

    返回：
      x_xgb, x_case, y, w_xgb, w_case
    """
    if isinstance(batch, (tuple, list)):
        if len(batch) < 5:
            raise ValueError(f"Batch length < 5, got {len(batch)}")

        x_xgb = batch[0]
        x_case = batch[1]
        y = batch[2]
        w_xgb = batch[3]
        w_case = batch[4]

    elif isinstance(batch, dict):
        x_xgb = batch["x_xgb"]
        x_case = batch["x_case"]
        y = batch["y"]
        w_xgb = batch["w_xgb"]
        w_case = batch["w_case"]
    else:
        raise ValueError(f"Unsupported batch type: {type(batch)}")

    # x_xgb
    if isinstance(x_xgb, np.ndarray):
        x_xgb = torch.from_numpy(x_xgb)
    x_xgb = x_xgb.to(device)

    # x_case
    if isinstance(x_case, np.ndarray):
        x_case = torch.from_numpy(x_case)
    x_case = x_case.to(device)

    # w_xgb
    if isinstance(w_xgb, np.ndarray):
        w_xgb = torch.from_numpy(w_xgb)
    w_xgb = w_xgb.to(device)

    # w_case
    if isinstance(w_case, np.ndarray):
        w_case = torch.from_numpy(w_case)
    w_case = w_case.to(device)

    # y
    y = label_to_class_index(y).to(device)
    if y.dim() == 0:
        y = y.view(1)

    return x_xgb, x_case, y, w_xgb, w_case


def compute_metrics_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    y_true: (N,) int
    y_prob: (N, C) float
    """
    y_pred = y_prob.argmax(axis=1)

    out = {}
    if accuracy_score is not None and f1_score is not None:
        out["Accuracy"] = float(accuracy_score(y_true, y_pred))
        out["Macro-F1"] = float(f1_score(y_true, y_pred, average="macro"))
        out["Weighted-F1"] = float(f1_score(y_true, y_pred, average="weighted"))
        out["Micro-F1"] = float(f1_score(y_true, y_pred, average="micro"))
    else:
        out.update(_compute_basic_metrics(y_true, y_pred))

    try:
        if roc_auc_score is None:
            raise RuntimeError("sklearn is not available")
        out["ROC-AUC"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        out["ROC-AUC"] = float("nan")

    return out


def _compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")

    f1s = []
    supports = []
    for c in classes:
        tp = int(np.sum((y_true == c) & (y_pred == c)))
        fp = int(np.sum((y_true != c) & (y_pred == c)))
        fn = int(np.sum((y_true == c) & (y_pred != c)))
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp / denom))
        supports.append(int(np.sum(y_true == c)))

    return {
        "Accuracy": accuracy,
        "Macro-F1": float(np.mean(f1s)) if f1s else float("nan"),
        "Weighted-F1": float(np.average(f1s, weights=supports)) if sum(supports) else float("nan"),
        "Micro-F1": accuracy,
    }


def pretty_metrics(m: Dict[str, float]) -> str:
    keys = ["Accuracy", "Macro-F1", "Weighted-F1", "ROC-AUC", "Micro-F1"]
    return " | ".join([f"{k}: {m.get(k, float('nan')):.4f}" for k in keys])


# -----------------------------
# Loss
# -----------------------------
def build_weighted_ce_loss(
    train_labels: np.ndarray,
    num_classes: int,
    device: str,
    label_smoothing: float = 0.0,
) -> nn.Module:
    """
    使用训练集标签计算 class weights，缓解不平衡：
    weight_c = N / (C * n_c)
    """
    counts = np.bincount(train_labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    N = counts.sum()
    weights = N / (num_classes * counts)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print(f"[INFO] class counts: {counts.tolist()}")
    print(f"[INFO] class weights: {weights.detach().cpu().numpy().round(4).tolist()}")
    print(f"[INFO] label_smoothing: {label_smoothing}")

    return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)


# -----------------------------
# Eval
# -----------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> Tuple[float, Dict[str, float]]:
    model.eval()

    losses = []
    all_probs = []
    all_true = []

    for batch in tqdm(loader, desc="eval", leave=False):
        x_xgb, x_case, y, w_xgb, w_case = extract_batch_dual(batch, device=device)

        logits, probs = model(x_xgb, x_case, w_xgb, w_case)
        loss = loss_fn(logits, y)

        losses.append(float(loss.item()))
        all_probs.append(to_numpy(probs))
        all_true.append(to_numpy(y))

    mean_loss = float(np.mean(losses)) if len(losses) else float("nan")
    y_prob = np.concatenate(all_probs, axis=0) if len(all_probs) else np.zeros((0, 0))
    y_true = np.concatenate(all_true, axis=0).astype(int) if len(all_true) else np.zeros((0,), dtype=int)

    metrics = compute_metrics_multiclass(y_true, y_prob)
    return mean_loss, metrics


# -----------------------------
# Train
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    use_amp: bool = True,
    grad_clip: Optional[float] = 1.0,
) -> float:
    model.train()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(use_amp and device.startswith("cuda"))
    )
    losses = []

    for batch in tqdm(loader, desc="train", leave=False):
        x_xgb, x_case, y, w_xgb, w_case = extract_batch_dual(batch, device=device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(use_amp and device.startswith("cuda"))
        ):
            logits, _ = model(x_xgb, x_case, w_xgb, w_case)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.item()))
    print("---------1----------")
    print(float(np.mean(losses)))
    return float(np.mean(losses)) if len(losses) else float("nan")


# -----------------------------
# Fit
# -----------------------------
@dataclass
class FitConfig:
    seed: int = 42
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 4
    num_workers: int = 2
    use_amp: bool = True
    grad_clip: float = 1.0
    patience: int = 8
    train_on_all_splits: bool = False
    train_on_train_val: bool = False
    monitor: str = "macro_f1"
    overfit_stop_loss: float = 0.0
    label_smoothing: float = 0.0
    save_dir: str = "saved_models_dl_dual"
    ckpt_name: str = "dual_gene_transformer_best.pt"
    history_csv: str = "results/genosmart_runs/history.csv"
    loss_png: str = "results/genosmart_runs/loss_curve.png"


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    device: str,
    train_labels_for_weight: np.ndarray,
    cfg: FitConfig,
):
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.save_dir, cfg.ckpt_name)

    loss_fn = build_weighted_ce_loss(
        train_labels_for_weight,
        num_classes=num_classes,
        device=device,
        label_smoothing=cfg.label_smoothing,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    if cfg.monitor not in {"macro_f1", "valid_loss", "train_loss"}:
        raise ValueError(f"Unsupported monitor: {cfg.monitor}")

    best_valid = float("inf") if cfg.monitor in {"valid_loss", "train_loss"} else -1.0
    best_epoch = -1
    bad = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            use_amp=cfg.use_amp,
            grad_clip=cfg.grad_clip,
        )

        valid_loss, valid_metrics = evaluate(model, valid_loader, loss_fn, device=device)
        if cfg.monitor == "valid_loss":
            valid_score = valid_loss
        elif cfg.monitor == "train_loss":
            valid_score = train_loss
        else:
            valid_score = valid_metrics["Macro-F1"]
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            **{f"valid_{k}": v for k, v in valid_metrics.items()},
        })
        save_history(history, cfg.history_csv, cfg.loss_png)

        dt = time.time() - t0
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f} | "
            f"{pretty_metrics(valid_metrics)} | time={dt:.1f}s"
        )

        improved = valid_score < best_valid if cfg.monitor in {"valid_loss", "train_loss"} else valid_score > best_valid
        if improved:
            best_valid = valid_score
            best_epoch = epoch
            bad = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_valid": best_valid,
                    "monitor": cfg.monitor,
                    "cfg": cfg.__dict__,
                },
                ckpt_path,
            )
            print(f"  [OK] Saved best checkpoint -> {ckpt_path} (best {cfg.monitor}={best_valid:.4f})")
        else:
            bad += 1
            if cfg.patience > 0 and bad >= cfg.patience:
                print(f"  [STOP] Early stop. best_epoch={best_epoch}, best {cfg.monitor}={best_valid:.4f}")
                break

        if (
            cfg.overfit_stop_loss > 0
            and valid_loss <= cfg.overfit_stop_loss
            and valid_metrics.get("Accuracy", 0.0) >= 0.999
        ):
            print(
                f"  [STOP] Overfit target reached: valid_loss={valid_loss:.4f}, "
                f"valid_accuracy={valid_metrics.get('Accuracy', float('nan')):.4f}"
            )
            break

    print(f"\n[TEST] Loading best checkpoint from epoch {best_epoch} ...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_loss, test_metrics = evaluate(model, test_loader, loss_fn, device=device)
    print(f"[TEST] loss={test_loss:.4f} | {pretty_metrics(test_metrics)}")

    return {
        "best_epoch": best_epoch,
        "monitor": cfg.monitor,
        "best_valid": best_valid,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "history_csv": cfg.history_csv,
        "loss_png": cfg.loss_png,
    }


def save_history(history, csv_path: str, loss_png: str = None):
    if not history:
        return

    out_dir = os.path.dirname(csv_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fieldnames = list(history[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    if not loss_png:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(loss_png), exist_ok=True)
        epochs = [r["epoch"] for r in history]
        train_loss = [r["train_loss"] for r in history]
        valid_loss = [r["valid_loss"] for r in history]

        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, train_loss, marker="o", label="Train loss")
        plt.plot(epochs, valid_loss, marker="o", label="Validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-entropy loss")
        plt.title("GenoSmart Training Curve")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_png, dpi=200)
        plt.close()
    except Exception as e:
        svg_path = os.path.splitext(loss_png)[0] + ".svg"
        save_loss_svg(history, svg_path)
        print(f"[WARN] Could not save loss curve PNG: {e}; saved SVG instead: {svg_path}")


def save_loss_svg(history, svg_path: str):
    os.makedirs(os.path.dirname(svg_path), exist_ok=True)
    epochs = np.asarray([r["epoch"] for r in history], dtype=float)
    train_loss = np.asarray([r["train_loss"] for r in history], dtype=float)
    valid_loss = np.asarray([r["valid_loss"] for r in history], dtype=float)

    width, height = 760, 460
    left, right, top, bottom = 70, 24, 30, 62
    plot_w = width - left - right
    plot_h = height - top - bottom

    all_loss = np.concatenate([train_loss, valid_loss])
    y_min = float(np.nanmin(all_loss))
    y_max = float(np.nanmax(all_loss))
    pad = max((y_max - y_min) * 0.08, 0.01)
    y_min -= pad
    y_max += pad

    x_min = float(np.nanmin(epochs))
    x_max = float(np.nanmax(epochs))
    if x_max == x_min:
        x_max = x_min + 1

    def xy(xs, ys):
        pts = []
        for x, y in zip(xs, ys):
            px = left + (x - x_min) / (x_max - x_min) * plot_w
            py = top + (y_max - y) / (y_max - y_min) * plot_h
            pts.append(f"{px:.1f},{py:.1f}")
        return " ".join(pts)

    train_pts = xy(epochs, train_loss)
    valid_pts = xy(epochs, valid_loss)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{width/2}" y="22" text-anchor="middle" font-family="Arial" font-size="16">GenoSmart Training Curve</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>
<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>
<text x="{left}" y="{height-20}" font-family="Arial" font-size="12">Epoch</text>
<text x="18" y="{top+plot_h/2}" transform="rotate(-90,18,{top+plot_h/2})" font-family="Arial" font-size="12">Loss</text>
<text x="{left-8}" y="{top+plot_h+4}" text-anchor="end" font-family="Arial" font-size="11">{y_min:.3f}</text>
<text x="{left-8}" y="{top+4}" text-anchor="end" font-family="Arial" font-size="11">{y_max:.3f}</text>
<polyline points="{train_pts}" fill="none" stroke="#1f77b4" stroke-width="2.5"/>
<polyline points="{valid_pts}" fill="none" stroke="#d62728" stroke-width="2.5"/>
<rect x="{width-190}" y="44" width="150" height="54" fill="white" stroke="#ddd"/>
<line x1="{width-178}" y1="62" x2="{width-140}" y2="62" stroke="#1f77b4" stroke-width="2.5"/>
<text x="{width-132}" y="66" font-family="Arial" font-size="12">Train loss</text>
<line x1="{width-178}" y1="84" x2="{width-140}" y2="84" stroke="#d62728" stroke-width="2.5"/>
<text x="{width-132}" y="88" font-family="Arial" font-size="12">Validation loss</text>
</svg>
"""
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg)


def compute_expr_weight_stats(dataset) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    base_datasets = dataset.datasets if isinstance(dataset, ConcatDataset) else [dataset]
    for ds in base_datasets:
        for i in range(len(ds.case_ids)):
            w_xgb = load_weights_from_meta(ds.xgb_meta_fps[i])
            w_case = load_weights_from_meta(ds.case_meta_fps[i])
            w = np.log1p(np.clip(np.concatenate([w_xgb, w_case]), 0.0, None) * 1000.0)
            rows.append(w.astype(np.float32))

    weights = np.stack(rows, axis=0)
    mean = weights.mean(axis=0).astype(np.float32)
    std = weights.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    print(
        "[INFO] expression weight normalization: "
        f"samples={weights.shape[0]}, dim={weights.shape[1]}, "
        f"mean_range=({mean.min():.4f}, {mean.max():.4f}), "
        f"std_range=({std.min():.4f}, {std.max():.4f})"
    )
    return mean, std


# -----------------------------
# Data loaders
# -----------------------------
def build_dataloaders(cfg: FitConfig):
    RAW_DIR = r"E:\workspace\Project_HCC\features_genept_ada_dualparts_globalnorm\raw_1"

    # 建议这里先不要做强标准化，先 identity
    transform_xgb = HCCTransform.identity
    transform_case = HCCTransform.identity

    train_ds = HCCDatasetDual(
        raw_dir=RAW_DIR,
        subset="train",
        num_classes=3,
        transform_xgb=transform_xgb,
        transform_case=transform_case,
        return_weights=True,
    )
    valid_ds = HCCDatasetDual(
        raw_dir=RAW_DIR,
        subset="val",
        num_classes=3,
        transform_xgb=transform_xgb,
        transform_case=transform_case,
        return_weights=True,
    )
    test_ds = HCCDatasetDual(
        raw_dir=RAW_DIR,
        subset="test",
        num_classes=3,
        transform_xgb=transform_xgb,
        transform_case=transform_case,
        return_weights=True,
    )

    train_source_ds = train_ds
    if cfg.train_on_all_splits:
        train_source_ds = ConcatDataset([train_ds, valid_ds, test_ds])
        print(
            "[INFO] Overfit sanity mode: training on train+val+test "
            f"({len(train_ds)}+{len(valid_ds)}+{len(test_ds)}={len(train_source_ds)} samples)."
        )
        print("[INFO] Validation/test loaders still evaluate the original test split.")
    elif cfg.train_on_train_val:
        train_source_ds = ConcatDataset([train_ds, valid_ds])
        print(
            "[INFO] Train+val mode: training on train+val, testing on held-out test "
            f"({len(train_ds)}+{len(valid_ds)}={len(train_source_ds)} train samples; "
            f"test={len(test_ds)} samples)."
        )
        print("[INFO] Checkpoint monitor can use train_loss; test is reported for diagnostics.")
    else:
        print(
            "[INFO] Standard mode: "
            f"train={len(train_ds)}, val={len(valid_ds)}, test={len(test_ds)} samples."
        )

    train_loader = DataLoader(
        train_source_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    valid_eval_ds = test_ds if (cfg.train_on_all_splits or cfg.train_on_train_val) else valid_ds

    valid_loader = DataLoader(
        valid_eval_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # 取训练集 labels 用于 class weight
    ys = []
    for i in range(len(train_source_ds)):
        item = train_source_ds[i]
        y = item[2]   # x_xgb, x_case, y, w_xgb, w_case, case_id
        y_idx = label_to_class_index(y).detach().cpu().numpy().astype(int).reshape(-1)
        ys.append(int(y_idx[0]))

    train_labels = np.asarray(ys, dtype=int)
    expr_mean, expr_std = compute_expr_weight_stats(train_source_ds)

    return train_loader, valid_loader, test_loader, train_labels, expr_mean, expr_std


# -----------------------------
# Main
# -----------------------------
def main():
    run_name = datetime.now().strftime("genosmart_trainval_test_%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", "genosmart_runs", run_name)
    cfg = FitConfig(
        epochs=60,
        lr=1e-3,
        weight_decay=0.0,
        batch_size=8,
        num_workers=0,
        use_amp=True,
        grad_clip=1.0,
        patience=15,
        train_on_all_splits=False,
        train_on_train_val=True,
        monitor="train_loss",
        overfit_stop_loss=0.0,
        save_dir="saved_models_dl_dual",
        ckpt_name="genosmart_trainval_best.pt",
        history_csv=os.path.join(run_dir, "history.csv"),
        loss_png=os.path.join(run_dir, "loss_curve.png"),
    )

    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")
    if device == "cuda":
        print(f"[INFO] gpu={torch.cuda.get_device_name(0)}")

    model = GenoSmart(
        in_dim=1536,
        seq_len_xgb=2000,
        seq_len_case=2000,
        num_classes=3,
        d_model=128,
        nhead=4,
        branch_layers=1,
        latent_len=64,
        latent_layers=1,
        dropout=0.0,
    ).to(device)

    train_loader, valid_loader, test_loader, train_labels, expr_mean, expr_std = build_dataloaders(cfg)
    model.set_expr_normalization(expr_mean, expr_std)

    results = fit(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        num_classes=3,
        device=device,
        train_labels_for_weight=train_labels,
        cfg=cfg,
    )

    print("\n[Done]")
    print(results)


if __name__ == "__main__":
    main()
