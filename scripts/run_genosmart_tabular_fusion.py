import csv
import os
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data.hcc_dataset import HCCDatasetDual, HCCTransform, load_weights_from_meta
import train


PROJECT_ROOT = r"E:\workspace\Project_HCC"
RAW_EXPR_ROOT = os.path.join(PROJECT_ROOT, "features")
TOKEN_RAW_DIR = os.path.join(PROJECT_ROOT, "features_genept_ada_dualparts_globalnorm", "raw_1")


def macro_f1_score(y_true, y_pred, num_classes=3):
    f1s = []
    for c in range(num_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(f1s))


def weighted_f1_score(y_true, y_pred, num_classes=3):
    total = len(y_true)
    out = 0.0
    for c in range(num_classes):
        support = int((y_true == c).sum())
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        denom = 2 * tp + fp + fn
        f1 = 0.0 if denom == 0 else (2 * tp) / denom
        out += (support / max(total, 1)) * f1
    return float(out)


def evaluate_logits(logits, y, loss):
    pred = logits.argmax(dim=1).detach().cpu().numpy()
    yy = y.detach().cpu().numpy()
    return {
        "loss": float(loss.detach().cpu().item()),
        "accuracy": float((pred == yy).mean()),
        "macro_f1": macro_f1_score(yy, pred),
        "weighted_f1": weighted_f1_score(yy, pred),
    }


class TabularFusionDataset(Dataset):
    def __init__(
        self,
        raw_split_dir,
        token_raw_dir,
        subset,
        gene_indices,
        seq_len_xgb=300,
        seq_len_case=300,
        raw_mean=None,
        raw_std=None,
        weight_mean=None,
        weight_std=None,
    ):
        self.subset = subset
        self.X_raw = np.load(os.path.join(raw_split_dir, f"X_{subset}.npy"), mmap_mode="r")
        self.y = np.load(os.path.join(raw_split_dir, f"y_{subset}.npy")).astype(np.int64)
        raw_ids_path = os.path.join(raw_split_dir, f"{subset}_ids.txt")
        with open(raw_ids_path, encoding="utf-8") as f:
            self.raw_ids = [line.strip() for line in f if line.strip()]
        if len(self.raw_ids) != len(self.y):
            raise ValueError(f"Raw id/label length mismatch for {subset}: {len(self.raw_ids)} vs {len(self.y)}")
        self.raw_id_to_row = {case_id: i for i, case_id in enumerate(self.raw_ids)}
        self.gene_indices = np.asarray(gene_indices, dtype=np.int64)
        self.seq_len_xgb = seq_len_xgb
        self.seq_len_case = seq_len_case
        self.raw_mean = raw_mean
        self.raw_std = raw_std
        self.weight_mean = weight_mean
        self.weight_std = weight_std
        self.token_ds = HCCDatasetDual(
            raw_dir=token_raw_dir,
            subset=subset,
            num_classes=3,
            transform_xgb=HCCTransform.identity,
            transform_case=HCCTransform.identity,
            return_weights=True,
        )
        if len(self.token_ds) != len(self.y):
            raise ValueError(f"Token/raw length mismatch for {subset}: {len(self.token_ds)} vs {len(self.y)}")
        missing_ids = [case_id for case_id in self.token_ds.case_ids if case_id not in self.raw_id_to_row]
        if missing_ids:
            raise ValueError(f"Token ids missing from raw {subset}: {missing_ids[:5]}")

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        case_id = self.token_ds.case_ids[idx]
        raw_idx = self.raw_id_to_row[case_id]
        raw = np.asarray(self.X_raw[raw_idx, self.gene_indices], dtype=np.float32)
        raw = np.log2(np.clip(raw, 0.0, None) + 1.0)
        if self.raw_mean is not None:
            raw = (raw - self.raw_mean) / self.raw_std

        w_xgb = load_weights_from_meta(self.token_ds.xgb_meta_fps[idx])[: self.seq_len_xgb]
        w_case = load_weights_from_meta(self.token_ds.case_meta_fps[idx])[: self.seq_len_case]
        weights = np.log1p(np.clip(np.concatenate([w_xgb, w_case]), 0.0, None) * 1000.0).astype(np.float32)
        if self.weight_mean is not None:
            weights = (weights - self.weight_mean) / self.weight_std

        return (
            torch.tensor(raw, dtype=torch.float32),
            torch.tensor(weights, dtype=torch.float32),
            torch.tensor(int(self.y[raw_idx]), dtype=torch.long),
        )


class FusionMLP(nn.Module):
    def __init__(self, raw_dim, weight_dim, num_classes=3, hidden=256, dropout=0.25, mode="raw_weight"):
        super().__init__()
        self.mode = mode
        in_dim = 0
        if mode in ("raw", "raw_weight"):
            in_dim += raw_dim
        if mode in ("weight", "raw_weight"):
            in_dim += weight_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, raw, weights):
        xs = []
        if self.mode in ("raw", "raw_weight"):
            xs.append(raw)
        if self.mode in ("weight", "raw_weight"):
            xs.append(weights)
        return self.net(torch.cat(xs, dim=1))


def load_top_gene_indices(top_n):
    path = os.path.join(PROJECT_ROOT, "results", "feature_importance", "combined_raw1_raw2_raw3_TOP2000.csv")
    names_path = os.path.join(RAW_EXPR_ROOT, "raw_1", "gene_feature_names.txt")
    with open(names_path, encoding="utf-8") as f:
        gene_to_idx = {line.strip(): i for i, line in enumerate(f) if line.strip()}

    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature_id = row.get("feature", "")
            if not feature_id:
                continue
            idx = gene_to_idx.get(feature_id)
            if idx is None:
                continue
            rows.append(idx)
            if len(rows) >= top_n:
                break
    if len(rows) < top_n:
        raise ValueError(f"Only found {len(rows)} feature indices in {path}, need {top_n}")
    return rows


def compute_stats(ds):
    raws, weights = [], []
    for i in range(len(ds)):
        raw, w, _ = ds[i]
        raws.append(raw.numpy())
        weights.append(w.numpy())
    raw_arr = np.stack(raws, axis=0).astype(np.float32)
    weight_arr = np.stack(weights, axis=0).astype(np.float32)
    return (
        raw_arr.mean(axis=0),
        np.maximum(raw_arr.std(axis=0), 1e-6),
        weight_arr.mean(axis=0),
        np.maximum(weight_arr.std(axis=0), 1e-6),
    )


def make_loaders(raw_id, top_n, seq_len_xgb, seq_len_case, batch_size):
    raw_split_dir = os.path.join(RAW_EXPR_ROOT, f"raw_{raw_id}")
    gene_indices = load_top_gene_indices(top_n)
    train_plain = TabularFusionDataset(raw_split_dir, TOKEN_RAW_DIR, "train", gene_indices, seq_len_xgb, seq_len_case)
    raw_mean, raw_std, weight_mean, weight_std = compute_stats(train_plain)

    train_ds = TabularFusionDataset(
        raw_split_dir, TOKEN_RAW_DIR, "train", gene_indices, seq_len_xgb, seq_len_case, raw_mean, raw_std, weight_mean, weight_std
    )
    val_ds = TabularFusionDataset(
        raw_split_dir, TOKEN_RAW_DIR, "val", gene_indices, seq_len_xgb, seq_len_case, raw_mean, raw_std, weight_mean, weight_std
    )
    test_ds = TabularFusionDataset(
        raw_split_dir, TOKEN_RAW_DIR, "test", gene_indices, seq_len_xgb, seq_len_case, raw_mean, raw_std, weight_mean, weight_std
    )
    loaders = [
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True),
    ]
    labels = np.asarray([int(train_ds.y[i]) for i in range(len(train_ds))], dtype=np.int64)
    print(
        f"[DATA] raw_{raw_id} top_n={top_n} train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"labels={dict(Counter(labels.tolist()))}"
    )
    return loaders, labels


def run_epoch(model, loader, optimizer, device, class_weight=None, label_smoothing=0.0):
    model.train()
    total_loss = 0.0
    total_n = 0
    for raw, weights, y in loader:
        raw = raw.to(device)
        weights = weights.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(raw, weights)
        loss = F.cross_entropy(logits, y, weight=class_weight, label_smoothing=label_smoothing)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu().item()) * y.size(0)
        total_n += y.size(0)
    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device, class_weight=None):
    model.eval()
    all_logits, all_y = [], []
    total_loss = 0.0
    total_n = 0
    for raw, weights, y in loader:
        raw = raw.to(device)
        weights = weights.to(device)
        y = y.to(device)
        logits = model(raw, weights)
        loss = F.cross_entropy(logits, y, weight=class_weight)
        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())
        total_loss += float(loss.detach().cpu().item()) * y.size(0)
        total_n += y.size(0)
    logits = torch.cat(all_logits, dim=0)
    y = torch.cat(all_y, dim=0)
    metrics = evaluate_logits(logits, y, torch.tensor(total_loss / max(total_n, 1)))
    return metrics


def make_class_weight(labels, device):
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_one(cfg, root_dir, device):
    train.set_seed(cfg["seed"])
    (train_loader, val_loader, test_loader), labels = make_loaders(
        cfg["raw_id"], cfg["top_n"], cfg["seq_len_xgb"], cfg["seq_len_case"], cfg["batch_size"]
    )
    model = FusionMLP(
        raw_dim=cfg["top_n"],
        weight_dim=cfg["seq_len_xgb"] + cfg["seq_len_case"],
        hidden=cfg["hidden"],
        dropout=cfg["dropout"],
        mode=cfg["mode"],
    ).to(device)
    class_weight = make_class_weight(labels, device) if cfg["class_weight"] else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    run_dir = os.path.join(root_dir, cfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    history_path = os.path.join(run_dir, "history.csv")
    best_state = None
    best_val = -1.0
    best_epoch = 0
    patience_left = cfg["patience"]
    rows = []

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss = run_epoch(model, train_loader, optimizer, device, class_weight, cfg["label_smoothing"])
        tr = evaluate(model, train_loader, device, class_weight)
        va = evaluate(model, val_loader, device, class_weight)
        te = evaluate(model, test_loader, device, class_weight)
        row = {
            "epoch": epoch,
            "train_loss_step": tr_loss,
            "train_loss": tr["loss"],
            "train_acc": tr["accuracy"],
            "train_macro_f1": tr["macro_f1"],
            "val_loss": va["loss"],
            "val_acc": va["accuracy"],
            "val_macro_f1": va["macro_f1"],
            "test_loss": te["loss"],
            "test_acc": te["accuracy"],
            "test_macro_f1": te["macro_f1"],
        }
        rows.append(row)
        with open(history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"[{cfg['name']}] epoch={epoch:03d} tr_f1={tr['macro_f1']:.3f} "
            f"val_f1={va['macro_f1']:.3f} test_f1={te['macro_f1']:.3f}"
        )
        if va["macro_f1"] > best_val:
            best_val = va["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    tr = evaluate(model, train_loader, device, class_weight)
    va = evaluate(model, val_loader, device, class_weight)
    te = evaluate(model, test_loader, device, class_weight)
    torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))
    return {
        "name": cfg["name"],
        "raw_id": cfg["raw_id"],
        "mode": cfg["mode"],
        "top_n": cfg["top_n"],
        "seed": cfg["seed"],
        "best_epoch": best_epoch,
        "train_acc": tr["accuracy"],
        "train_macro_f1": tr["macro_f1"],
        "val_acc": va["accuracy"],
        "val_macro_f1": va["macro_f1"],
        "test_acc": te["accuracy"],
        "test_macro_f1": te["macro_f1"],
        "test_weighted_f1": te["weighted_f1"],
        "history_csv": history_path,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")
    if device == "cuda":
        print(f"[INFO] gpu={torch.cuda.get_device_name(0)}")
    root_dir = os.path.join(PROJECT_ROOT, "results", "genosmart_runs", datetime.now().strftime("genosmart_tabular_fusion_%Y%m%d_%H%M%S"))
    os.makedirs(root_dir, exist_ok=True)

    base = {
        "raw_id": 1,
        "seq_len_xgb": 300,
        "seq_len_case": 300,
        "batch_size": 16,
        "epochs": 80,
        "patience": 12,
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "label_smoothing": 0.03,
        "hidden": 256,
        "dropout": 0.25,
        "class_weight": True,
        "seed": 42,
    }
    configs = []
    for top_n in [100, 300, 800]:
        for mode in ["raw", "weight", "raw_weight"]:
            cfg = dict(base)
            cfg["top_n"] = top_n
            cfg["mode"] = mode
            cfg["name"] = f"raw1_{mode}_top{top_n}_seed42"
            configs.append(cfg)

    summary_path = os.path.join(root_dir, "summary.csv")
    rows = []
    for cfg in configs:
        rows.append(run_one(cfg, root_dir, device))
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[SUMMARY UPDATED] {summary_path}")

    print("[DONE]", summary_path)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
