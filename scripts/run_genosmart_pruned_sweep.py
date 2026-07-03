import csv
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import train
from data.hcc_dataset import HCCDatasetDual, HCCTransform, load_weights_from_meta
from models.Transformer import GenoSmart


RAW_DIR = r"E:\workspace\Project_HCC\features_genept_ada_dualparts_globalnorm\raw_1"


class SlicedDualDataset(Dataset):
    def __init__(self, base_ds, seq_len_xgb, seq_len_case):
        self.base_ds = base_ds
        self.seq_len_xgb = seq_len_xgb
        self.seq_len_case = seq_len_case

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x_xgb, x_case, y, w_xgb, w_case, case_id = self.base_ds[idx]
        return (
            x_xgb[:self.seq_len_xgb],
            x_case[:self.seq_len_case],
            y,
            w_xgb[:self.seq_len_xgb],
            w_case[:self.seq_len_case],
            case_id,
        )


def make_base_ds(subset):
    return HCCDatasetDual(
        raw_dir=RAW_DIR,
        subset=subset,
        num_classes=3,
        transform_xgb=HCCTransform.identity,
        transform_case=HCCTransform.identity,
        return_weights=True,
    )


def iter_base_datasets(dataset):
    if isinstance(dataset, ConcatDataset):
        return dataset.datasets
    return [dataset]


def get_underlying_base(ds):
    return ds.base_ds if isinstance(ds, SlicedDualDataset) else ds


def compute_expr_weight_stats(dataset, seq_len_xgb, seq_len_case):
    rows = []
    for ds in iter_base_datasets(dataset):
        base_ds = get_underlying_base(ds)
        for i in range(len(base_ds.case_ids)):
            w_xgb = load_weights_from_meta(base_ds.xgb_meta_fps[i])[:seq_len_xgb]
            w_case = load_weights_from_meta(base_ds.case_meta_fps[i])[:seq_len_case]
            w = np.log1p(np.clip(np.concatenate([w_xgb, w_case]), 0.0, None) * 1000.0)
            rows.append(w.astype(np.float32))

    weights = np.stack(rows, axis=0)
    mean = weights.mean(axis=0).astype(np.float32)
    std = np.maximum(weights.std(axis=0).astype(np.float32), 1e-6)
    print(
        "[INFO] expression stats: "
        f"samples={weights.shape[0]}, dim={weights.shape[1]}, "
        f"mean_range=({mean.min():.4f}, {mean.max():.4f}), "
        f"std_range=({std.min():.4f}, {std.max():.4f})"
    )
    return mean, std


def collect_labels(dataset):
    ys = []
    for i in range(len(dataset)):
        item = dataset[i]
        y_idx = train.label_to_class_index(item[2]).detach().cpu().numpy().astype(int).reshape(-1)
        ys.append(int(y_idx[0]))
    return np.asarray(ys, dtype=int)


def build_loaders(params, cfg):
    train_base = make_base_ds("train")
    val_base = make_base_ds("val")
    test_base = make_base_ds("test")

    train_ds = SlicedDualDataset(train_base, params["seq_len_xgb"], params["seq_len_case"])
    val_ds = SlicedDualDataset(val_base, params["seq_len_xgb"], params["seq_len_case"])
    test_ds = SlicedDualDataset(test_base, params["seq_len_xgb"], params["seq_len_case"])

    if params.get("train_on_train_val", False):
        train_source = ConcatDataset([train_ds, val_ds])
        valid_source = test_ds
        print(
            "[INFO] train+val mode: "
            f"{len(train_ds)}+{len(val_ds)} train samples, test={len(test_ds)}"
        )
    else:
        train_source = train_ds
        valid_source = val_ds
        print(
            "[INFO] standard mode: "
            f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
        )

    train_loader = DataLoader(
        train_source,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_source,
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

    labels = collect_labels(train_source)
    expr_mean, expr_std = compute_expr_weight_stats(
        train_source,
        params["seq_len_xgb"],
        params["seq_len_case"],
    )
    return train_loader, valid_loader, test_loader, labels, expr_mean, expr_std


def build_model(params, device):
    model = GenoSmart(
        in_dim=1536,
        seq_len_xgb=params["seq_len_xgb"],
        seq_len_case=params["seq_len_case"],
        num_classes=3,
        d_model=params["d_model"],
        nhead=params["nhead"],
        branch_layers=params["branch_layers"],
        latent_len=params["latent_len"],
        latent_layers=params["latent_layers"],
        dropout=params["dropout"],
    ).to(device)
    return model


def run_one(params, root_dir, device):
    run_dir = os.path.join(root_dir, params["name"])
    cfg = train.FitConfig(
        seed=params.get("seed", 42),
        epochs=params["epochs"],
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        batch_size=params.get("batch_size", 8),
        num_workers=0,
        use_amp=True,
        grad_clip=1.0,
        patience=params["patience"],
        train_on_all_splits=False,
        train_on_train_val=params.get("train_on_train_val", False),
        monitor=params["monitor"],
        overfit_stop_loss=0.0,
        label_smoothing=params["label_smoothing"],
        save_dir="saved_models_dl_dual",
        ckpt_name=f"{params['name']}.pt",
        history_csv=os.path.join(run_dir, "history.csv"),
        loss_png=os.path.join(run_dir, "loss_curve.png"),
    )

    print(f"\n[RUN] {params['name']}")
    print(params)
    train.set_seed(cfg.seed)
    train_loader, valid_loader, test_loader, train_labels, expr_mean, expr_std = build_loaders(params, cfg)
    model = build_model(params, device)
    model.set_expr_normalization(expr_mean, expr_std)

    result = train.fit(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        num_classes=3,
        device=device,
        train_labels_for_weight=train_labels,
        cfg=cfg,
    )
    metrics = result["test_metrics"]
    return {
        "name": params["name"],
        "seq_len_xgb": params["seq_len_xgb"],
        "seq_len_case": params["seq_len_case"],
        "train_on_train_val": params.get("train_on_train_val", False),
        "monitor": params["monitor"],
        "best_epoch": result["best_epoch"],
        "best_valid": result["best_valid"],
        "test_loss": result["test_loss"],
        "test_accuracy": metrics.get("Accuracy"),
        "test_macro_f1": metrics.get("Macro-F1"),
        "test_weighted_f1": metrics.get("Weighted-F1"),
        "d_model": params["d_model"],
        "dropout": params["dropout"],
        "weight_decay": params["weight_decay"],
        "label_smoothing": params["label_smoothing"],
        "lr": params["lr"],
        "history_csv": result["history_csv"],
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")
    if device == "cuda":
        print(f"[INFO] gpu={torch.cuda.get_device_name(0)}")

    run_name = datetime.now().strftime("genosmart_pruned_sweep_%Y%m%d_%H%M%S")
    root_dir = os.path.join("results", "genosmart_runs", run_name)
    os.makedirs(root_dir, exist_ok=True)

    configs = [
        {
            "name": "proper_top500_500_d64_drop03",
            "seq_len_xgb": 500,
            "seq_len_case": 500,
            "train_on_train_val": False,
            "monitor": "macro_f1",
            "epochs": 35,
            "patience": 8,
            "lr": 3e-4,
            "weight_decay": 1e-2,
            "label_smoothing": 0.05,
            "d_model": 64,
            "nhead": 4,
            "branch_layers": 1,
            "latent_len": 32,
            "latent_layers": 1,
            "dropout": 0.3,
        },
        {
            "name": "proper_top300_300_d64_drop03",
            "seq_len_xgb": 300,
            "seq_len_case": 300,
            "train_on_train_val": False,
            "monitor": "macro_f1",
            "epochs": 35,
            "patience": 8,
            "lr": 3e-4,
            "weight_decay": 1e-2,
            "label_smoothing": 0.05,
            "d_model": 64,
            "nhead": 4,
            "branch_layers": 1,
            "latent_len": 32,
            "latent_layers": 1,
            "dropout": 0.3,
        },
        {
            "name": "proper_top200_100_d64_drop04",
            "seq_len_xgb": 200,
            "seq_len_case": 100,
            "train_on_train_val": False,
            "monitor": "macro_f1",
            "epochs": 40,
            "patience": 10,
            "lr": 3e-4,
            "weight_decay": 2e-2,
            "label_smoothing": 0.05,
            "d_model": 64,
            "nhead": 4,
            "branch_layers": 1,
            "latent_len": 16,
            "latent_layers": 1,
            "dropout": 0.4,
        },
        {
            "name": "trainval_top300_300_d64_drop04_short",
            "seq_len_xgb": 300,
            "seq_len_case": 300,
            "train_on_train_val": True,
            "monitor": "train_loss",
            "epochs": 12,
            "patience": 0,
            "lr": 3e-4,
            "weight_decay": 2e-2,
            "label_smoothing": 0.05,
            "d_model": 64,
            "nhead": 4,
            "branch_layers": 1,
            "latent_len": 32,
            "latent_layers": 1,
            "dropout": 0.4,
        },
    ]

    rows = []
    summary_csv = os.path.join(root_dir, "summary.csv")
    for params in configs:
        rows.append(run_one(params, root_dir, device))
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] updated summary: {summary_csv}")

    print("\n[SUMMARY]")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
