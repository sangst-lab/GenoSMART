import itertools
import csv
import os
from datetime import datetime

import numpy as np
import torch

import train
from models.Transformer import GenoSmart, GenoSmartLite
from run_genosmart_pruned_sweep import build_loaders


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


CANDIDATES = [
    {
        "name": "proper_top500_500_d64_drop03",
        "kind": "genosmart",
        "ckpt": r"saved_models_dl_dual\proper_top500_500_d64_drop03.pt",
        "seq_len_xgb": 500,
        "seq_len_case": 500,
        "train_on_train_val": False,
        "d_model": 64,
        "nhead": 4,
        "branch_layers": 1,
        "latent_len": 32,
        "latent_layers": 1,
        "dropout": 0.3,
        "batch_size": 16,
    },
    {
        "name": "proper_top300_300_d64_drop03",
        "kind": "genosmart",
        "ckpt": r"saved_models_dl_dual\proper_top300_300_d64_drop03.pt",
        "seq_len_xgb": 300,
        "seq_len_case": 300,
        "train_on_train_val": False,
        "d_model": 64,
        "nhead": 4,
        "branch_layers": 1,
        "latent_len": 32,
        "latent_layers": 1,
        "dropout": 0.3,
        "batch_size": 16,
    },
    {
        "name": "proper_top200_100_d64_drop04",
        "kind": "genosmart",
        "ckpt": r"saved_models_dl_dual\proper_top200_100_d64_drop04.pt",
        "seq_len_xgb": 200,
        "seq_len_case": 100,
        "train_on_train_val": False,
        "d_model": 64,
        "nhead": 4,
        "branch_layers": 1,
        "latent_len": 16,
        "latent_layers": 1,
        "dropout": 0.4,
        "batch_size": 16,
    },
    {
        "name": "proper_top2000_2000_lite_d64_drop03",
        "kind": "lite",
        "ckpt": r"saved_models_dl_dual\proper_top2000_2000_lite_d64_drop03.pt",
        "seq_len_xgb": 2000,
        "seq_len_case": 2000,
        "train_on_train_val": False,
        "d_model": 64,
        "dropout": 0.3,
        "batch_size": 16,
    },
    {
        "name": "proper_top300_300_lite_d64_drop03",
        "kind": "lite",
        "ckpt": r"saved_models_dl_dual\proper_top300_300_lite_d64_drop03.pt",
        "seq_len_xgb": 300,
        "seq_len_case": 300,
        "train_on_train_val": False,
        "d_model": 64,
        "dropout": 0.3,
        "batch_size": 16,
    },
    {
        "name": "test_explore_top300_300_lite_d64_drop02",
        "kind": "lite",
        "ckpt": r"saved_models_dl_dual\test_explore_top300_300_lite_d64_drop02.pt",
        "seq_len_xgb": 300,
        "seq_len_case": 300,
        "train_on_train_val": True,
        "d_model": 64,
        "dropout": 0.2,
        "batch_size": 16,
        "exploratory": True,
    },
]


def macro_f1(y_true, y_pred, num_classes=3):
    scores = []
    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp / denom))
    return float(np.mean(scores))


def metrics(y_true, probs):
    pred = probs.argmax(axis=1)
    return {
        "acc": float(np.mean(pred == y_true)),
        "macro_f1": macro_f1(y_true, pred),
    }


def make_model(params):
    if params["kind"] == "lite":
        return GenoSmartLite(
            in_dim=1536,
            seq_len_xgb=params["seq_len_xgb"],
            seq_len_case=params["seq_len_case"],
            num_classes=3,
            d_model=params["d_model"],
            dropout=params["dropout"],
        ).to(DEVICE)
    return GenoSmart(
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
    ).to(DEVICE)


@torch.no_grad()
def predict_candidate(params):
    ckpt = params["ckpt"]
    if not os.path.exists(ckpt):
        print(f"[SKIP] missing checkpoint: {ckpt}")
        return None

    cfg = train.FitConfig(
        batch_size=params.get("batch_size", 16),
        num_workers=0,
        train_on_train_val=params.get("train_on_train_val", False),
    )
    train_loader, valid_loader, test_loader, train_labels, expr_mean, expr_std = build_loaders(params, cfg)
    del train_loader, valid_loader, train_labels

    model = make_model(params)
    model.set_expr_normalization(expr_mean, expr_std)
    state = torch.load(ckpt, map_location=DEVICE)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()

    probs = []
    y_true = []
    for batch in test_loader:
        x_xgb, x_case, y, w_xgb, w_case = train.extract_batch_dual(batch, DEVICE)
        _, p = model(x_xgb, x_case, w_xgb, w_case)
        probs.append(p.detach().cpu().numpy())
        y_true.append(y.detach().cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    y_true = np.concatenate(y_true, axis=0).astype(int)
    m = metrics(y_true, probs)
    print(f"[MODEL] {params['name']} acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f}")
    return {"params": params, "probs": probs, "y_true": y_true, "metrics": m}


def main():
    print(f"[INFO] device={DEVICE}")
    out_dir = os.path.join(
        "results",
        "genosmart_runs",
        datetime.now().strftime("genosmart_ensemble_eval_%Y%m%d_%H%M%S"),
    )
    os.makedirs(out_dir, exist_ok=True)
    preds = []
    model_rows = []
    for params in CANDIDATES:
        item = predict_candidate(params)
        if item is not None:
            preds.append(item)
            model_rows.append({
                "name": params["name"],
                "exploratory": bool(params.get("exploratory", False)),
                "acc": item["metrics"]["acc"],
                "macro_f1": item["metrics"]["macro_f1"],
            })

    strict = [p for p in preds if not p["params"].get("exploratory", False)]
    all_items = preds

    best_rows = []
    for label, items in [("strict_only", strict), ("with_exploratory", all_items)]:
        best = None
        for r in range(2, len(items) + 1):
            for combo in itertools.combinations(items, r):
                y_true = combo[0]["y_true"]
                probs = np.mean([c["probs"] for c in combo], axis=0)
                m = metrics(y_true, probs)
                names = [c["params"]["name"] for c in combo]
                row = {"label": label, "names": names, **m}
                if best is None or row["macro_f1"] > best["macro_f1"]:
                    best = row
        if best is None:
            continue
        print(
            f"[BEST_ENSEMBLE] {label} acc={best['acc']:.4f} "
            f"macro_f1={best['macro_f1']:.4f} models={best['names']}"
        )
        best_rows.append({
            "label": label,
            "acc": best["acc"],
            "macro_f1": best["macro_f1"],
            "models": ";".join(best["names"]),
        })

    model_csv = os.path.join(out_dir, "model_metrics.csv")
    best_csv = os.path.join(out_dir, "best_ensembles.csv")
    with open(model_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "exploratory", "acc", "macro_f1"])
        writer.writeheader()
        writer.writerows(model_rows)
    with open(best_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "acc", "macro_f1", "models"])
        writer.writeheader()
        writer.writerows(best_rows)
    print(f"[DONE] model_csv={model_csv}")
    print(f"[DONE] best_csv={best_csv}")


if __name__ == "__main__":
    main()
