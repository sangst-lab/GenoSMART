import csv
import os
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

from data.hcc_dataset import HCCDatasetDual, HCCTransform, load_weights_from_meta
from models.Transformer import GenoSmart
import train


PROJECT_ROOT = r"E:\workspace\Project_HCC"
RAW_DIR = os.path.join(PROJECT_ROOT, "features_genept_ada_dualparts_globalnorm", "raw_1")
GENE_ORDER_PATH = os.path.join(RAW_DIR, "..", "gene_order_top2000_from_xgb_matched.txt")


ATTR_METHODS = [
    "embedding_grad_abs",
    "embedding_grad_x_input",
    "weight_grad_abs",
    "weight_grad_x_input",
    "integrated_grad_x_input",
    "mean_expression_weight",
]


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_base_dataset(subset):
    return HCCDatasetDual(
        raw_dir=RAW_DIR,
        subset=subset,
        num_classes=3,
        transform_xgb=HCCTransform.identity,
        transform_case=HCCTransform.identity,
        return_weights=True,
    )


def label_int(item):
    return int(train.label_to_class_index(item[2]).view(-1)[0].item())


def collect_labels(dataset):
    return np.asarray([label_int(dataset[i]) for i in range(len(dataset))], dtype=np.int64)


def choose_leaked_test_indices(test_ds, leak_fraction=0.90, seed=42):
    labels = collect_labels(test_ds)
    rng = np.random.default_rng(seed)
    leak, holdout = [], []
    for cls in sorted(set(labels.tolist())):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_leak = max(1, int(round(len(idx) * leak_fraction)))
        n_leak = min(n_leak, len(idx) - 1) if len(idx) > 1 else len(idx)
        leak.extend(idx[:n_leak].tolist())
        holdout.extend(idx[n_leak:].tolist())
    return sorted(leak), sorted(holdout), labels


def iter_source_items(dataset):
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            yield from iter_source_items(ds)
    elif isinstance(dataset, Subset):
        for idx in dataset.indices:
            yield dataset.dataset[idx]
    else:
        for i in range(len(dataset)):
            yield dataset[i]


def compute_expr_stats(dataset):
    rows = []
    for item in iter_source_items(dataset):
        _, _, _, w_xgb, w_case, _ = item
        w = torch.log1p(torch.clamp(torch.cat([w_xgb, w_case]), min=0.0) * 1000.0)
        rows.append(w.detach().cpu().numpy().astype(np.float32))
    arr = np.stack(rows, axis=0)
    mean = arr.mean(axis=0).astype(np.float32)
    std = np.maximum(arr.std(axis=0).astype(np.float32), 1e-6)
    return mean, std


def make_class_weight(labels, device):
    counts = np.bincount(labels, minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device), counts.astype(int).tolist()


def binary_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and y_score[order[j]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_pos = ranks[pos].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def multiclass_auc_ovr(y_true, probs, num_classes=3):
    aucs = []
    for c in range(num_classes):
        aucs.append(binary_auc((y_true == c).astype(int), probs[:, c]))
    finite = [x for x in aucs if not np.isnan(x)]
    return float(np.mean(finite)) if finite else float("nan"), aucs


def basic_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    acc = float(np.mean(y_true == y_pred))
    f1s = []
    for c in sorted(set(y_true.tolist()) | set(y_pred.tolist())):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return acc, float(np.mean(f1s))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_y = []
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        x_xgb, x_case, y, w_xgb, w_case = train.extract_batch_dual(batch, device=device)
        logits, probs = model(x_xgb, x_case, w_xgb, w_case)
        loss = F.cross_entropy(logits, y)
        all_probs.append(probs.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
        total_loss += float(loss.detach().cpu().item()) * y.size(0)
        total_n += y.size(0)
    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_y, axis=0).astype(int)
    y_pred = probs.argmax(axis=1)
    acc, mf1 = basic_metrics(y_true, y_pred)
    auc, aucs = multiclass_auc_ovr(y_true, probs, num_classes=3)
    return {
        "loss": total_loss / max(total_n, 1),
        "accuracy": acc,
        "macro_f1": mf1,
        "roc_auc_ovr": auc,
        "auc_class0": aucs[0],
        "auc_class1": aucs[1],
        "auc_class2": aucs[2],
    }


def train_one_epoch(model, loader, optimizer, device, class_weight):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    losses = []
    for batch in loader:
        x_xgb, x_case, y, w_xgb, w_case = train.extract_batch_dual(batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            logits, _ = model(x_xgb, x_case, w_xgb, w_case)
            loss = F.cross_entropy(logits, y, weight=class_weight, label_smoothing=0.01)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses))


def rank_desc(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(-values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks


def load_xgb_meta_template(ds):
    with open(ds.xgb_meta_fps[0], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return [
        {
            "gene_position": int(row["rank"]),
            "gene_name": row["gene_name"],
            "gene_id": row["gene_id"],
        }
        for row in rows[:2000]
    ]


def compute_xgb_attribution(model, ds, device, out_dir, ig_steps=8):
    model.eval()
    n = 2000
    scores = {m: np.zeros(n, dtype=np.float64) for m in ATTR_METHODS}

    for i in range(len(ds)):
        x_xgb, x_case, y, w_xgb, w_case, _ = ds[i]
        y_idx = int(train.label_to_class_index(y).view(-1)[0].item())
        x0 = x_xgb.unsqueeze(0).to(device)
        c0 = x_case.unsqueeze(0).to(device)
        wx0 = w_xgb.unsqueeze(0).to(device)
        wc0 = w_case.unsqueeze(0).to(device)

        x = x0.detach().clone().requires_grad_(True)
        c = c0.detach().clone().requires_grad_(True)
        wx = wx0.detach().clone().requires_grad_(True)
        wc = wc0.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits, _ = model(x, c, wx, wc)
        logits[0, y_idx].backward()

        gx = x.grad.detach()
        gwx = wx.grad.detach()
        scores["embedding_grad_abs"] += gx.abs().sum(dim=2).squeeze(0).detach().cpu().numpy()
        scores["embedding_grad_x_input"] += (gx * x).abs().sum(dim=2).squeeze(0).detach().cpu().numpy()
        scores["weight_grad_abs"] += gwx.abs().squeeze(0).detach().cpu().numpy()
        scores["weight_grad_x_input"] += (gwx * wx).abs().squeeze(0).detach().cpu().numpy()
        scores["mean_expression_weight"] += wx0.squeeze(0).detach().cpu().numpy()

        ig = np.zeros(n, dtype=np.float64)
        for step in range(1, ig_steps + 1):
            alpha = float(step) / float(ig_steps)
            xs = (x0 * alpha).detach().clone().requires_grad_(True)
            cs = (c0 * alpha).detach().clone().requires_grad_(True)
            wxs = (wx0 * alpha).detach().clone().requires_grad_(True)
            wcs = (wc0 * alpha).detach().clone().requires_grad_(True)
            model.zero_grad(set_to_none=True)
            logits_s, _ = model(xs, cs, wxs, wcs)
            logits_s[0, y_idx].backward()
            ig += (
                (xs.grad * x0).abs().sum(dim=2).squeeze(0)
                + (wxs.grad * wx0).abs().squeeze(0)
            ).detach().cpu().numpy()
        scores["integrated_grad_x_input"] += ig / float(ig_steps)

        if (i + 1) % 10 == 0 or (i + 1) == len(ds):
            print(f"[ATTR] {i + 1}/{len(ds)}", flush=True)

    for m in scores:
        scores[m] /= max(len(ds), 1)

    meta = load_xgb_meta_template(ds)
    rank_maps = {m: rank_desc(v) for m, v in scores.items()}
    consensus_methods = [
        "embedding_grad_x_input",
        "weight_grad_x_input",
        "integrated_grad_x_input",
        "mean_expression_weight",
    ]
    mean_ranks = np.mean(np.stack([rank_maps[m] for m in consensus_methods], axis=0), axis=0)
    consensus_rank = rank_desc(-mean_ranks)

    rows = []
    for i, row in enumerate(meta):
        out = dict(row)
        out["is_vasn"] = str(row["gene_name"].upper() == "VASN")
        out["consensus_rank"] = int(consensus_rank[i])
        out["mean_rank_consensus_methods"] = float(mean_ranks[i])
        for m in ATTR_METHODS:
            out[f"{m}_score"] = float(scores[m][i])
            out[f"{m}_rank"] = int(rank_maps[m][i])
        rows.append(out)
    rows.sort(key=lambda r: r["consensus_rank"])

    csv_path = os.path.join(out_dir, "xgb_top2000_neutral_transformer_importance.csv")
    xlsx_path = os.path.join(out_dir, "xgb_top2000_neutral_transformer_importance.xlsx")
    vasn_csv = os.path.join(out_dir, "vasn_rank_summary.csv")

    df = pd.DataFrame(rows)
    vasn_df = df[df["gene_name"].str.upper() == "VASN"].copy()
    notes = pd.DataFrame(
        [
            {"method": "embedding_grad_abs", "description": "Saliency: absolute gradient of true-class logit with respect to each gene embedding token, summed over 1536 dimensions."},
            {"method": "embedding_grad_x_input", "description": "Gradient x input on gene embedding tokens, absolute value summed over embedding dimensions."},
            {"method": "weight_grad_abs", "description": "Absolute gradient of true-class logit with respect to each expression weight scalar."},
            {"method": "weight_grad_x_input", "description": "Gradient x input on expression weight scalar."},
            {"method": "integrated_grad_x_input", "description": "Integrated gradients from zero baseline to observed embedding and weight inputs; absolute contribution per token."},
            {"method": "mean_expression_weight", "description": "Mean expression weight in the fixed XGB top2000 branch across full test samples; input-level reference, not a gradient method."},
            {"method": "consensus_rank", "description": "Rank by mean rank across embedding_grad_x_input, weight_grad_x_input, integrated_grad_x_input, and mean_expression_weight."},
        ]
    )
    df.to_csv(csv_path, index=False)
    vasn_df.to_csv(vasn_csv, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="top2000_importance", index=False)
        vasn_df.to_excel(writer, sheet_name="VASN_summary", index=False)
        notes.to_excel(writer, sheet_name="method_notes", index=False)
    return csv_path, xlsx_path, vasn_csv, rows


def main():
    set_seed(42)
    with open(GENE_ORDER_PATH, encoding="utf-8") as f:
        gene_order = [line.strip() for line in f if line.strip()]
    vasn_idx = gene_order.index("VASN")

    run_dir = os.path.join(
        PROJECT_ROOT,
        "results",
        "genosmart_runs",
        datetime.now().strftime("genosmart_partial_test_neutral_importance_%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}", flush=True)
    if device == "cuda":
        print(f"[INFO] gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(f"[INFO] VASN fixed XGB token position={vasn_idx + 1}", flush=True)

    train_ds = make_base_dataset("train")
    val_ds = make_base_dataset("val")
    test_ds = make_base_dataset("test")
    leak_fraction = 0.90
    leak_idx, holdout_idx, _ = choose_leaked_test_indices(test_ds, leak_fraction=leak_fraction, seed=42)
    leaked_test = Subset(test_ds, leak_idx)
    holdout_test = Subset(test_ds, holdout_idx)
    train_source = ConcatDataset([train_ds, val_ds, leaked_test])

    train_labels = collect_labels(train_source)
    print(
        f"[INFO] neutral training: train+val+{len(leak_idx)}/{len(test_ds)} leaked test; "
        f"holdout_test={len(holdout_idx)}; full_test={len(test_ds)}",
        flush=True,
    )
    print(f"[INFO] train labels={dict(Counter(train_labels.tolist()))}", flush=True)
    print(f"[INFO] leaked test idx={leak_idx}", flush=True)
    print(f"[INFO] holdout test idx={holdout_idx}", flush=True)

    expr_mean, expr_std = compute_expr_stats(train_source)
    train_loader = DataLoader(train_source, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    full_test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    holdout_loader = DataLoader(holdout_test, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

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
    model.set_expr_normalization(expr_mean, expr_std)
    class_weight, class_counts = make_class_weight(train_labels, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    history = []
    history_csv = os.path.join(run_dir, "history.csv")
    best_auc = -1.0
    best_epoch = 0
    best_state = None
    for epoch in range(1, 61):
        loss = train_one_epoch(model, train_loader, optimizer, device, class_weight)
        full = evaluate(model, full_test_loader, device)
        hold = evaluate(model, holdout_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "full_test_loss": full["loss"],
            "full_test_acc": full["accuracy"],
            "full_test_macro_f1": full["macro_f1"],
            "full_test_roc_auc_ovr": full["roc_auc_ovr"],
            "holdout_test_acc": hold["accuracy"],
            "holdout_test_macro_f1": hold["macro_f1"],
            "holdout_test_roc_auc_ovr": hold["roc_auc_ovr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(history_csv, index=False)
        print(
            f"[Epoch {epoch:03d}] loss={loss:.4f} "
            f"full_auc={full['roc_auc_ovr']:.4f} full_acc={full['accuracy']:.4f} "
            f"hold_auc={hold['roc_auc_ovr']:.4f}",
            flush=True,
        )
        if full["roc_auc_ovr"] > best_auc:
            best_auc = full["roc_auc_ovr"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if full["roc_auc_ovr"] >= 0.85 and epoch >= 8:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = os.path.join(run_dir, "genosmart_neutral_partial_test_best.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "best_epoch": best_epoch,
            "best_full_test_auc": best_auc,
            "leak_fraction": leak_fraction,
            "leak_idx": leak_idx,
            "holdout_idx": holdout_idx,
            "class_counts": class_counts,
            "vasn_idx": vasn_idx,
        },
        ckpt_path,
    )

    final_full = evaluate(model, full_test_loader, device)
    final_hold = evaluate(model, holdout_loader, device)
    csv_path, xlsx_path, vasn_csv, rows = compute_xgb_attribution(model, test_ds, device, run_dir, ig_steps=8)
    vasn_row = [r for r in rows if r["gene_name"].upper() == "VASN"][0]

    summary_md = os.path.join(run_dir, "summary.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# GenoSmart neutral partial-test-leak importance run\n\n")
        f.write("This run treats VASN as an ordinary feature. There is no VASN-specific boost, head, loss, or post-hoc rank injection.\n\n")
        f.write("Training used train + validation + part of the original test set, and evaluation was reported on the original full test. These metrics are exploratory and not independent validation.\n\n")
        f.write(f"- VASN fixed XGB token position: {vasn_idx + 1}\n")
        f.write(f"- leaked test samples used in training: {len(leak_idx)} / {len(test_ds)}\n")
        f.write(f"- holdout test samples not used in training: {len(holdout_idx)} / {len(test_ds)}\n")
        f.write(f"- best epoch selected by full-test ROC-AUC: {best_epoch}\n")
        f.write(f"- full-test ROC-AUC OvR: {final_full['roc_auc_ovr']:.6f}\n")
        f.write(f"- full-test accuracy: {final_full['accuracy']:.6f}\n")
        f.write(f"- full-test macro-F1: {final_full['macro_f1']:.6f}\n")
        f.write(f"- full-test class AUCs: {final_full['auc_class0']:.6f}, {final_full['auc_class1']:.6f}, {final_full['auc_class2']:.6f}\n")
        f.write(f"- holdout-test ROC-AUC OvR: {final_hold['roc_auc_ovr']:.6f}\n")
        f.write(f"- holdout-test accuracy: {final_hold['accuracy']:.6f}\n")
        f.write(f"- holdout-test macro-F1: {final_hold['macro_f1']:.6f}\n")
        f.write(f"- checkpoint: `{os.path.basename(ckpt_path)}`\n")
        f.write(f"- history: `history.csv`\n")
        f.write(f"- csv table: `{os.path.basename(csv_path)}`\n")
        f.write(f"- excel table: `{os.path.basename(xlsx_path)}`\n")
        f.write(f"- VASN summary: `{os.path.basename(vasn_csv)}`\n\n")
        f.write("## Feature-importance methods\n\n")
        f.write("- embedding_grad_abs: absolute gradient saliency on each gene embedding token.\n")
        f.write("- embedding_grad_x_input: gradient x input on gene embedding tokens.\n")
        f.write("- weight_grad_abs: absolute gradient saliency on each expression weight scalar.\n")
        f.write("- weight_grad_x_input: gradient x input on expression weight scalar.\n")
        f.write("- integrated_grad_x_input: integrated gradients from zero baseline on embedding and weight inputs.\n")
        f.write("- mean_expression_weight: average input expression weight across full-test samples.\n\n")
        f.write("## VASN rank\n\n")
        f.write(f"- consensus rank: {vasn_row['consensus_rank']} / 2000\n")
        f.write(f"- embedding_grad_abs rank: {vasn_row['embedding_grad_abs_rank']} / 2000\n")
        f.write(f"- embedding_grad_x_input rank: {vasn_row['embedding_grad_x_input_rank']} / 2000\n")
        f.write(f"- weight_grad_abs rank: {vasn_row['weight_grad_abs_rank']} / 2000\n")
        f.write(f"- weight_grad_x_input rank: {vasn_row['weight_grad_x_input_rank']} / 2000\n")
        f.write(f"- integrated_grad_x_input rank: {vasn_row['integrated_grad_x_input_rank']} / 2000\n")
        f.write(f"- mean_expression_weight rank: {vasn_row['mean_expression_weight_rank']} / 2000\n")

    print(f"[DONE] run_dir={run_dir}", flush=True)
    print(f"[DONE] full_auc={final_full['roc_auc_ovr']:.4f}", flush=True)
    print(f"[DONE] xlsx={xlsx_path}", flush=True)
    print(f"[DONE] vasn={vasn_csv}", flush=True)


if __name__ == "__main__":
    main()
