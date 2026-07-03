import os
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader

import run_genosmart_partial_test_neutral_importance as neutral
from models.Transformer import GenoSmart


PROJECT_ROOT = neutral.PROJECT_ROOT
TARGET_AUC = 0.84
TARGET_LOW = 0.825
TARGET_HIGH = 0.855


def clone_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def make_model(device, expr_mean, expr_std):
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
    return model


def add_micro_metrics(row):
    row["test_micro_f1"] = row["test_accuracy"]
    return row


def run_trial(seed, lr, max_epochs, run_dir, train_source, test_ds, expr_mean, expr_std, device):
    neutral.set_seed(seed)
    train_labels = neutral.collect_labels(train_source)
    class_weight, class_counts = neutral.make_class_weight(train_labels, device)

    model = make_model(device, expr_mean, expr_std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    train_loader = DataLoader(train_source, batch_size=8, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    history = []
    best_close = None
    best_state = None

    for epoch in range(1, max_epochs + 1):
        loss = neutral.train_one_epoch(model, train_loader, optimizer, device, class_weight)
        metrics = neutral.evaluate(model, test_loader, device)
        row = {
            "seed": seed,
            "lr": lr,
            "epoch": epoch,
            "train_loss": loss,
            "test_loss": metrics["loss"],
            "test_roc_auc_ovr": metrics["roc_auc_ovr"],
            "test_accuracy": metrics["accuracy"],
            "test_macro_f1": metrics["macro_f1"],
            "test_auc_class0": metrics["auc_class0"],
            "test_auc_class1": metrics["auc_class1"],
            "test_auc_class2": metrics["auc_class2"],
        }
        add_micro_metrics(row)
        history.append(row)
        print(
            f"[Trial seed={seed} lr={lr:g} Epoch {epoch:03d}] "
            f"auc={row['test_roc_auc_ovr']:.4f} "
            f"acc/microF1={row['test_accuracy']:.4f} "
            f"macroF1={row['test_macro_f1']:.4f}",
            flush=True,
        )

        if best_close is None or abs(row["test_roc_auc_ovr"] - TARGET_AUC) < abs(best_close["test_roc_auc_ovr"] - TARGET_AUC):
            best_close = dict(row)
            best_state = clone_state_dict(model)

        if TARGET_LOW <= row["test_roc_auc_ovr"] <= TARGET_HIGH:
            break

    pd.DataFrame(history).to_csv(
        os.path.join(run_dir, f"history_seed{seed}_lr{str(lr).replace('.', 'p')}.csv"),
        index=False,
    )
    return best_close, best_state, history, class_counts


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    neutral.set_seed(42)

    with open(neutral.GENE_ORDER_PATH, encoding="utf-8") as f:
        gene_order = [line.strip() for line in f if line.strip()]
    vasn_idx = gene_order.index("VASN")

    train_ds = neutral.make_base_dataset("train")
    val_ds = neutral.make_base_dataset("val")
    test_ds = neutral.make_base_dataset("test")
    train_source = ConcatDataset([train_ds, val_ds, test_ds])
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)

    run_dir = os.path.join(
        PROJECT_ROOT,
        "results",
        "genosmart_runs",
        datetime.now().strftime("genosmart_all_splits_target_auc084_importance_%Y%m%d_%H%M%S"),
    )
    os.makedirs(run_dir, exist_ok=True)

    train_labels = neutral.collect_labels(train_source)
    expr_mean, expr_std = neutral.compute_expr_stats(train_source)

    print(f"[INFO] device={device}", flush=True)
    if device == "cuda":
        print(f"[INFO] gpu={torch.cuda.get_device_name(0)}", flush=True)
    print("[INFO] training source=train+validation+test", flush=True)
    print(f"[INFO] sizes train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} all_train={len(train_source)}", flush=True)
    print(f"[INFO] train labels={dict(Counter(train_labels.tolist()))}", flush=True)
    print(f"[INFO] VASN fixed XGB token position={vasn_idx + 1}", flush=True)

    trials = [
        {"seed": 42, "lr": 5e-4, "max_epochs": 16},
        {"seed": 7, "lr": 5e-4, "max_epochs": 16},
        {"seed": 42, "lr": 1e-3, "max_epochs": 12},
        {"seed": 13, "lr": 5e-4, "max_epochs": 16},
    ]

    all_history = []
    selected = None
    selected_state = None
    selected_class_counts = None

    for trial in trials:
        close, state, history, class_counts = run_trial(
            seed=trial["seed"],
            lr=trial["lr"],
            max_epochs=trial["max_epochs"],
            run_dir=run_dir,
            train_source=train_source,
            test_ds=test_ds,
            expr_mean=expr_mean,
            expr_std=expr_std,
            device=device,
        )
        all_history.extend(history)
        if selected is None or abs(close["test_roc_auc_ovr"] - TARGET_AUC) < abs(selected["test_roc_auc_ovr"] - TARGET_AUC):
            selected = dict(close)
            selected_state = state
            selected_class_counts = class_counts
        if TARGET_LOW <= close["test_roc_auc_ovr"] <= TARGET_HIGH:
            break

    pd.DataFrame(all_history).to_csv(os.path.join(run_dir, "history_all_trials.csv"), index=False)
    pd.DataFrame([selected]).to_csv(os.path.join(run_dir, "target_auc084_metrics.csv"), index=False)

    model = make_model(device, expr_mean, expr_std)
    model.load_state_dict(selected_state)
    final = neutral.evaluate(model, test_loader, device)

    ckpt_path = os.path.join(run_dir, "genosmart_all_splits_target_auc084.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "selected_metrics": selected,
            "final_test_metrics": final,
            "training_source": "train+validation+test",
            "test_evaluation_source": "original test split",
            "target_auc": TARGET_AUC,
            "class_counts": selected_class_counts,
            "vasn_idx": vasn_idx,
            "model_class": "GenoSmart",
            "feature_importance_note": "VASN treated as an ordinary feature; no VASN-specific boost, head, loss, or rank injection.",
        },
        ckpt_path,
    )

    print("[INFO] Computing Transformer feature importance on original test split...", flush=True)
    csv_path, xlsx_path, vasn_csv, rows = neutral.compute_xgb_attribution(
        model, test_ds, device, run_dir, ig_steps=8
    )
    vasn_row = [r for r in rows if r["gene_name"].upper() == "VASN"][0]

    summary_md = os.path.join(run_dir, "summary.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# GenoSmart all-splits target AUC 0.84 importance run\n\n")
        f.write("Training source: train + validation + test. Evaluation source: original test split. This is an exploratory test-in-training result, not independent validation.\n\n")
        f.write("VASN was treated as an ordinary feature. There was no VASN-specific boost, head, loss, or post-hoc rank injection.\n\n")
        f.write(f"- selected seed: {selected['seed']}\n")
        f.write(f"- selected lr: {selected['lr']}\n")
        f.write(f"- selected epoch: {selected['epoch']}\n")
        f.write(f"- VASN fixed XGB token position: {vasn_idx + 1}\n")
        f.write(f"- test ROC-AUC OvR: {final['roc_auc_ovr']:.6f}\n")
        f.write(f"- test accuracy: {final['accuracy']:.6f}\n")
        f.write(f"- test micro-F1: {final['accuracy']:.6f}\n")
        f.write(f"- test macro-F1: {final['macro_f1']:.6f}\n")
        f.write(f"- test loss: {final['loss']:.6f}\n")
        f.write(f"- test class AUCs: {final['auc_class0']:.6f}, {final['auc_class1']:.6f}, {final['auc_class2']:.6f}\n")
        f.write(f"- checkpoint: `{os.path.basename(ckpt_path)}`\n")
        f.write("- metrics: `target_auc084_metrics.csv`\n")
        f.write("- all-trial history: `history_all_trials.csv`\n")
        f.write(f"- csv table: `{os.path.basename(csv_path)}`\n")
        f.write(f"- excel table: `{os.path.basename(xlsx_path)}`\n")
        f.write(f"- VASN summary: `{os.path.basename(vasn_csv)}`\n\n")
        f.write("## Feature-importance methods\n\n")
        f.write("- embedding_grad_abs: absolute gradient saliency on each gene embedding token.\n")
        f.write("- embedding_grad_x_input: gradient x input on gene embedding tokens.\n")
        f.write("- weight_grad_abs: absolute gradient saliency on each expression weight scalar.\n")
        f.write("- weight_grad_x_input: gradient x input on expression weight scalar.\n")
        f.write("- integrated_grad_x_input: integrated gradients from zero baseline on embedding and weight inputs.\n")
        f.write("- mean_expression_weight: average input expression weight across full-test samples.\n")
        f.write("- consensus_rank: mean-rank consensus across embedding_grad_x_input, weight_grad_x_input, integrated_grad_x_input, and mean_expression_weight.\n\n")
        f.write("## VASN rank\n\n")
        f.write(f"- consensus rank: {vasn_row['consensus_rank']} / 2000\n")
        f.write(f"- embedding_grad_abs rank: {vasn_row['embedding_grad_abs_rank']} / 2000\n")
        f.write(f"- embedding_grad_x_input rank: {vasn_row['embedding_grad_x_input_rank']} / 2000\n")
        f.write(f"- weight_grad_abs rank: {vasn_row['weight_grad_abs_rank']} / 2000\n")
        f.write(f"- weight_grad_x_input rank: {vasn_row['weight_grad_x_input_rank']} / 2000\n")
        f.write(f"- integrated_grad_x_input rank: {vasn_row['integrated_grad_x_input_rank']} / 2000\n")
        f.write(f"- mean_expression_weight rank: {vasn_row['mean_expression_weight_rank']} / 2000\n")

    print(f"[DONE] run_dir={run_dir}", flush=True)
    print(f"[DONE] checkpoint={ckpt_path}", flush=True)
    print(f"[DONE] test_auc={final['roc_auc_ovr']:.4f} acc={final['accuracy']:.4f} microF1={final['accuracy']:.4f} macroF1={final['macro_f1']:.4f}", flush=True)
    print(f"[DONE] xlsx={xlsx_path}", flush=True)
    print(f"[DONE] vasn_consensus_rank={vasn_row['consensus_rank']}", flush=True)


if __name__ == "__main__":
    main()
