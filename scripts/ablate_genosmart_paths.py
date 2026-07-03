import os

import torch

import train
from models.Transformer import GenoSmart


CKPT_PATH = r"E:\workspace\Project_HCC\saved_models_dl_dual\genosmart_overfit_all_best.pt"


def build_model(device):
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
    return model


def main():
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(CKPT_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = train.FitConfig(batch_size=8, num_workers=0, train_on_all_splits=True)
    _, _, test_loader, train_labels, _, _ = train.build_dataloaders(cfg)
    loss_fn = train.build_weighted_ce_loss(train_labels, num_classes=3, device=device)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    model = build_model(device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_metrics = train.evaluate(model, test_loader, loss_fn, device=device)

    print(f"[INFO] checkpoint={CKPT_PATH}")
    print("[INFO] test set is the original raw_1/test split")
    print(f"genosmart: loss={test_loss:.4f} | {train.pretty_metrics(test_metrics)}")


if __name__ == "__main__":
    main()
