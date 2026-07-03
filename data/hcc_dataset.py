import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# =========================
# 类别顺序（建议与你的 label_encoder 一致）
# =========================
HCC_CLASSES_3 = ["Stage I", "Stage II", "Stage III"]


def to_onehot(y_int: int, num_classes: int) -> torch.Tensor:
    y = torch.zeros(num_classes, dtype=torch.float32)
    y[int(y_int)] = 1.0
    return y


def load_label(label_path_npy: str, label_path_txt: str = None):
    """
    优先读 .npy (int label)，如果没有再读 .txt (string label)。
    返回 (y_int, y_str_or_None)
    """
    if os.path.exists(label_path_npy):
        y_int = int(np.load(label_path_npy).item())
        return y_int, None

    if label_path_txt is not None and os.path.exists(label_path_txt):
        with open(label_path_txt, "r", encoding="utf-8") as f:
            y_str = f.read().strip()
        return None, y_str

    raise FileNotFoundError(f"Cannot find label: {label_path_npy} or {label_path_txt}")


def map_stage_to_3class(stage_str: str) -> int:
    """
    把字符串 stage 映射到 3 类：
      Stage I -> 0
      Stage II/IIA/IIB/IIC -> 1
      Stage III/IIIA/IIIB/IIIC -> 2
    """
    s = stage_str.strip().upper().replace("STAGE", "STAGE ").replace("  ", " ").strip()

    if s == "STAGE I":
        return 0

    if s in {"STAGE II", "STAGE IIA", "STAGE IIB", "STAGE IIC"}:
        return 1

    if s in {"STAGE III", "STAGE IIIA", "STAGE IIIB", "STAGE IIIC"}:
        return 2

    raise ValueError(f"Unknown stage string for 3-class mapping: {stage_str}")


def load_weights_from_meta(meta_path: str) -> np.ndarray:
    """
    从 *_meta.tsv 中读取 weight 列，返回 shape=(2000,)
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    df = pd.read_csv(meta_path, sep="\t")
    if "weight" not in df.columns:
        raise ValueError(f"'weight' column not found in: {meta_path}")

    w = df["weight"].to_numpy(dtype=np.float32)
    return w


class HCCDatasetDual(Dataset):
    """
    新目录结构（你现在生成的）：
      raw_1/
        train/
          TCGA-xxx_xgb_fixed.npy
          TCGA-xxx_xgb_fixed_meta.tsv
          TCGA-xxx_case_topexpr.npy
          TCGA-xxx_case_topexpr_meta.tsv
        train_label/
          TCGA-xxx.npy
          TCGA-xxx.txt (可选)
        val/
        val_label/
        test/
        test_label/
    """

    def __init__(
        self,
        raw_dir: str,
        subset: str = "train",
        num_classes: int = 3,
        stage_string_to_onehot: bool = False,
        transform_xgb=None,
        transform_case=None,
        dtype=torch.float32,
        return_weights: bool = True,
    ):
        """
        raw_dir:
            例如 E:/workspace/Project_HCC/features_genept_ada_dualparts_globalnorm/raw_1

        subset:
            train / val / test

        num_classes:
            one-hot 维度

        stage_string_to_onehot:
            False: 直接用 label .npy 的 int -> onehot
            True: 优先用 .txt 的 stage 字符串映射成 3 类

        transform_xgb:
            对 xgb_fixed 矩阵做 transform

        transform_case:
            对 case_topexpr 矩阵做 transform

        return_weights:
            是否返回两个矩阵各自对应的 weight
        """
        self.raw_dir = raw_dir
        self.subset = subset
        self.num_classes = num_classes
        self.stage_string_to_onehot = stage_string_to_onehot
        self.transform_xgb = transform_xgb
        self.transform_case = transform_case
        self.dtype = dtype
        self.return_weights = return_weights

        self.data_dir = os.path.join(raw_dir, subset)
        self.label_dir = os.path.join(raw_dir, f"{subset}_label")

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data dir not found: {self.data_dir}")
        if not os.path.isdir(self.label_dir):
            raise FileNotFoundError(f"Label dir not found: {self.label_dir}")

        # 以 *_xgb_fixed.npy 为基准收集 case_id
        self.case_ids = sorted([
            f.replace("_xgb_fixed.npy", "")
            for f in os.listdir(self.data_dir)
            if f.endswith("_xgb_fixed.npy")
        ])

        if len(self.case_ids) == 0:
            raise RuntimeError(f"No *_xgb_fixed.npy found in {self.data_dir}")

        # 检查对应文件是否都存在
        self.xgb_fps = []
        self.case_fps = []
        self.xgb_meta_fps = []
        self.case_meta_fps = []
        self.label_fps = []
        self.label_txt_fps = []

        for case_id in self.case_ids:
            xgb_fp = os.path.join(self.data_dir, f"{case_id}_xgb_fixed.npy")
            case_fp = os.path.join(self.data_dir, f"{case_id}_case_topexpr.npy")
            xgb_meta_fp = os.path.join(self.data_dir, f"{case_id}_xgb_fixed_meta.tsv")
            case_meta_fp = os.path.join(self.data_dir, f"{case_id}_case_topexpr_meta.tsv")
            label_fp = os.path.join(self.label_dir, f"{case_id}.npy")
            label_txt_fp = os.path.join(self.label_dir, f"{case_id}.txt")

            if not os.path.exists(xgb_fp):
                raise FileNotFoundError(f"Missing file: {xgb_fp}")
            if not os.path.exists(case_fp):
                raise FileNotFoundError(f"Missing file: {case_fp}")
            if not os.path.exists(xgb_meta_fp):
                raise FileNotFoundError(f"Missing file: {xgb_meta_fp}")
            if not os.path.exists(case_meta_fp):
                raise FileNotFoundError(f"Missing file: {case_meta_fp}")
            if not os.path.exists(label_fp) and not os.path.exists(label_txt_fp):
                raise FileNotFoundError(f"Missing label for case: {case_id}")

            self.xgb_fps.append(xgb_fp)
            self.case_fps.append(case_fp)
            self.xgb_meta_fps.append(xgb_meta_fp)
            self.case_meta_fps.append(case_meta_fp)
            self.label_fps.append(label_fp)
            self.label_txt_fps.append(label_txt_fp)

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, i):
        case_id = self.case_ids[i]

        # 1) load two matrices
        x_xgb = np.load(self.xgb_fps[i])
        x_case = np.load(self.case_fps[i])

        x_xgb = torch.tensor(x_xgb, dtype=self.dtype)
        x_case = torch.tensor(x_case, dtype=self.dtype)

        # 2) optional transform
        if self.transform_xgb is not None:
            x_xgb = self.transform_xgb(x_xgb)

        if self.transform_case is not None:
            x_case = self.transform_case(x_case)

        # 3) load weights from meta
        w_xgb = load_weights_from_meta(self.xgb_meta_fps[i])
        w_case = load_weights_from_meta(self.case_meta_fps[i])

        w_xgb = torch.tensor(w_xgb, dtype=self.dtype)
        w_case = torch.tensor(w_case, dtype=self.dtype)

        # 4) load label
        y_int, y_str = load_label(self.label_fps[i], self.label_txt_fps[i])

        if self.stage_string_to_onehot:
            if y_str is None:
                y = to_onehot(y_int, self.num_classes)
            else:
                y3 = map_stage_to_3class(y_str)
                y = to_onehot(y3, self.num_classes)
        else:
            if y_int is None:
                y3 = map_stage_to_3class(y_str)
                y = to_onehot(y3, self.num_classes)
            else:
                y = to_onehot(y_int, self.num_classes)

        if self.return_weights:
            return x_xgb, x_case, y, w_xgb, w_case, case_id
        else:
            return x_xgb, x_case, y, case_id


class HCCTransform:
    """
    针对 gene-token 序列 (2000,1536) 的 preprocessing
    """

    @staticmethod
    def standardize_per_case(x: torch.Tensor, eps: float = 1e-6):
        """
        对整个矩阵做标准化
        """
        mu = x.mean()
        std = x.std()
        return (x - mu) / (std + eps)

    @staticmethod
    def standardize_per_gene(x: torch.Tensor, eps: float = 1e-6):
        """
        对每个 gene token 的 embedding 做标准化
        """
        mu = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        return (x - mu) / (std + eps)

    @staticmethod
    def identity(x: torch.Tensor):
        return x

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from collections import Counter

    def count_ds_labels(ds, name="dataset"):
        counts = Counter()
        for i in range(len(ds)):
            sample = ds[i]
            y = sample[2]   # 现在返回: x_xgb, x_case, y, w_xgb, w_case, case_id
            cls = int(torch.argmax(y).item())
            counts[cls] += 1

        total = len(ds)
        print("\n==============================")
        print(f"{name.upper()}  (total={total})")
        print("==============================")
        for c in range(ds.num_classes if hasattr(ds, "num_classes") else 3):
            n = counts.get(c, 0)
            pct = (n / total * 100.0) if total > 0 else 0.0
            print(f"class {c}: {n}  ({pct:.2f}%)")
        return counts

    raw_dir = r"E:\workspace\Project_HCC\features_genept_ada_dualparts_globalnorm\raw_1"

    transform_xgb = lambda x: HCCTransform.standardize_per_case(x)
    transform_case = lambda x: HCCTransform.standardize_per_case(x)

    train_ds = HCCDatasetDual(
        raw_dir,
        subset="train",
        num_classes=3,
        transform_xgb=transform_xgb,
        transform_case=transform_case,
    )
    val_ds = HCCDatasetDual(
        raw_dir,
        subset="val",
        num_classes=3,
        transform_xgb=transform_xgb,
        transform_case=transform_case,
    )
    test_ds = HCCDatasetDual(
        raw_dir,
        subset="test",
        num_classes=3,
        transform_xgb=transform_xgb,
        transform_case=transform_case,
    )

    for i in range(5):
        x_xgb, x_case, y, w_xgb, w_case, case_id = train_ds[i]
        print("case:", case_id)
        print("x_xgb shape:", x_xgb.shape)
        print("x_case shape:", x_case.shape)
        print("w_xgb shape:", w_xgb.shape)
        print("w_case shape:", w_case.shape)
        print("onehot label:", y)
        print("argmax label:", torch.argmax(y).item())
        print("xgb weight sum:", w_xgb.sum().item())
        print("case weight sum:", w_case.sum().item())
        print("-" * 40)

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # ===== 统计分布 =====
    count_ds_labels(train_ds, "train")
    count_ds_labels(val_ds, "val")
    count_ds_labels(test_ds, "test")