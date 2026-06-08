from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from tqdm import tqdm

from comic_dinov2_binary.src.data import build_transforms
from comic_dinov2_binary.src.model import build_model
from comic_dinov2_binary.src.utils import ensure_dir, save_json


class UnlabeledImageDataset:
    def __init__(self, image_dir: str | Path, image_size: int = 518, recursive: bool = True) -> None:
        self.image_dir = Path(image_dir)
        self.transform = build_transforms(image_size=image_size, is_train=False)

        self.image_paths: List[Path] = []
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        if recursive:
            for ext in image_extensions:
                self.image_paths.extend(self.image_dir.rglob(f"*{ext}"))
        else:
            for ext in image_extensions:
                self.image_paths.extend(self.image_dir.glob(f"*{ext}"))
        self.image_paths = sorted(self.image_paths)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), str(p)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 DINOv2 + Logistic Regression 进行高置信伪标签并加入训练集")
    parser.add_argument("--dataset-root", type=str, required=True, help="已有标注数据根目录（包含 train/val/test）")
    parser.add_argument("--unlabeled-dir", type=str, required=True, help="未标注图片目录")
    parser.add_argument("--logreg-path", type=str, required=True, help="训练好的 Logistic Regression 模型路径")
    parser.add_argument("--output-dataset-root", type=str, default="post_training/outputs/pseudo_dataset", help="输出新数据集目录")
    parser.add_argument("--threshold", type=float, default=0.95, help="高置信阈值，默认 0.95")
    parser.add_argument("--max-add", type=int, default=0, help="最多加入样本数，0 表示不限制")
    parser.add_argument("--backbone", type=str, default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def merge_cls_patch_mean(feat: torch.Tensor) -> torch.Tensor:
    if feat.ndim == 3:
        cls_token = feat[:, 0, :]
        patch_mean = feat[:, 1:, :].mean(dim=1) if feat.size(1) > 1 else cls_token
        return torch.cat([cls_token, patch_mean], dim=1)
    if feat.ndim > 2:
        return feat.flatten(1)
    return feat


def extract_features(model: torch.nn.Module, loader, device: torch.device) -> tuple[np.ndarray, List[str]]:
    model.eval()
    feats: List[np.ndarray] = []
    paths: List[str] = []

    with torch.no_grad():
        for images, image_paths in tqdm(loader, desc="提取未标注特征"):
            images = images.to(device, non_blocking=True)
            if hasattr(model, "forward_features"):
                out = model.forward_features(images)
            elif hasattr(model, "backbone"):
                out = model.backbone(images)
            else:
                out = model(images)
            out = merge_cls_patch_mean(out)
            feats.append(out.cpu().numpy())
            paths.extend(image_paths)

    if not feats:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.concatenate(feats, axis=0), paths


def copy_dataset_tree(src_root: Path, dst_root: Path) -> None:
    if dst_root.exists():
        raise FileExistsError(f"output-dataset-root already exists: {dst_root}")
    shutil.copytree(src_root, dst_root)


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    output_dataset_root = Path(args.output_dataset_root)
    threshold = float(args.threshold)
    max_add = int(args.max_add)

    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset-root not found: {dataset_root}")
    if not (dataset_root / "train").exists():
        raise FileNotFoundError(f"train split not found under dataset-root: {dataset_root}")

    clf = joblib.load(args.logreg_path)
    if not isinstance(clf, LogisticRegression):
        print("警告: 提供的模型不是 sklearn LogisticRegression，将继续尝试推理。")

    model = build_model(backbone_name=args.backbone, freeze_backbone=True).to(args.device)

    ds = UnlabeledImageDataset(args.unlabeled_dir, image_size=args.image_size, recursive=args.recursive)
    if len(ds) == 0:
        raise ValueError(f"No images found in unlabeled-dir: {args.unlabeled_dir}")

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    features, paths = extract_features(model, loader, torch.device(args.device))
    features = normalize(features, norm="l2", axis=1)

    if not hasattr(clf, "predict_proba"):
        raise RuntimeError("Logistic Regression 模型不支持 predict_proba，无法按置信度阈值筛选")

    probs = clf.predict_proba(features)
    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)

    candidates = []
    for p, pred, conf in zip(paths, preds, confs):
        if float(conf) >= threshold:
            candidates.append({
                "image_path": p,
                "pred": int(pred),
                "pred_name": "comics" if int(pred) == 1 else "noncomics",
                "confidence": float(conf),
            })

    candidates = sorted(candidates, key=lambda x: x["confidence"], reverse=True)
    if max_add > 0:
        candidates = candidates[:max_add]

    copy_dataset_tree(dataset_root, output_dataset_root)

    train_comics = ensure_dir(output_dataset_root / "train" / "comics")
    train_noncomics = ensure_dir(output_dataset_root / "train" / "noncomics")

    added_rows = []
    for i, row in enumerate(candidates):
        src = Path(row["image_path"])
        if not src.exists():
            continue
        dst_dir = train_comics if row["pred"] == 1 else train_noncomics
        dst_name = f"pseudo_{i:06d}_{src.name}"
        dst = dst_dir / dst_name
        shutil.copy2(src, dst)

        row_out = dict(row)
        row_out["added_path"] = str(dst)
        added_rows.append(row_out)

    out_meta_dir = ensure_dir(output_dataset_root / "pseudo_labeling")
    save_json({"rows": added_rows}, out_meta_dir / "pseudo_added_rows.json")

    comics_count = sum(1 for r in added_rows if r["pred"] == 1)
    noncomics_count = sum(1 for r in added_rows if r["pred"] == 0)
    summary: Dict[str, object] = {
        "dataset_root": str(dataset_root),
        "output_dataset_root": str(output_dataset_root),
        "unlabeled_dir": str(args.unlabeled_dir),
        "threshold": threshold,
        "max_add": max_add,
        "unlabeled_total": len(ds),
        "high_conf_candidates": len(candidates),
        "actually_added": len(added_rows),
        "added_distribution": {
            "comics": comics_count,
            "noncomics": noncomics_count,
        },
    }
    save_json(summary, out_meta_dir / "summary.json")

    print("伪标签完成")
    print(f"未标注总数: {len(ds)}")
    print(f"高置信样本: {len(candidates)}")
    print(f"实际加入训练集: {len(added_rows)}")
    print(f"新增 comics: {comics_count}")
    print(f"新增 noncomics: {noncomics_count}")
    print(f"输出数据集: {output_dataset_root}")


if __name__ == "__main__":
    main()
