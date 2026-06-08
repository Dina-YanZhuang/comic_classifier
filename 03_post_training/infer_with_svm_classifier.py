from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from tqdm import tqdm

from comic_dinov2_binary.src.data import build_transforms
from comic_dinov2_binary.src.model import build_model
from comic_dinov2_binary.src.utils import ensure_dir, save_json
from post_training.train_feature_classifier_common import ProgressiveTokenMLP


class NonAnnotatedDataset:
    def __init__(self, image_dir: str | Path, image_size: int = 518, recursive: bool = True) -> None:
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.transform = build_transforms(image_size=image_size, is_train=False)
        self.recursive = recursive

        # Find all image files
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

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, str(image_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用已训练的 SVM/LogReg/MLP 分类器对无标注数据进行推理")
    parser.add_argument("--image-dir", type=str, required=True, help="输入无标注图片目录")
    parser.add_argument("--classifier-path", type=str, required=True, help="SVM/LogReg/MLP 模型文件路径（*.pkl）")
    parser.add_argument("--output-dir", type=str, default="post_training/outputs/svm_classifier_infer")
    parser.add_argument("--backbone", type=str, default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--recursive", action="store_true", default=True, help="递归搜索子目录")
    return parser.parse_args()


def merge_cls_patch_mean(feat: torch.Tensor) -> torch.Tensor:
    if feat.ndim == 3:
        cls_token = feat[:, 0, :]
        patch_mean = feat[:, 1:, :].mean(dim=1) if feat.size(1) > 1 else cls_token
        return torch.cat([cls_token, patch_mean], dim=1)
    if feat.ndim > 2:
        return feat.flatten(1)
    return feat


def to_token_tensor(feat: torch.Tensor) -> torch.Tensor:
    if feat.ndim == 3:
        return feat
    if feat.ndim == 2:
        return feat.unsqueeze(1)
    if feat.ndim > 3:
        b = feat.size(0)
        return feat.flatten(1).unsqueeze(1)
    raise ValueError(f"不支持的特征形状: {tuple(feat.shape)}")


def extract_features(model: torch.nn.Module, loader, device: torch.device, feature_strategy: str = "cls_plus_patch_mean_concat"):
    model.eval()
    feat_list = []
    paths = []

    with torch.no_grad():
        for images, image_paths in tqdm(loader, desc="提取特征"):
            images = images.to(device, non_blocking=True)

            # 兼容特征抽取接口
            if hasattr(model, "forward_features"):
                features = model.forward_features(images)
            elif hasattr(model, "backbone"):
                features = model.backbone(images)
            else:
                features = model(images)

            if feature_strategy == "cls_plus_patch_mean_concat":
                features = merge_cls_patch_mean(features)
            elif feature_strategy == "cls_plus_all_patch_tokens":
                features = to_token_tensor(features)
            else:
                raise ValueError(f"不支持的特征策略: {feature_strategy}")

            feat_list.append(features.cpu().numpy())
            paths.extend(image_paths)

    features = np.concatenate(feat_list, axis=0) if feat_list else np.zeros((0, 0))
    return features, paths


def infer_with_svm_classifier(
    image_dir: str,
    classifier_path: str,
    output_dir: str | Path,
    backbone: str = "vit_small_patch14_dinov2.lvd142m",
    image_size: int = 518,
    batch_size: int = 16,
    num_workers: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    recursive: bool = True,
) -> Dict[str, object]:
    out_dir = ensure_dir(output_dir)

    dataset = NonAnnotatedDataset(image_dir=image_dir, image_size=image_size, recursive=recursive)
    if len(dataset) == 0:
        raise ValueError(f"No images found in {image_dir}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    model = build_model(backbone_name=backbone, freeze_backbone=True).to(device)

    clf = joblib.load(classifier_path)

    feature_strategy = "cls_plus_patch_mean_concat"
    feature_l2_normalized = True

    is_torch_mlp = isinstance(clf, dict) and clf.get("model_type") == "torch_progressive_token_mlp"
    if is_torch_mlp:
        feature_strategy = "cls_plus_all_patch_tokens"
        feature_l2_normalized = False

    features, image_paths = extract_features(model, loader, torch.device(device), feature_strategy=feature_strategy)

    if is_torch_mlp:
        cfg = clf["config"]
        net = ProgressiveTokenMLP(
            token_dim=int(cfg["token_dim"]),
            hidden_dims=tuple(int(v) for v in cfg["hidden_dims"]),
            num_classes=int(cfg.get("num_classes", 2)),
        ).to(device)
        net.load_state_dict(clf["state_dict"], strict=True)
        net.eval()

        with torch.no_grad():
            x = torch.from_numpy(features.astype(np.float32)).to(device)
            logits = net(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        confs = probs.max(axis=1)
    else:
        features = normalize(features, norm="l2", axis=1)
        preds = clf.predict(features)

        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(features)
            confs = probs.max(axis=1)
        else:
            confs = [1.0] * len(preds)

    rows = []
    for path, pred, conf in zip(image_paths, preds, confs):
        rows.append({
            "image_path": path,
            "pred": int(pred),
            "pred_name": "comics" if int(pred) == 1 else "nocomics",
            "confidence": float(conf),
            "relative_path": str(Path(path).relative_to(image_dir)),
        })

    save_json({"predictions": rows}, out_dir / "predictions.json")
    save_json({"rows": rows}, out_dir / "predictions_detailed.json")

    class_dirs = {0: ensure_dir(out_dir / "nocomics"), 1: ensure_dir(out_dir / "comics")}
    for row in rows:
        src = Path(row["image_path"])
        if src.exists():
            dst = class_dirs[row["pred"]] / src.name
            shutil.copy2(src, dst)

    class_counts = {0: 0, 1: 0}
    for p in preds:
        class_counts[int(p)] += 1

    summary = {
        "total": len(rows),
        "class_distribution": class_counts,
        "avg_confidence": float(sum(confs) / len(confs)) if len(confs) > 0 else 0.0,
        "feature_strategy": feature_strategy,
        "feature_l2_normalized": feature_l2_normalized,
        "output_folders": {
            "nocomics": str(class_dirs[0]),
            "comics": str(class_dirs[1]),
        },
    }
    save_json(summary, out_dir / "summary.json")

    return summary


def main() -> None:
    args = parse_args()
    infer_with_svm_classifier(
        image_dir=args.image_dir,
        classifier_path=args.classifier_path,
        output_dir=args.output_dir,
        backbone=args.backbone,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
