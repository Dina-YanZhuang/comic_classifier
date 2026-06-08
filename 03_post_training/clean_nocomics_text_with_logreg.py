from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from tqdm import tqdm

from comic_dinov2_binary.src.data import DatasetInspector, SampleItem, build_dataloader
from comic_dinov2_binary.src.model import build_model
from comic_dinov2_binary.src.utils import ensure_dir, save_json, set_seed
from post_training.train_feature_classifier_common import extract_features


TEXT_LABEL_ALIASES = {
    "text": 1,
    "txt": 1,
    "positive": 1,
    "1": 1,
    "nontext": 0,
    "no_text": 0,
    "notext": 0,
    "negative": 0,
    "0": 0,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def _label_from_name(name: str) -> Optional[int]:
    return TEXT_LABEL_ALIASES.get(name.strip().lower())


def _read_manifest_like(split_dir: Path) -> List[SampleItem]:
    items: List[SampleItem] = []
    manifest = split_dir / "manifest.jsonl"
    if manifest.exists():
        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                raw_label = rec.get("label", rec.get("class", rec.get("class_name")))
                label = _label_from_name(str(raw_label)) if not isinstance(raw_label, int) else int(raw_label)
                if label not in (0, 1):
                    continue
                image_file = rec.get("image", {}).get("file") or rec.get("image_file")
                if not image_file:
                    continue
                image_path = split_dir / image_file
                if not image_path.exists():
                    continue
                sample_id = rec.get("sample_id", image_path.stem)
                items.append(SampleItem(image_path=image_path, label=label, sample_id=sample_id))

    if items:
        return items

    for json_path in sorted(split_dir.glob("*.json")):
        try:
            rec = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        raw_label = rec.get("label", rec.get("class", rec.get("class_name")))
        label = _label_from_name(str(raw_label)) if not isinstance(raw_label, int) else int(raw_label)
        if label not in (0, 1):
            continue
        image_file = rec.get("image", {}).get("file") or rec.get("image_file")
        image_path = split_dir / image_file if image_file else split_dir / f"{json_path.stem}.jpg"
        if not image_path.exists():
            continue
        items.append(SampleItem(image_path=image_path, label=label, sample_id=json_path.stem))

    return items


def load_text_nontext_samples(dataset_root: str | Path, split: str) -> List[SampleItem]:
    split_dir = Path(dataset_root) / split
    if not split_dir.exists():
        return []

    class_dirs = [p for p in split_dir.iterdir() if p.is_dir()]
    items: List[SampleItem] = []
    if class_dirs:
        for class_dir in sorted(class_dirs):
            label = _label_from_name(class_dir.name)
            if label is None:
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if image_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                items.append(SampleItem(image_path=image_path, label=label, sample_id=image_path.stem))
        if items:
            return items

    return _read_manifest_like(split_dir)


def _count_binary(labels: np.ndarray) -> Dict[str, int]:
    return {
        "nontext": int(np.sum(labels == 0)),
        "text": int(np.sum(labels == 1)),
    }


def train_text_nontext_logreg(
    text_dataset_root: str,
    output_dir: str,
    backbone: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    logreg_c: float,
    max_iter: int,
    seed: int,
    device: torch.device,
) -> tuple[LogisticRegression, Dict[str, object]]:
    print("[1/4] 加载 text/nontext 数据集...")
    train_samples = load_text_nontext_samples(text_dataset_root, "train")
    val_samples = load_text_nontext_samples(text_dataset_root, "val")
    test_samples = load_text_nontext_samples(text_dataset_root, "test")

    if not train_samples:
        raise RuntimeError("text/nontext 数据集 train 切分为空，无法训练")

    print(f"text/nontext train 样本数: {len(train_samples)}")
    print(f"text/nontext val 样本数: {len(val_samples)}")
    print(f"text/nontext test 样本数: {len(test_samples)}")

    print("[2/4] 构建 DINOv2 特征提取模型...")
    model = build_model(backbone_name=backbone, freeze_backbone=True).to(device)

    def _extract(split_samples: Iterable[SampleItem], is_train: bool, split_name: str):
        split_samples = list(split_samples)
        if not split_samples:
            return None, None
        print(f"提取 {split_name} 特征...")
        loader = build_dataloader(
            samples=split_samples,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            is_train=is_train,
            aggressive_aug=is_train,
        )
        feats, labels, _ = extract_features(model, loader, device)
        feats = normalize(feats, norm="l2", axis=1)
        return feats, labels

    x_train, y_train = _extract(train_samples, True, "train")
    x_val, y_val = _extract(val_samples, False, "val")
    x_test, y_test = _extract(test_samples, False, "test")

    fit_x = x_train
    fit_y = y_train
    if x_val is not None and y_val is not None:
        fit_x = np.concatenate([fit_x, x_val], axis=0)
        fit_y = np.concatenate([fit_y, y_val], axis=0)

    print("[3/4] 训练 Logistic Regression（class_weight=balanced）...")
    clf = LogisticRegression(
        penalty="l2",
        C=logreg_c,
        solver="lbfgs",
        max_iter=max_iter,
        random_state=seed,
        class_weight="balanced",
    )
    clf.fit(fit_x, fit_y)

    train_pred = clf.predict(x_train)
    metrics: Dict[str, object] = {
        "fit_label_distribution": _count_binary(fit_y),
        "train_report": classification_report(y_train, train_pred, target_names=["nontext", "text"], output_dict=True),
    }

    if x_val is not None and y_val is not None:
        val_pred = clf.predict(x_val)
        metrics["val_report"] = classification_report(y_val, val_pred, target_names=["nontext", "text"], output_dict=True)
    if x_test is not None and y_test is not None:
        test_pred = clf.predict(x_test)
        metrics["test_report"] = classification_report(y_test, test_pred, target_names=["nontext", "text"], output_dict=True)

    out_dir = ensure_dir(output_dir)
    joblib.dump(clf, out_dir / "text_nontext_logreg.pkl")
    save_json(metrics, out_dir / "text_nontext_results.json")
    print(f"text/nontext 模型已保存: {out_dir / 'text_nontext_logreg.pkl'}")

    return clf, metrics


def rewrite_manifest_if_exists(split_dir: Path, removed_ids: set[str]) -> Optional[Path]:
    manifest = split_dir / "manifest.jsonl"
    if not manifest.exists():
        return None

    kept_lines: List[str] = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            image_file = rec.get("image", {}).get("file") or rec.get("image_file")
            sample_id = rec.get("sample_id", Path(image_file).stem if image_file else "")
            if sample_id in removed_ids:
                continue
            if image_file:
                image_path = split_dir / image_file
                if not image_path.exists():
                    continue
            kept_lines.append(json.dumps(rec, ensure_ascii=False))

    manifest.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
    return manifest


def clean_nocomics_text_samples(
    original_dataset_root: str,
    target_split: str,
    clf: LogisticRegression,
    backbone: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    confidence_threshold: float,
    target_text_to_nontext_ratio: float,
    dry_run: bool,
    device: torch.device,
) -> Dict[str, object]:
    if not (0.0 < confidence_threshold <= 1.0):
        raise ValueError("confidence_threshold 必须在 (0, 1]")
    if target_text_to_nontext_ratio <= 0:
        raise ValueError("target_text_to_nontext_ratio 必须 > 0")

    print("[4/4] 清洗原数据集 nocomics 中的 text 样本...")
    inspector = DatasetInspector(original_dataset_root)
    split_samples = inspector.load_split(target_split)
    nocomics_samples = [s for s in split_samples if int(s.label) == 0]
    if not nocomics_samples:
        raise RuntimeError(f"{target_split} 切分中没有 nocomics 样本")

    print(f"目标切分: {target_split}, nocomics 样本数: {len(nocomics_samples)}")

    model = build_model(backbone_name=backbone, freeze_backbone=True).to(device)
    loader = build_dataloader(
        samples=nocomics_samples,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        is_train=False,
    )
    feats, _, sample_ids = extract_features(model, loader, device)
    feats = normalize(feats, norm="l2", axis=1)

    probs = clf.predict_proba(feats)
    preds = clf.predict(feats)
    text_probs = probs[:, 1]
    is_text = preds == 1
    text_count = int(np.sum(is_text))
    nontext_count = int(len(preds) - text_count)

    target_text_count = int(np.floor(nontext_count * target_text_to_nontext_ratio))
    need_remove = max(0, text_count - target_text_count)

    high_conf_indices = np.where((is_text) & (text_probs >= confidence_threshold))[0]
    high_conf_sorted = high_conf_indices[np.argsort(text_probs[high_conf_indices])[::-1]]
    remove_num = min(need_remove, len(high_conf_sorted))
    remove_indices = high_conf_sorted[:remove_num]
    remove_ids = {sample_ids[int(i)] for i in remove_indices}

    id_to_sample = {s.sample_id: s for s in nocomics_samples}
    to_remove_samples = [id_to_sample[sid] for sid in remove_ids if sid in id_to_sample]

    deleted_files = 0
    deleted_meta = 0
    split_dir = Path(original_dataset_root) / target_split
    if not dry_run and to_remove_samples:
        for sample in tqdm(to_remove_samples, desc="删除高置信 text 样本"):
            if sample.image_path.exists():
                sample.image_path.unlink()
                deleted_files += 1
            meta_json = split_dir / f"{sample.sample_id}.json"
            if meta_json.exists():
                meta_json.unlink()
                deleted_meta += 1
        rewrite_manifest_if_exists(split_dir, remove_ids)

    after_text = text_count - remove_num
    after_nontext = nontext_count
    report: Dict[str, object] = {
        "target_split": target_split,
        "threshold": confidence_threshold,
        "target_text_to_nontext_ratio": target_text_to_nontext_ratio,
        "dry_run": dry_run,
        "before": {
            "text": text_count,
            "nontext": nontext_count,
            "ratio_text_to_nontext": float(text_count / max(1, nontext_count)),
        },
        "target_text_count": target_text_count,
        "need_remove": need_remove,
        "high_conf_text_candidates": int(len(high_conf_sorted)),
        "removed": remove_num,
        "after": {
            "text": after_text,
            "nontext": after_nontext,
            "ratio_text_to_nontext": float(after_text / max(1, after_nontext)),
        },
        "could_not_reach_target": bool(remove_num < need_remove),
        "deleted_files": deleted_files,
        "deleted_meta_json": deleted_meta,
        "removed_sample_ids": sorted(remove_ids),
    }

    print("清洗完成。")
    print(f"清洗前 text/nontext: {text_count}/{nontext_count}")
    print(f"实际删除: {remove_num}（候选高置信样本: {len(high_conf_sorted)}）")
    print(f"清洗后 text/nontext: {after_text}/{after_nontext}")
    if report["could_not_reach_target"]:
        print("警告: 高置信 text 样本不足，无法完全达到目标比例。")

    return report


def train_and_clean_text_in_nocomics(
    text_dataset_root: str,
    original_dataset_root: str,
    output_dir: str = "post_training/outputs/text_cleaning",
    backbone: str = "vit_small_patch14_dinov2.lvd142m",
    image_size: int = 518,
    batch_size: int = 32,
    num_workers: int = 4,
    logreg_c: float = 1.0,
    max_iter: int = 500,
    confidence_threshold: float = 0.9,
    target_text_to_nontext_ratio: float = 1.0,
    target_split: str = "train",
    dry_run: bool = False,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, object]:
    """训练 text/nontext 模型并清洗原数据集 nocomics（单函数入口，适合 Notebook 调用）。"""
    set_seed(seed)
    out_dir = ensure_dir(output_dir)
    dev = torch.device(device)

    print("========== 开始执行 text 清洗流水线 ==========")
    print(f"text 数据集: {text_dataset_root}")
    print(f"原始数据集: {original_dataset_root}")
    print(f"输出目录: {out_dir}")

    clf, train_metrics = train_text_nontext_logreg(
        text_dataset_root=text_dataset_root,
        output_dir=str(out_dir),
        backbone=backbone,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        logreg_c=logreg_c,
        max_iter=max_iter,
        seed=seed,
        device=dev,
    )

    clean_report = clean_nocomics_text_samples(
        original_dataset_root=original_dataset_root,
        target_split=target_split,
        clf=clf,
        backbone=backbone,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        confidence_threshold=confidence_threshold,
        target_text_to_nontext_ratio=target_text_to_nontext_ratio,
        dry_run=dry_run,
        device=dev,
    )

    final_report = {
        "train_metrics": train_metrics,
        "clean_report": clean_report,
        "artifacts": {
            "classifier": str(out_dir / "text_nontext_logreg.pkl"),
            "train_results": str(out_dir / "text_nontext_results.json"),
            "clean_report": str(out_dir / "clean_report.json"),
        },
    }
    save_json(final_report, out_dir / "clean_report.json")
    print(f"流水线执行完成，报告已保存: {out_dir / 'clean_report.json'}")
    print("============================================")
    return final_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 text/nontext DINO+LogReg 并清洗原数据集 nocomics")
    parser.add_argument("--text-dataset-root", type=str, required=True, help="text/nontext 数据集根目录（需包含 train，支持 val/test）")
    parser.add_argument("--original-dataset-root", type=str, required=True, help="原始 comics/nocomics 数据集根目录")
    parser.add_argument("--output-dir", type=str, default="post_training/outputs/text_cleaning")
    parser.add_argument("--target-split", type=str, default="train", help="要清洗的切分（默认 train）")
    parser.add_argument("--backbone", type=str, default="vit_small_patch14_dinov2.lvd142m")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--confidence-threshold", type=float, default=0.9, help="仅删除 text 预测置信度 >= 阈值 的样本")
    parser.add_argument("--target-text-to-nontext-ratio", type=float, default=1.0, help="目标 text:nontext 比例，1.0 即 1:1")
    parser.add_argument("--dry-run", action="store_true", help="仅模拟，不实际删除文件")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_clean_text_in_nocomics(
        text_dataset_root=args.text_dataset_root,
        original_dataset_root=args.original_dataset_root,
        output_dir=args.output_dir,
        backbone=args.backbone,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        logreg_c=args.logreg_c,
        max_iter=args.max_iter,
        confidence_threshold=args.confidence_threshold,
        target_text_to_nontext_ratio=args.target_text_to_nontext_ratio,
        target_split=args.target_split,
        dry_run=args.dry_run,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
