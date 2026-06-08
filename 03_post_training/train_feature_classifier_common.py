from __future__ import annotations

import argparse
from typing import Dict, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from tqdm import tqdm

from comic_dinov2_binary.src.data import DatasetInspector, build_dataloader
from comic_dinov2_binary.src.model import build_model
from comic_dinov2_binary.src.utils import ensure_dir, save_json, set_seed


def add_common_args(parser: argparse.ArgumentParser, default_output_dir: str) -> None:
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=default_output_dir)
    parser.add_argument("--backbone", type=str, default="vit_small_patch14_dinov2.lvd142m")

    # SVM: C/gamma 矩阵搜索
    parser.add_argument("--svm-c-grid", type=str, default="0.1,1,10", help="SVM C 搜索网格，逗号分隔")
    parser.add_argument("--svm-gamma-grid", type=str, default="1e-4,1e-3,1e-2", help="SVM gamma 搜索网格，逗号分隔")
    parser.add_argument("--svm-cv", type=int, default=3, help="SVM 网格搜索交叉验证折数")

    # Logistic Regression
    parser.add_argument("--logreg-c", type=float, default=1.0, help="Logistic Regression 的 L2 反正则化强度 C")

    # 小 MLP
    parser.add_argument("--mlp-hidden", type=str, default="256", help="MLP 隐藏层，逗号分隔，例如 256 或 256,64")
    parser.add_argument("--mlp-alpha", type=float, default=1e-4, help="MLP L2 正则系数 alpha")
    parser.add_argument("--mlp-lr", type=float, default=1e-3, help="MLP 初始学习率")
    parser.add_argument("--max-iter", type=int, default=300, help="LogReg/MLP 最大迭代轮数")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--aggressive-aug", action="store_true", default=True, help="训练特征提取时启用更强数据增强")
    parser.add_argument("--train-tta", type=int, default=3, help="训练集特征提取的 TTA 次数（>=1）")
    parser.add_argument("--tta-comics-only", dest="tta_comics_only", action="store_true", default=True, help="仅对 comics 类别应用多次 TTA（默认开启）")
    parser.add_argument("--no-tta-comics-only", dest="tta_comics_only", action="store_false", help="对所有类别应用多次 TTA")
    parser.add_argument(
        "--imbalance-strategy",
        type=str,
        default="both",
        choices=["none", "class_weight", "oversample", "both"],
        help="类别不平衡处理策略：none/class_weight/oversample/both",
    )
    parser.add_argument("--oversample-target-ratio", type=float, default=1.0, help="少数类过采样目标占比（相对多数类，1.0 表示采样到同数量）")
    parser.add_argument("--n-jobs", type=int, default=-1, help="网格搜索并行线程数")
    parser.add_argument("--svm-scoring", type=str, default="f1_macro", help="SVM 网格搜索评分指标（默认 f1_macro，更适合不平衡场景）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)


def parse_float_grid(raw: str) -> list[float]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("网格参数不能为空")
    return [float(v) for v in vals]


def parse_hidden_layers(raw: str) -> tuple[int, ...]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("mlp-hidden 不能为空")
    layers = tuple(int(v) for v in vals)
    if any(v <= 0 for v in layers):
        raise ValueError("mlp-hidden 的每一层都必须是正整数")
    return layers


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=2).astype(np.float32)
    weights = np.ones(2, dtype=np.float32)
    total = float(max(1, counts.sum()))
    for c in (0, 1):
        if counts[c] > 0:
            weights[c] = total / (2.0 * counts[c])
    return weights


class ProgressiveTokenMLP(nn.Module):
    """逐层聚合 patch token 的 MLP 分类器。"""

    def __init__(self, token_dim: int, hidden_dims: tuple[int, ...], num_classes: int = 2) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims 不能为空")

        self.token_dim = int(token_dim)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.num_classes = int(num_classes)

        cls_layers = []
        patch_layers = []
        score_layers = []
        in_dim = self.token_dim
        for h in self.hidden_dims:
            cls_layers.append(nn.Linear(in_dim, h))
            patch_layers.append(nn.Linear(in_dim, h))
            score_layers.append(nn.Linear(h, 1))
            in_dim = h

        self.cls_layers = nn.ModuleList(cls_layers)
        self.patch_layers = nn.ModuleList(patch_layers)
        self.score_layers = nn.ModuleList(score_layers)
        self.classifier = nn.Linear(self.hidden_dims[-1], self.num_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, C]，第 0 个 token 为 CLS。
        if tokens.ndim != 3:
            raise ValueError(f"Expected token tensor [B, T, C], got shape {tuple(tokens.shape)}")

        cls_token = tokens[:, 0, :]
        patch_tokens = tokens[:, 1:, :]
        if patch_tokens.size(1) == 0:
            patch_tokens = cls_token.unsqueeze(1)

        cls_state = cls_token
        patch_state = patch_tokens
        for cls_fc, patch_fc, score_fc in zip(self.cls_layers, self.patch_layers, self.score_layers):
            cls_state = F.relu(cls_fc(cls_state))
            patch_state = F.relu(patch_fc(patch_state))
            attn_logits = score_fc(patch_state).squeeze(-1)
            attn = torch.softmax(attn_logits, dim=1)
            patch_context = torch.sum(patch_state * attn.unsqueeze(-1), dim=1)
            cls_state = cls_state + patch_context

        return self.classifier(cls_state)


def _extract_backbone_output(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        return model.forward_features(images)
    if hasattr(model, "backbone"):
        return model.backbone(images)
    return model(images)


def _to_token_tensor(feat: torch.Tensor) -> torch.Tensor:
    if feat.ndim == 3:
        return feat
    if feat.ndim == 2:
        return feat.unsqueeze(1)
    if feat.ndim > 3:
        b = feat.size(0)
        return feat.flatten(1).unsqueeze(1)
    raise ValueError(f"不支持的特征形状: {tuple(feat.shape)}")


def merge_cls_patch_mean(feat: torch.Tensor) -> torch.Tensor:
    if feat.ndim == 3:
        cls_token = feat[:, 0, :]
        patch_mean = feat[:, 1:, :].mean(dim=1) if feat.size(1) > 1 else cls_token
        return torch.cat([cls_token, patch_mean], dim=1)
    if feat.ndim > 2:
        return feat.flatten(1)
    return feat


def extract_features(
    model: nn.Module,
    loader,
    device: torch.device,
    feature_strategy: str = "cls_plus_patch_mean_concat",
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    features_list = []
    labels_list = []
    sample_ids_all: list[str] = []

    with torch.no_grad():
        for images, labels, sample_ids in tqdm(loader, desc="提取特征"):
            images = images.to(device, non_blocking=True)

            raw_feat = _extract_backbone_output(model, images)
            if feature_strategy == "cls_plus_patch_mean_concat":
                feat = merge_cls_patch_mean(raw_feat)
            elif feature_strategy == "cls_plus_all_patch_tokens":
                feat = _to_token_tensor(raw_feat)
            else:
                raise ValueError(f"不支持的特征策略: {feature_strategy}")

            features_list.append(feat.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            sample_ids_all.extend(list(sample_ids))

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels, sample_ids_all


def extract_train_features_with_tta(
    model: nn.Module,
    samples,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    aggressive_aug: bool,
    tta_times: int,
    tta_comics_only: bool,
    comics_label: int = 1,
    feature_strategy: str = "cls_plus_patch_mean_concat",
) -> Tuple[np.ndarray, np.ndarray]:
    if tta_times <= 1:
        loader = build_dataloader(
            samples=samples,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            is_train=True,
            aggressive_aug=aggressive_aug,
        )
        feats, labels, _ = extract_features(model, loader, device, feature_strategy=feature_strategy)
        return feats, labels

    feat_sums: Dict[str, np.ndarray] = {}
    feat_counts: Dict[str, int] = {}
    labels_by_id: Dict[str, int] = {}

    print(f"训练集 TTA 特征提取: pass 1/{tta_times}")
    loader = build_dataloader(
        samples=samples,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        is_train=True,
        aggressive_aug=aggressive_aug,
    )
    feats, labels, sample_ids = extract_features(model, loader, device, feature_strategy=feature_strategy)
    for sid, feat, label in zip(sample_ids, feats, labels):
        feat_sums[sid] = feat.astype(np.float64, copy=True)
        feat_counts[sid] = 1
        labels_by_id[sid] = int(label)

    for i in range(1, tta_times):
        print(f"训练集 TTA 特征提取: pass {i + 1}/{tta_times}")
        loader = build_dataloader(
            samples=samples,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            is_train=True,
            aggressive_aug=aggressive_aug,
        )
        feats, labels, sample_ids = extract_features(model, loader, device, feature_strategy=feature_strategy)
        for sid, feat, label in zip(sample_ids, feats, labels):
            if tta_comics_only and int(label) != comics_label:
                continue
            if sid not in feat_sums:
                feat_sums[sid] = feat.astype(np.float64, copy=True)
                feat_counts[sid] = 1
                labels_by_id[sid] = int(label)
            else:
                feat_sums[sid] += feat
                feat_counts[sid] += 1

    ordered_ids = [s.sample_id for s in samples]
    features_out = []
    labels_out = []
    for sid in ordered_ids:
        if sid not in feat_sums:
            continue
        features_out.append((feat_sums[sid] / max(1, feat_counts[sid])).astype(np.float32))
        labels_out.append(labels_by_id[sid])

    return np.stack(features_out, axis=0), np.asarray(labels_out, dtype=np.int64)


def binary_label_counts(labels: np.ndarray) -> Dict[str, int]:
    labels = np.asarray(labels, dtype=np.int64)
    return {
        "nocomics": int(np.sum(labels == 0)),
        "comics": int(np.sum(labels == 1)),
    }


def oversample_minority_binary(
    features: np.ndarray,
    labels: np.ndarray,
    target_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    if target_ratio <= 0:
        raise ValueError("oversample-target-ratio 必须 > 0")

    y = np.asarray(labels, dtype=np.int64)
    counts = {0: int(np.sum(y == 0)), 1: int(np.sum(y == 1))}
    if counts[0] == 0 or counts[1] == 0:
        return features, labels, {"applied": False, "reason": "single_class", "before": {"nocomics": counts[0], "comics": counts[1]}}

    minority = 0 if counts[0] < counts[1] else 1
    majority = 1 - minority
    minority_count = counts[minority]
    majority_count = counts[majority]

    target_minority_count = int(np.ceil(majority_count * target_ratio))
    if minority_count >= target_minority_count:
        return features, labels, {
            "applied": False,
            "reason": "already_sufficient",
            "before": {"nocomics": counts[0], "comics": counts[1]},
            "target_ratio": float(target_ratio),
        }

    add_count = target_minority_count - minority_count
    rng = np.random.default_rng(seed)
    minority_indices = np.where(y == minority)[0]
    sampled_indices = rng.choice(minority_indices, size=add_count, replace=True)

    features_aug = np.concatenate([features, features[sampled_indices]], axis=0)
    labels_aug = np.concatenate([labels, labels[sampled_indices]], axis=0)

    after_counts = binary_label_counts(labels_aug)
    return features_aug, labels_aug, {
        "applied": True,
        "minority_class": int(minority),
        "majority_class": int(majority),
        "added_samples": int(add_count),
        "target_ratio": float(target_ratio),
        "before": {"nocomics": counts[0], "comics": counts[1]},
        "after": after_counts,
    }


def train_classifier(classifier_type: str, features: np.ndarray, labels: np.ndarray, args: argparse.Namespace) -> tuple[object, Dict[str, object]]:
    extra: Dict[str, object] = {}
    class_weight = "balanced" if args.imbalance_strategy in ("class_weight", "both") else None

    if classifier_type == "svm":
        c_grid = parse_float_grid(args.svm_c_grid)
        gamma_grid = parse_float_grid(args.svm_gamma_grid)
        base_clf = SVC(
            kernel="rbf",
            probability=True,
            random_state=args.seed,
            class_weight=class_weight,
        )
        grid = GridSearchCV(
            estimator=base_clf,
            param_grid={"C": c_grid, "gamma": gamma_grid},
            scoring=args.svm_scoring,
            cv=args.svm_cv,
            n_jobs=args.n_jobs,
            refit=True,
        )
        print("训练 SVM 分类器（C/gamma 网格搜索）...")
        grid.fit(features, labels)
        clf = grid.best_estimator_
        extra["best_params"] = grid.best_params_
        extra["best_cv_score"] = float(grid.best_score_)

    elif classifier_type == "logreg":
        clf = LogisticRegression(
            penalty="l2",
            C=args.logreg_c,
            solver="lbfgs",
            max_iter=args.max_iter,
            random_state=args.seed,
            class_weight=class_weight,
        )
        print("训练 Logistic Regression 分类器...")
        clf.fit(features, labels)

    elif classifier_type == "mlp":
        if features.ndim != 3:
            raise ValueError(f"MLP 期望 token 特征 [N, T, C]，实际: {features.shape}")

        hidden_layers = parse_hidden_layers(args.mlp_hidden)
        token_dim = int(features.shape[-1])
        net = ProgressiveTokenMLP(token_dim=token_dim, hidden_dims=hidden_layers, num_classes=2).to(args.device)

        x = torch.from_numpy(features.astype(np.float32))
        y = torch.from_numpy(labels.astype(np.int64))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=max(1, min(int(args.batch_size), 128)), shuffle=True)

        class_weight = None
        if args.imbalance_strategy in ("class_weight", "both"):
            class_weight = torch.tensor(compute_class_weights(labels), dtype=torch.float32, device=args.device)

        criterion = nn.CrossEntropyLoss(weight=class_weight)
        optimizer = torch.optim.Adam(net.parameters(), lr=float(args.mlp_lr), weight_decay=float(args.mlp_alpha))

        print(f"训练 token 聚合 MLP 分类器（hidden={hidden_layers}）...")
        net.train()
        for epoch in range(int(args.max_iter)):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(args.device, non_blocking=True)
                yb = yb.to(args.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item()) * xb.size(0)

            if (epoch + 1) % 20 == 0 or epoch == 0:
                avg_loss = epoch_loss / max(1, len(dataset))
                print(f"Epoch {epoch + 1}/{int(args.max_iter)} loss={avg_loss:.6f}")

        net.eval()
        clf = {
            "model_type": "torch_progressive_token_mlp",
            "state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "config": {
                "token_dim": token_dim,
                "hidden_dims": list(hidden_layers),
                "num_classes": 2,
            },
        }
        extra["model_type"] = "torch_progressive_token_mlp"
        extra["hidden_dims"] = list(hidden_layers)
        extra["token_dim"] = token_dim

    else:
        raise ValueError(f"不支持的分类器类型: {classifier_type}")

    return clf, extra


def evaluate_classifier(clf, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if isinstance(clf, dict) and clf.get("model_type") == "torch_progressive_token_mlp":
        if features.ndim != 3:
            raise ValueError(f"MLP 评估期望 token 特征 [N, T, C]，实际: {features.shape}")
        cfg = clf["config"]
        net = ProgressiveTokenMLP(
            token_dim=int(cfg["token_dim"]),
            hidden_dims=tuple(int(v) for v in cfg["hidden_dims"]),
            num_classes=int(cfg.get("num_classes", 2)),
        )
        net.load_state_dict(clf["state_dict"], strict=True)
        net.eval()

        with torch.no_grad():
            x = torch.from_numpy(features.astype(np.float32))
            logits = net(x)
            probs_t = torch.softmax(logits, dim=1)
            probs = probs_t.cpu().numpy()
            preds = np.argmax(probs, axis=1)
    else:
        preds = clf.predict(features)
        probs = clf.predict_proba(features) if hasattr(clf, "predict_proba") else None

    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=["nocomics", "comics"], output_dict=True)
    cm = confusion_matrix(labels, preds)

    results = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    if probs is not None:
        confidences = np.max(probs, axis=1)
        results["mean_confidence"] = float(np.mean(confidences))

    return results


def run_training(args: argparse.Namespace, classifier_type: str) -> None:
    set_seed(args.seed)
    device = torch.device(args.device)

    out_dir = ensure_dir(args.output_dir)

    inspector = DatasetInspector(args.dataset_root)
    train_samples = inspector.load_split("train")
    val_samples = inspector.load_split("val")
    test_samples = inspector.load_split("test")

    print(f"训练样本: {len(train_samples)}")
    print(f"验证样本: {len(val_samples)}")
    print(f"测试样本: {len(test_samples)}")

    val_loader = build_dataloader(
        samples=val_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False,
    )
    test_loader = build_dataloader(
        samples=test_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False,
    )

    model = build_model(backbone_name=args.backbone, freeze_backbone=True).to(device)

    feature_strategy = "cls_plus_all_patch_tokens" if classifier_type == "mlp" else "cls_plus_patch_mean_concat"

    print("提取训练集特征...")
    train_features, train_labels = extract_train_features_with_tta(
        model=model,
        samples=train_samples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        aggressive_aug=args.aggressive_aug,
        tta_times=max(1, int(args.train_tta)),
        tta_comics_only=bool(args.tta_comics_only),
        feature_strategy=feature_strategy,
    )

    print("提取验证集特征...")
    val_features, val_labels, _ = extract_features(model, val_loader, device, feature_strategy=feature_strategy)

    print("提取测试集特征...")
    test_features, test_labels, _ = extract_features(model, test_loader, device, feature_strategy=feature_strategy)

    feature_l2_normalized = False
    if classifier_type != "mlp":
        train_features = normalize(train_features, norm="l2", axis=1)
        val_features = normalize(val_features, norm="l2", axis=1)
        test_features = normalize(test_features, norm="l2", axis=1)
        feature_l2_normalized = True

    combined_features = np.concatenate([train_features, val_features], axis=0)
    combined_labels = np.concatenate([train_labels, val_labels], axis=0)

    combined_before = binary_label_counts(combined_labels)
    fit_features = combined_features
    fit_labels = combined_labels
    imbalance_extra: Dict[str, object] = {
        "strategy": args.imbalance_strategy,
        "class_weight": "balanced" if args.imbalance_strategy in ("class_weight", "both") else None,
        "combined_before": combined_before,
    }
    if args.imbalance_strategy in ("oversample", "both"):
        fit_features, fit_labels, oversample_info = oversample_minority_binary(
            combined_features,
            combined_labels,
            target_ratio=float(args.oversample_target_ratio),
            seed=args.seed,
        )
        imbalance_extra["oversample"] = oversample_info
        print(f"不平衡处理（过采样）: {oversample_info}")

    print(f"训练拟合标签分布（处理前）: {combined_before}")
    print(f"训练拟合标签分布（处理后）: {binary_label_counts(fit_labels)}")

    clf, train_extra = train_classifier(classifier_type, fit_features, fit_labels, args)

    print("评估分类器...")
    train_results = evaluate_classifier(clf, train_features, train_labels)
    val_results = evaluate_classifier(clf, val_features, val_labels)
    test_results = evaluate_classifier(clf, test_features, test_labels)

    results = {
        "classifier_type": classifier_type,
        "train_accuracy": train_results["accuracy"],
        "val_accuracy": val_results["accuracy"],
        "test_accuracy": test_results["accuracy"],
        "train_report": train_results["classification_report"],
        "val_report": val_results["classification_report"],
        "test_report": test_results["classification_report"],
        "train_confusion_matrix": train_results["confusion_matrix"],
        "val_confusion_matrix": val_results["confusion_matrix"],
        "test_confusion_matrix": test_results["confusion_matrix"],
        "feature_strategy": feature_strategy,
        "feature_l2_normalized": feature_l2_normalized,
        "train_with_augmentation": True,
        "train_tta_times": max(1, int(args.train_tta)),
        "train_tta_comics_only": bool(args.tta_comics_only),
        "imbalance_handling": imbalance_extra,
        "train_extra": train_extra,
        "args": vars(args),
    }

    if "mean_confidence" in train_results:
        results["train_mean_confidence"] = train_results["mean_confidence"]
    if "mean_confidence" in val_results:
        results["val_mean_confidence"] = val_results["mean_confidence"]
    if "mean_confidence" in test_results:
        results["test_mean_confidence"] = test_results["mean_confidence"]

    save_json(results, out_dir / "results.json")
    joblib.dump(clf, out_dir / f"{classifier_type}_classifier.pkl")

    print("训练完成!")
    print(f"分类器类型: {classifier_type.upper()}")
    print(f"Train Acc: {train_results['accuracy']:.4f}")
    print(f"Val   Acc: {val_results['accuracy']:.4f}")
    print(f"Test  Acc: {test_results['accuracy']:.4f}")
    print(f"结果保存至: {out_dir}")
