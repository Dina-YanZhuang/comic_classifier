from __future__ import annotations

import argparse

from post_training.train_feature_classifier_common import add_common_args, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 DINOv2 特征提取 + SVM 分类器")
    add_common_args(parser, default_output_dir="post_training/outputs/svm_classifier")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training(args, classifier_type="svm")


if __name__ == "__main__":
    main()
