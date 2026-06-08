import os
import random
import shutil

random.seed(42)

source_dir = "/media/tmpuser/DATA/yan_serier/new_training_dataset"
target_dir = "/media/tmpuser/DATA/yan_serier/new_training_dataset/split"

classes = ["comics", "noncomics"]

split_ratio = (0.7, 0.15, 0.15)

for cls in classes:
    images = os.listdir(os.path.join(source_dir, cls))
    random.shuffle(images)

    n = len(images)
    train_end = int(n * split_ratio[0])
    val_end = int(n * (split_ratio[0] + split_ratio[1]))

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split in splits:
        dest = os.path.join(target_dir, split, cls)
        os.makedirs(dest, exist_ok=True)

        for img in splits[split]:
            shutil.copy(
                os.path.join(source_dir, cls, img),
                os.path.join(dest, img)
            )
    for split in splits:
        dest = os.path.join(target_dir, split, cls)
        os.makedirs(dest, exist_ok=True)

        for img in splits[split]:
            shutil.copy(
                os.path.join(source_dir, cls, img),
                os.path.join(dest, img)
            )