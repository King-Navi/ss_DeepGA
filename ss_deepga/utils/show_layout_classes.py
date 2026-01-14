from collections import Counter
import numpy as np

from ss_deepga.stratified_loaders import make_stratified_loaders_v2

from constant import PATH_TO_CLASSES
def show_split_distribution(ds, split_idx, name):
    y = np.array(ds.targets)[split_idx]
    c = Counter(y)
    total = len(split_idx)
    print(f"\n{name} total = {total}")
    for class_id, class_name in enumerate(ds.classes):
        n = c.get(class_id, 0)
        print(f"  {class_name:35s} {n:4d}  ({n/total:.2%})")

train_dl, val_dl, test_dl, n_channels, n_classes, out_size, ds, train_idx, val_idx, test_idx = make_stratified_loaders_v2(
    data_dir=PATH_TO_CLASSES,
    image_size=128,
    batch_size=32,
    val_split=0.15,
    test_split=0.15,
    num_workers=2,
)

show_split_distribution(ds, train_idx, "TRAIN")
show_split_distribution(ds, val_idx,   "VAL")
show_split_distribution(ds, test_idx,  "TEST")
