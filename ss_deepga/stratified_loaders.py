import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

def make_stratified_loaders_v2(
    data_dir,
    image_size=128,
    batch_size=32,
    val_split=0.15,
    test_split=0.15,
    seed=42,
    num_workers=2,
):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    ds = datasets.ImageFolder(root=data_dir, transform=transform)
    y = np.array(ds.targets)
    idx = np.arange(len(ds))

    # 1) Split: (train+val) vs test
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=seed)
    trainval_idx, test_idx = next(sss_test.split(idx, y))

    # 2) Split: train vs val (sobre trainval)
    y_trainval = y[trainval_idx]
    val_rel = val_split / (1.0 - test_split)  # val como fracción de lo que quedó
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed)
    train_rel, val_rel_idx = next(sss_val.split(trainval_idx, y_trainval))

    train_idx = trainval_idx[train_rel]
    val_idx = trainval_idx[val_rel_idx]

    pin = torch.cuda.is_available()
    train_dl = DataLoader(Subset(ds, train_idx), batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin)
    val_dl   = DataLoader(Subset(ds, val_idx),   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    test_dl  = DataLoader(Subset(ds, test_idx),  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)

    print("Counts:", len(train_idx), len(val_idx), len(test_idx))
    return train_dl, val_dl, test_dl, 1, len(ds.classes), image_size, ds, train_idx, val_idx, test_idx
