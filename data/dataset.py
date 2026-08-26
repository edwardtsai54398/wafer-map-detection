import os
import pickle
import sys

import numpy as np
import cv2
import kagglehub
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from constant import LABEL_MAP, MEAN, STD

_MEAN = np.array(MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.array(STD, dtype=np.float32).reshape(3, 1, 1)


def _load_legacy_lswmd_pickle(pkl_path):
    """Load the original WM-811K LSWMD.pkl.

    The file was pickled under a pre-1.0 pandas (module layout
    ``pandas.indexes.*``, Python 2 str objects) so it can't be read directly
    with a current pandas/Python. Shim the old module paths to their current
    equivalents and unpickle with the Python-2-compatible ``latin1`` encoding.
    """
    import pandas.core.indexes.base as _idx_base
    import pandas.core.indexes.range as _idx_range

    added = {
        "pandas.indexes": _idx_base,
        "pandas.indexes.base": _idx_base,
        "pandas.indexes.range": _idx_range,
    }
    previous = {name: sys.modules.get(name) for name in added}
    sys.modules.update(added)
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f, encoding="latin1")
    finally:
        for name, mod in previous.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def load_raw_data():
    path = kagglehub.dataset_download("qingyi/wm811k-wafer-map")
    v2_path = f"{path}/LSWMD_v2.pkl"

    if os.path.exists(v2_path):
        df = pd.read_pickle(v2_path)
    else:
        df = _load_legacy_lswmd_pickle(f"{path}/LSWMD.pkl")
        df.to_pickle(v2_path)

    df["failureType"] = df["failureType"].apply(
        lambda x: x[0][0] if isinstance(x, np.ndarray) and x.size > 0 else x
    )
    return df


def split_df(df, stratify_col, test_size=0.3, val_size=0.5, random_seed=42):
    train_df, tmp_df = train_test_split(
        df, test_size=test_size, random_state=random_seed, stratify=df[stratify_col]
    )
    val_df, test_df = train_test_split(
        tmp_df, test_size=val_size, random_state=random_seed, stratify=tmp_df[stratify_col]
    )
    return train_df, val_df, test_df


def undersample_none(train_df, none_cap, random_seed=42):
    none_mask = train_df["failureType"] == "none"
    sampled_none = train_df[none_mask].sample(n=none_cap, random_state=random_seed)
    result = pd.concat([train_df[~none_mask], sampled_none], ignore_index=True)
    return result.sample(frac=1, random_state=random_seed).reset_index(drop=True)


# Geometric augmentations for wafer maps. Each takes a 2D array and returns a
# transformed 2D array of the same shape (rotations keep shape only for square
# maps, which is fine since maps are resized downstream).
_AUGMENTATIONS = {
    "hflip":  lambda w: np.fliplr(w),
    "vflip":  lambda w: np.flipud(w),
    "rot90":  lambda w: np.rot90(w, k=1),
    "rot180": lambda w: np.rot90(w, k=2),
    "rot270": lambda w: np.rot90(w, k=3),
}


def augment_minority_classes(
    train_df,
    target_count=None,
    label_col="failureType",
    exclude=("none",),
    random_seed=42,
):
    """Oversample minority defect-pattern classes via flips and rotations.

    For every class whose sample count is below ``target_count``, new samples
    are created by applying horizontal/vertical flips and 90/180/270-degree
    rotations to existing samples of that class.

    Augmentation is limited to a single round: each distinct
    ``(base sample, augmentation)`` pair is used at most once, so no duplicate
    augmented rows are produced. Since there are ``len(_AUGMENTATIONS)`` (5)
    augmentations, a class can grow to at most ``1 + len(_AUGMENTATIONS)`` (6x)
    its original size, regardless of how large ``target_count`` is. A class
    that already exceeds its effective target is left untouched.

    Parameters
    ----------
    train_df     : DataFrame with a "waferMap" column and ``label_col``.
    target_count : Desired number of samples per minority class. Defaults to
                   the size of the largest non-excluded class. The effective
                   per-class target is capped at 6x the class's original size.
    label_col    : Column holding the class label.
    exclude      : Labels that should not be augmented (e.g. the "none" class).
    random_seed  : Seed for reproducible sampling/augmentation choices.

    Returns
    -------
    A new shuffled DataFrame containing the original rows plus the augmented
    rows (with fresh deep-copied "waferMap" arrays).
    """
    rng = np.random.default_rng(random_seed)
    aug_names = list(_AUGMENTATIONS.keys())
    n_aug = len(aug_names)

    counts = train_df[label_col].value_counts()
    eligible = counts.drop(labels=[c for c in exclude if c in counts.index])
    if target_count is None:
        target_count = int(eligible.max())

    augmented_rows = []
    for label, count in eligible.items():
        count = int(count)
        # One round of augmentation adds at most n_aug unique variants per
        # original sample, so the class can reach at most (1 + n_aug)x its size.
        effective_target = min(target_count, count * (1 + n_aug))
        n_needed = effective_target - count
        if n_needed <= 0:
            continue

        class_df = train_df[train_df[label_col] == label]
        # Draw n_needed *distinct* (base, augmentation) pairs from the
        # count * n_aug possible combinations so no augmented row repeats.
        pair_ids = rng.choice(count * n_aug, size=n_needed, replace=False)
        base_idx = pair_ids // n_aug
        aug_idx = pair_ids % n_aug

        for b, a in zip(base_idx, aug_idx):
            row = class_df.iloc[int(b)].copy()
            row["waferMap"] = _AUGMENTATIONS[aug_names[int(a)]](row["waferMap"]).copy()
            augmented_rows.append(row)

    if not augmented_rows:
        return train_df.reset_index(drop=True)

    result = pd.concat([train_df, pd.DataFrame(augmented_rows)], ignore_index=True)
    return result.sample(frac=1, random_state=random_seed).reset_index(drop=True)


def augment_selected_classes(
    train_df,
    classes,
    target_count=None,
    label_col="failureType",
    random_seed=42,
):
    
    exclude = [c for c in train_df[label_col].unique() if c not in classes]
    return augment_minority_classes(
        train_df,
        target_count=target_count,
        label_col=label_col,
        exclude=exclude,
        random_seed=random_seed,
    )


def make_dataloaders(datasets, batch_size):
    return {
        split: DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        for split in datasets
    }


def _unwrap_failure_type(v):
    """WM-811K stores failureType as numpy arrays; extract the scalar."""
    if isinstance(v, np.ndarray):
        return v[0] if v.size > 0 else 0
    return v


def extract_labeled_patterned_data(df):
    """Split df into (all-labeled, labeled+pattern, labeled+none)."""
    df = df.copy()
    df["failureType"] = df["failureType"].apply(_unwrap_failure_type)
    df_with_label = df[df["failureType"] != 0].reset_index()
    df_with_pattern = df_with_label[df_with_label["failureType"] != "none"].reset_index()
    df_without_pattern = df_with_label[df_with_label["failureType"] == "none"]
    return df_with_label, df_with_pattern, df_without_pattern


class WaferDataset(Dataset):
    """
    Unified wafer-map dataset.

    Parameters
    ----------
    df         : DataFrame containing "waferMap" and the label column.
    label_col  : Column to use as label. "failureType" for multi-class,
                 "isDefect" for binary stage-1.
    label_map  : Dict mapping str label → int index.
                 If None the column value is cast to int directly
                 (used for binary isDefect column).
    image_size : (W, H) target resize dimensions.
    """

    def __init__(self, df, label_col="failureType", label_map=LABEL_MAP, image_size=(32, 32)):
        self.df        = df.reset_index(drop=True)
        self.label_col = label_col
        self.label_map = label_map
        self.image_size = image_size

        if label_map is not None:
            self.targets = self.df[label_col].map(label_map).values
        else:
            self.targets = self.df[label_col].astype(int).values
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wafer = self._resize_wafer(row["waferMap"])

        label = self.label_map[row[self.label_col]] if self.label_map is not None else int(row[self.label_col])

        wafer = np.stack([wafer] * 3, axis=0).astype(np.float32) / 2.0
        wafer = (wafer - _MEAN) / _STD
        return torch.from_numpy(wafer), label

    def _resize_wafer(self, wafer_map):
        return cv2.resize(
            wafer_map, self.image_size, interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)


class WaferSimpleDataset(Dataset):
    """
    1-channel wafer-map dataset for small from-scratch models (e.g. SimpleCNN).

    Unlike WaferDataset, this does not stack to 3 channels or apply ImageNet
    normalization (those exist for pretrained backbones). Pixels are scaled
    to [0, 1].

    Parameters
    ----------
    df         : DataFrame containing "waferMap" and the label column.
    label_col  : Column to use as label.
    label_map  : Dict mapping str label → int index.
                 If None the column value is cast to int directly.
    image_size : (W, H) target resize dimensions.
    """

    def __init__(self, df, label_col="failureType", label_map=LABEL_MAP, image_size=(32, 32)):
        self.df        = df.reset_index(drop=True)
        self.label_col = label_col
        self.label_map = label_map
        self.image_size = image_size

        if label_map is not None:
            self.targets = self.df[label_col].map(label_map).values
        else:
            self.targets = self.df[label_col].astype(int).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wafer = self._resize_wafer(row["waferMap"])

        label = self.label_map[row[self.label_col]] if self.label_map is not None else int(row[self.label_col])

        wafer = torch.from_numpy(wafer.astype(np.float32)).unsqueeze(0) / 2.0
        return wafer, label

    def _resize_wafer(self, wafer_map):
        return cv2.resize(
            wafer_map, self.image_size, interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
