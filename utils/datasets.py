import os
import json

import numpy as np
import torch

from monai import data
import monai.transforms as med


def _load_dataset_json(data_dir: str) -> dict:
    datalist_json = os.path.join(data_dir, "dataset.json")
    if not os.path.exists(datalist_json):
        raise FileNotFoundError(f"dataset.json not found in: {data_dir}")
    with open(datalist_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _norm_path(p: str) -> str:
    """Normalize path separators for the current OS."""
    if p is None:
        return p
    return os.path.normpath(p)


def _join_root(data_dir: str, items: list) -> list:
    """
    Join relative paths in dataset.json with data_dir.
    Keep absolute paths unchanged.
    """
    for it in items:
        if "image" in it and it["image"] is not None:
            it["image"] = _norm_path(it["image"])
            if not os.path.isabs(it["image"]):
                it["image"] = os.path.join(data_dir, it["image"])
        if "label" in it and it["label"] is not None:
            it["label"] = _norm_path(it["label"])
            if not os.path.isabs(it["label"]):
                it["label"] = os.path.join(data_dir, it["label"])
    return items


def get_loader(
    data_dir: str,
    size: int,
    mode: str = "pretrain",
    split: float = 0.2,
    seed: int = 0,
    sigma=None,
):
    """
    Build MONAI datasets.

    Args:
        data_dir: dataset root containing dataset.json
        size: patch size (D=H=W=size)
        mode: "pretrain" | "finetune" | "test"
        split: validation split ratio for finetune
        seed: random seed for train/val split
        sigma: optional Gaussian smoothing (applied only when mode != "test")

    Returns:
        - pretrain: (dataset, None)
        - finetune: (train_dataset, val_dataset)
        - test: (test_dataset, None)
    """
    assert mode in ["pretrain", "finetune", "test"], f"Unsupported mode={mode}"
    full_data = _load_dataset_json(data_dir)

    if mode == "pretrain":
        if "pretrain" not in full_data:
            raise KeyError("dataset.json must contain key: pretrain")
        datalist = [{"image": item["image"]} for item in full_data["pretrain"]]
        datalist = _join_root(data_dir, datalist)

    elif mode == "finetune":
        if "finetune" not in full_data:
            raise KeyError("dataset.json must contain key: finetune")
        datalist = [{"image": item["image"], "label": item["label"]} for item in full_data["finetune"]]
        datalist = _join_root(data_dir, datalist)

        np.random.seed(seed)
        torch.manual_seed(seed)

        total_size = len(datalist)
        if total_size < 2:
            raise ValueError("Finetune datalist is too small to split into train/val.")

        indices = np.random.permutation(total_size)
        train_size = int((1 - split) * total_size)
        train_size = max(1, min(train_size, total_size - 1))

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_datalist = [datalist[i] for i in train_indices]
        val_datalist = [datalist[i] for i in val_indices]

    else:
        if "test" not in full_data:
            raise KeyError("dataset.json must contain key: test")
        datalist = [{"image": item["image"], "label": item["label"]} for item in full_data["test"]]
        datalist = _join_root(data_dir, datalist)

    keys_img = ["image"]
    keys_all = ["image"] + (["label"] if mode != "pretrain" else [])

    base_transform = [
        med.LoadImaged(keys=keys_all, allow_missing_keys=True),
        med.EnsureChannelFirstd(keys=keys_all, allow_missing_keys=True),
        med.Orientationd(keys=keys_all, axcodes="RAS", allow_missing_keys=True),
        med.Spacingd(
            keys=keys_all,
            pixdim=(1.5, 1.5, 1.5),
            mode=("bilinear", "nearest") if mode != "pretrain" else "bilinear",
            allow_missing_keys=True,
        ),
        med.ScaleIntensityRanged(keys=keys_img, a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
    ]

    if sigma is not None and mode != "test":
        base_transform.insert(3, med.GaussianSmooth(sigma=sigma))

    if mode == "pretrain":
        transform = med.Compose(
            base_transform
            + [
                med.CropForegroundd(keys=keys_img, source_key="image", allow_missing_keys=True),
                med.SpatialPadd(keys=keys_img, spatial_size=(size, size, size), mode="constant"),
                med.RandCropByPosNegLabeld(
                    spatial_size=(size, size, size),
                    keys=keys_img,
                    label_key="image",
                    pos=1,
                    neg=0,
                    num_samples=2,
                ),
                med.RandFlipd(keys=keys_img, prob=0.2, spatial_axis=0),
                med.RandFlipd(keys=keys_img, prob=0.2, spatial_axis=1),
                med.RandFlipd(keys=keys_img, prob=0.1, spatial_axis=2),
                med.ToTensord(keys=keys_img),
            ]
        )

        dataset = data.CacheNTransDataset(
            data=datalist,
            transform=transform,
            cache_n_trans=6,
            cache_dir=os.path.join(data_dir, "cache_pretrain"),
        )
        return dataset, None

    if mode == "finetune":
        train_transform = med.Compose(
            base_transform
            + [
                med.CropForegroundd(keys=keys_all, source_key="image", allow_missing_keys=True),
                med.SpatialPadd(keys=keys_all, spatial_size=(size, size, size)),
                med.RandCropByPosNegLabeld(
                    spatial_size=(size, size, size),
                    keys=keys_all,
                    label_key="image",
                    pos=1,
                    neg=0,
                    num_samples=2,
                ),
                med.RandFlipd(keys=keys_all, prob=0.2, spatial_axis=0),
                med.RandFlipd(keys=keys_all, prob=0.2, spatial_axis=1),
                med.RandFlipd(keys=keys_all, prob=0.1, spatial_axis=2),
                med.ToTensord(keys=keys_all),
            ]
        )

        val_transform = med.Compose(
            base_transform
            + [
                med.CropForegroundd(keys=keys_all, source_key="image", allow_missing_keys=True),
                med.SpatialPadd(keys=keys_all, spatial_size=(size, size, size)),
                med.ToTensord(keys=keys_all),
            ]
        )

        train_dataset = data.CacheNTransDataset(
            data=train_datalist,
            transform=train_transform,
            cache_n_trans=2,
            cache_dir=os.path.join(data_dir, "cache_finetune_train"),
        )

        val_dataset = data.CacheNTransDataset(
            data=val_datalist,
            transform=val_transform,
            cache_n_trans=1,
            cache_dir=os.path.join(data_dir, "cache_finetune_val"),
        )

        return train_dataset, val_dataset

    test_transform = med.Compose(
        base_transform
        + [
            med.CropForegroundd(keys=keys_all, source_key="image", allow_missing_keys=True),
            med.SpatialPadd(keys=keys_all, spatial_size=(size, size, size)),
            med.ToTensord(keys=keys_all),
        ]
    )

    test_dataset = data.CacheNTransDataset(
        data=datalist,
        transform=test_transform,
        cache_n_trans=1,
        cache_dir=os.path.join(data_dir, "cache_test"),
    )
    return test_dataset, None
