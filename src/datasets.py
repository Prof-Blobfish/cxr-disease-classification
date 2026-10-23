# Reads an image from image_path and returns image tensor, target tensor, and maybe image_id

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import functional as TF

from data import DataBundle


def load_image(path: str, attempts: int = 5) -> Image.Image:
    # Windows can briefly lock a file (antivirus, indexer); one denied read shouldn't kill a run.
    for attempt in range(attempts):
        try:
            with Image.open(path) as image_file:
                return image_file.convert("RGB")
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5 * 2 ** attempt)


class CXRDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            target_columns: list[str],
            transform=None,
            include_metadata: bool = False,
    ) -> None:
        required_columns = ["image_id", "image_path", *target_columns]
        missing_columns = [column for column in required_columns if column not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in dataframe: {missing_columns}")
        
        self.df = df.reset_index(drop=True).copy()
        self.target_columns = list(target_columns)
        self.transform = transform
        self.include_metadata = include_metadata

    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]

        image = load_image(row["image_path"])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = TF.to_tensor(image)

        target = torch.tensor(
            row[self.target_columns].to_numpy(dtype="float32"),
            dtype=torch.float32,
        )

        sample: dict[str, Any] = {
            "image": image,
            "target": target,
            "image_id": row["image_id"]
        }

        if self.include_metadata:
            sample["finding_labels"] = row.get("finding_labels", "")
            sample["split"] = row.get("split", "")

        return sample
    
class JointDataset(Dataset):
    """Primary (NIH) images followed by auxiliary (CheXpert) images, for joint training. Targets span
    both label sets, primary first; `mask` marks the outputs an image supervises (its own dataset's)."""

    def __init__(self, primary: CXRDataset, auxiliary: CXRDataset) -> None:
        self.primary, self.auxiliary = primary, auxiliary

    def __len__(self) -> int:
        return len(self.primary) + len(self.auxiliary)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        is_primary = idx < len(self.primary)
        sample = self.primary[idx] if is_primary else self.auxiliary[idx - len(self.primary)]
        own = sample["target"]
        other = torch.zeros(len(self.auxiliary.target_columns) if is_primary else len(self.primary.target_columns))
        own_mask, other_mask = torch.ones_like(own), torch.zeros_like(other)
        sample["target"] = torch.cat([own, other] if is_primary else [other, own])
        sample["mask"] = torch.cat([own_mask, other_mask] if is_primary else [other_mask, own_mask])
        return sample


class JointSampler(Sampler[int]):
    """Each epoch: every primary image once, plus `ratio` times as many auxiliary images drawn at
    random without replacement, all shuffled together."""

    def __init__(self, num_primary: int, num_auxiliary: int, ratio: float, generator: torch.Generator) -> None:
        self.num_primary, self.num_auxiliary, self.generator = num_primary, num_auxiliary, generator
        self.auxiliary_per_epoch = min(num_auxiliary, round(ratio * num_primary))

    def __len__(self) -> int:
        return self.num_primary + self.auxiliary_per_epoch

    def __iter__(self):
        auxiliary = torch.randperm(self.num_auxiliary, generator=self.generator)[:self.auxiliary_per_epoch] + self.num_primary
        order = torch.cat([torch.arange(self.num_primary), auxiliary])
        yield from order[torch.randperm(len(order), generator=self.generator)].tolist()


def build_datasets(
        bundle: DataBundle,
        train_transform=None,
        eval_transform=None,
        include_metadata: bool = False,
) -> dict[str, CXRDataset]:
    return {
        "train": CXRDataset(
            bundle.train_df,
            target_columns=bundle.target_columns,
            transform=train_transform,
            include_metadata=include_metadata,
        ),
        "val": CXRDataset(
            bundle.val_df,
            target_columns=bundle.target_columns,
            transform=eval_transform,
            include_metadata=include_metadata,
        ),
        "test": CXRDataset(
            bundle.test_df,
            target_columns=bundle.target_columns,
            transform=eval_transform,
            include_metadata=include_metadata,
        ),
    }