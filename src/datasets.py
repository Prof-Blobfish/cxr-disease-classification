# Reads an image from image_path and returns image tensor, target tensor, and maybe image_id

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
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