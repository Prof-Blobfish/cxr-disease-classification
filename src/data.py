from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import RANDOM_SEED, VAL_SPLIT, dataset_path


@dataclass
class DataBundle:
    all_df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    bbox_df: pd.DataFrame # currently unused for classification, but may be useful for future localization tasks
    class_names: list[str]
    label_to_index: dict[str, int]
    target_columns: list[str]
    pos_weights: np.ndarray # for handling class imbalance during training

def prepare_data(
        root: str | Path | None = None,
        val_fraction: float = VAL_SPLIT,
        split_seed: int = RANDOM_SEED,
) -> DataBundle:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    
    dataset_root = _resolve_root(root)

    metadata_df = _load_metadata(dataset_root)
    image_lookup = _build_image_lookup(dataset_root)
    metadata_df = _attach_image_paths(metadata_df, image_lookup)

    metadata_df, class_names, target_columns = _encode_targets(metadata_df)

    bbox_df = _load_bbox_annotations(dataset_root)
    bbox_summary_df = _summarize_bbox_annotations(bbox_df)

    metadata_df = metadata_df.merge(bbox_summary_df, on="image_id", how="left")
    metadata_df["bbox_count"] = metadata_df["bbox_count"].fillna(0).astype(int)
    metadata_df["has_bbox"] = metadata_df["has_bbox"].fillna(False).astype(bool)

    train_df, val_df, test_df = _apply_splits(
        metadata_df,
        dataset_root,
        val_fraction,
        split_seed,
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    label_to_index = {
        label: index for index, label in enumerate(class_names)
    }

    pos_weights = _compute_pos_weight(train_df, target_columns)

    return DataBundle(
        all_df=all_df,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        bbox_df=bbox_df,
        class_names=class_names,
        label_to_index=label_to_index,
        target_columns=target_columns,
        pos_weights=pos_weights,
    )


    
# Standardize dataset loc into Path object
def _resolve_root(root: str | Path | None) -> Path:
    dataset_root = Path(dataset_path if root is None else root).expanduser().resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    return dataset_root

# Loads metadata, normalizes column names
def _load_metadata(root: Path) -> pd.DataFrame:
    metadata_path = root / "Data_Entry_2017.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)].copy()
    df.columns = df.columns.str.strip()

    rename_map = {
        "Image Index": "image_id",
        "Finding Labels": "finding_labels",
        "Follow-up #": "follow_up",
        "Patient ID": "patient_id",
        "Patient Age": "patient_age",
        "Patient Gender": "patient_gender",
        "View Position": "view_position",
        "OriginalImage[Width": "orig_width",
        "Height]": "orig_height",
        "OriginalImagePixelSpacing[x": "pixel_spacing_x",
        "y]": "pixel_spacing_y",
    }
    df = df.rename(columns=rename_map)

    required_columns = [
        "image_id",
        "finding_labels",
        "follow_up",
        "patient_id",
        "patient_age",
        "patient_gender",
        "view_position",
    ]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in metadata: {missing_columns}")
    
    numeric_columns = [
        "follow_up",
        "patient_id",
        "patient_age",
        "orig_width",
        "orig_height",
        "pixel_spacing_x",
        "pixel_spacing_y",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df

# Builds a directionary from iamge filename to absolute path for quick lookup when attaching image paths to the metadata dataframe
def _build_image_lookup(root: Path) -> dict[str, Path]:
    image_lookup: dict[str, Path] = {}

    for image_path in root.glob("images_*/images/*.png"):
        if image_path.name.startswith("._"):
            continue

        if image_path.name in image_lookup:
            raise ValueError(f"Duplicate image ID found: {image_path.name}")
        
        image_lookup[image_path.name] = image_path.resolve()

    if not image_lookup:
        raise FileNotFoundError(f"No images found under images_*/images")
    
    return image_lookup

# Attaches absolute image paths to the metadata dataframe using the image lookup dictionary
def _attach_image_paths(
        df: pd.DataFrame,
        image_lookup: dict[str, Path],
) -> pd.DataFrame:
    metadata_df = df.copy()
    metadata_df["image_path"] = metadata_df["image_id"].map(image_lookup)

    missing_mask = metadata_df["image_path"].isna()
    if missing_mask.any():
        missing_ids = metadata_df.loc[missing_mask, "image_id"].head(10).tolist()
        raise ValueError(f"Could not resolve image paths for image ids such as: {missing_ids}")
    
    metadata_df["image_path"] = metadata_df["image_path"].astype(str)
    return metadata_df
    
# Encodes the multi-label "finding_labels" column into separate binary columns for each class, returns updated dataframe, list of class names, and list of targets
def _encode_targets(
        df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    metadata_df = df.copy()

    def parse_label_list(raw_value: str) -> list[str]:
        labels = [label.strip() for label in str(raw_value).split("|") if label.strip()]
        labels = [label for label in labels if label != "No Finding"]
        return sorted(set(labels))
    
    def slug(label: str) -> str:
        return label.lower().replace(" ", "_").replace("-", "_")

    metadata_df["label_list"] = metadata_df["finding_labels"].fillna("").apply(parse_label_list)
    metadata_df["is_no_finding"] = metadata_df["label_list"].str.len().eq(0)

    class_names = sorted({label for labels in metadata_df["label_list"] for label in labels})
    target_columns = [f"target_{slug(label)}" for label in class_names]

    for label, target_column in zip(class_names, target_columns):
        metadata_df[target_column] = metadata_df["label_list"].apply(lambda labels: int(label in labels))

    return metadata_df, class_names, target_columns
    
# Loads bounding box annotations from the dataset, returns a dataframe with columns: image_id, finding_label, x_min, y_min, x_max, y_max
def _load_bbox_annotations(root: Path) -> pd.DataFrame:
    bbox_path = root / "BBox_List_2017.csv"
    if not bbox_path.exists():
        return pd.DataFrame(columns=["image_id", "bbox_label", "bbox_x", "bbox_y", "bbox_w", "bbox_h"])
    
    df = pd.read_csv(bbox_path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)].copy()
    df.columns = df.columns.str.strip()

    rename_map = {
        "Image Index": "image_id",
        "Finding Label": "bbox_label",
        "Bbox [x": "bbox_x",
        "y": "bbox_y",
        "w": "bbox_w",
        "h]": "bbox_h",
    }
    df = df.rename(columns=rename_map)

    for column in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df

# Summarizes the bounding box annotations by counting how many boxes each image has, whether it has any boxes at all, and which unique labels are present in the boxes for each image
def _summarize_bbox_annotations(bbox_df: pd.DataFrame) -> pd.DataFrame:
    if bbox_df.empty:
        return pd.DataFrame(columns=["image_id", "bbox_count", "has_bbox", "bbox_labels"])
    
    summary = (
        bbox_df.groupby("image_id", as_index=False)
        .agg(
            bbox_count=("image_id", "size"),
            bbox_labels=("bbox_label", lambda labels: sorted(set(labels))),
        )
    )
    summary["has_bbox"] = summary["bbox_count"].gt(0)
    return summary

# Applies a patient-level split to the metadata dataframe, ensuring that all images from the same patient are in the same split, returns train/val/test dataframes
def _apply_splits(
        df: pd.DataFrame,
        root: Path,
        val_fraction: float,
        split_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val_path = root / "train_val_list.txt"
    test_path = root / "test_list.txt"

    if not train_val_path.exists():
        raise FileNotFoundError(f"Missing split file: {train_val_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing split file: {test_path}")
    
    with train_val_path.open("r", encoding="utf-8") as handle:
        train_val_ids = {line.strip() for line in handle if line.strip()}

    with test_path.open("r", encoding="utf-8") as handle:
        test_ids = {line.strip() for line in handle if line.strip()}

    overlap = train_val_ids & test_ids
    if overlap:
        raise ValueError("train_val_list.txt and test_list.txt overlap")
    
    train_val_df = df[df["image_id"].isin(train_val_ids)].copy()
    test_df = df[df["image_id"].isin(test_ids)].copy()

    patient_ids = train_val_df["patient_id"].dropna().astype(int).unique()
    patient_ids = np.array(sorted(patient_ids))

    rng = np.random.default_rng(split_seed)
    rng.shuffle(patient_ids)

    val_count = max(1, int(round(len(patient_ids) * val_fraction)))
    val_patients = set(patient_ids[:val_count])

    val_df = train_val_df[train_val_df["patient_id"].isin(val_patients)].copy()
    train_df = train_val_df[~train_val_df["patient_id"].isin(val_patients)].copy()

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )
    
def _compute_pos_weight(
        train_df: pd.DataFrame,
        target_columns: list[str],
) -> np.ndarray:
    positive_counts = train_df[target_columns].sum(axis=0).to_numpy(dtype=np.float32)
    negative_counts = len(train_df) - positive_counts

    pos_weights = negative_counts / np.clip(positive_counts, a_min=1.0, a_max=None)
    return pos_weights.astype(np.float32)