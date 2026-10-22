# Train and eval transform builders from the YAML config

from __future__ import annotations

from typing import Any, Callable, Sequence

from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode


DEFAULT_IMAGE_SIZE = 256
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)

ImageTransform = Callable[[Image.Image], Tensor]

def _get_split_cfg(cfg: dict[str, Any], split: str) -> dict[str, Any]:
    return cfg.get("augmentation", {}).get(split, {}) or {}

def _as_pair(
        value: float | int | Sequence[float] | None,
        default: tuple[float, float],
) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return (float(value), float(value))
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (float(value[0]), float(value[1]))
    raise ValueError(f"Expected a scalar or 2-item list/tuple, got: {value}")

def _scale_from_delta(scale_delta: float | int | None) -> tuple[float, float]:
    delta = 0.0 if scale_delta is None else float(scale_delta)
    
    if delta < 0:
        raise ValueError(f"scale_delta must be >= 0, got: {delta}")
    if delta >= 1.0:
        raise ValueError(f"scale_delta must be < 1.0, got: {delta}")
    
    return (1.0 - delta, 1.0 + delta)

def build_train_transform(cfg: dict[str, Any]) -> ImageTransform:
    image_size = int(cfg.get("image_size", DEFAULT_IMAGE_SIZE))
    train_cfg = _get_split_cfg(cfg, "train")

    resize = int(train_cfg.get("resize", image_size))
    crop_size = train_cfg.get("crop_size")
    horizontal_flip = bool(train_cfg.get("horizontal_flip", False))

    rotation_deg = float(train_cfg.get("rotation_deg", 0.0))
    translate = _as_pair(train_cfg.get("translate"), (0.0, 0.0))
    scale_delta = float(train_cfg.get("scale_delta", 0.0))
    scale = _scale_from_delta(scale_delta)

    brightness = float(train_cfg.get("brightness", 0.0))
    contrast = float(train_cfg.get("contrast", 0.0))

    normalization_cfg = cfg.get("normalization", {})
    mean = tuple(normalization_cfg.get("mean", DEFAULT_MEAN))
    std = tuple(normalization_cfg.get("std", DEFAULT_STD))

    crop_scale = train_cfg.get("random_resized_crop_scale")
    if crop_scale is not None:
        # Zoom in on a random 70-100% (say) of the image, then resize; mild aspect jitter.
        steps: list[Callable[..., Any]] = [
            transforms.RandomResizedCrop(
                resize,
                scale=_as_pair(crop_scale, (1.0, 1.0)),
                ratio=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
            )
        ]
    else:
        steps = [
            transforms.Resize(
                (resize, resize),
                interpolation=InterpolationMode.BILINEAR,
            )
        ]

    if crop_size is not None:
        steps.append(transforms.RandomCrop(int(crop_size)))

    if horizontal_flip:
        steps.append(transforms.RandomHorizontalFlip(p=0.5))

    use_affine = (
        rotation_deg > 0
        or  translate != (0.0, 0.0)
        or scale_delta > 0
    )
    if use_affine:
        affine_kwargs: dict[str, Any] = {
            "degrees": rotation_deg,
            "interpolation": InterpolationMode.BILINEAR,
            "fill": 0,
        }

        if translate != (0.0, 0.0):
            affine_kwargs["translate"] = translate

        if scale_delta > 0:
            affine_kwargs["scale"] = scale

        steps.append(transforms.RandomAffine(**affine_kwargs))

    if brightness > 0 or contrast > 0:
        steps.append(transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
        ))

    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(steps)

def build_eval_transform(cfg: dict[str, Any]) -> ImageTransform:
    image_size = int(cfg.get("image_size", DEFAULT_IMAGE_SIZE))
    eval_cfg = _get_split_cfg(cfg, "eval")

    resize = int(eval_cfg.get("resize", image_size))
    crop_size = eval_cfg.get("crop_size")

    normalization_cfg = cfg.get("normalization", {})
    mean = tuple(normalization_cfg.get("mean", DEFAULT_MEAN))
    std = tuple(normalization_cfg.get("std", DEFAULT_STD))

    steps: list[Callable[..., Any]] = [
        transforms.Resize(
            (resize, resize),
            interpolation=InterpolationMode.BILINEAR,
        )
    ]

    if crop_size is not None:
        steps.append(transforms.CenterCrop(int(crop_size)))

    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(steps)

def build_transforms(cfg: dict[str, Any]) -> dict[str, ImageTransform]:
    return {
        "train": build_train_transform(cfg),
        "eval": build_eval_transform(cfg),
    }