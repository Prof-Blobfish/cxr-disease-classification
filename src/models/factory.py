# Model construction from model_name in the YAML config

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    DenseNet121_Weights,
    EfficientNet_V2_S_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    convnext_small,
    convnext_tiny,
    densenet121,
    efficientnet_v2_s,
    resnet50,
    vit_b_16,
)

# model_name -> (constructor, ImageNet weights, attribute holding the classifier head)
ARCHITECTURES = {
    "densenet121": (densenet121, DenseNet121_Weights.DEFAULT, "classifier"),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT, "fc"),
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT, "classifier"),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT, "classifier"),
    "efficientnet_v2_s": (efficientnet_v2_s, EfficientNet_V2_S_Weights.DEFAULT, "classifier"),
    # SWAG end-to-end weights are fixed at 384px (set image_size: 384); no stochastic depth.
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1, "heads"),
}


def _architecture(cfg: dict[str, Any]):
    model_name = str(cfg["model_name"]).lower()
    if model_name not in ARCHITECTURES:
        raise ValueError(f"Unsupported model_name '{model_name}'. Available: {', '.join(ARCHITECTURES)}")
    return ARCHITECTURES[model_name]


def build_model(cfg: dict[str, Any], num_classes: int) -> nn.Module:
    constructor, weights, head_attr = _architecture(cfg)
    kwargs = {}
    if cfg.get("drop_path") is not None:  # stochastic depth; ConvNeXt and EfficientNet only
        kwargs["stochastic_depth_prob"] = float(cfg["drop_path"])
    model = constructor(weights=weights if cfg.get("pretrained", True) else None, **kwargs)

    head = getattr(model, head_attr)
    if isinstance(head, nn.Sequential):  # ConvNeXt: LayerNorm -> Flatten -> Linear; EfficientNet: Dropout -> Linear
        head[-1] = nn.Linear(head[-1].in_features, num_classes)
    else:
        setattr(model, head_attr, nn.Linear(head.in_features, num_classes))
    return model


def build_parameter_groups(model: nn.Module, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Backbone and head groups, head last. With optimizer.layer_decay (ViT only), the
    backbone is split per transformer block, each block's LR scaled by decay ** depth-from-top."""
    _, _, head_attr = _architecture(cfg)
    optimizer_cfg = cfg["optimizer"]
    head_ids = {id(param) for param in getattr(model, head_attr).parameters()}
    weight_decay = float(optimizer_cfg["weight_decay"])
    backbone_lr = float(optimizer_cfg["backbone_lr"])
    head_group = {
        "params": list(getattr(model, head_attr).parameters()),
        "lr": float(optimizer_cfg["head_lr"]),
        "weight_decay": weight_decay,
    }

    layer_decay = optimizer_cfg.get("layer_decay")
    if layer_decay is None:
        backbone = [param for param in model.parameters() if id(param) not in head_ids]
        return [{"params": backbone, "lr": backbone_lr, "weight_decay": weight_decay}, head_group]

    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layers"):
        raise ValueError("optimizer.layer_decay is only supported for ViT models.")
    num_layers = len(model.encoder.layers)

    def depth(name: str) -> int:  # 0 = patch embedding, 1..num_layers = blocks, num_layers + 1 = final norm
        if name.startswith("encoder.layers."):
            return int(name.split(".")[2].removeprefix("encoder_layer_")) + 1
        if name.startswith("encoder.ln"):
            return num_layers + 1
        return 0

    groups: dict[tuple[int, bool], dict[str, Any]] = {}
    for name, param in model.named_parameters():
        if id(param) in head_ids:
            continue
        layer = depth(name)
        # Standard ViT fine-tuning: no weight decay on biases, norms, position embedding, class token.
        decay = param.ndim > 1 and name not in {"class_token", "encoder.pos_embedding"}
        group = groups.setdefault((layer, decay), {
            "params": [],
            "lr": backbone_lr * float(layer_decay) ** (num_layers + 1 - layer),
            "weight_decay": weight_decay if decay else 0.0,
        })
        group["params"].append(param)
    return [groups[key] for key in sorted(groups)] + [head_group]


def _checkpoint_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("Domain checkpoint must be a mapping or contain model_state_dict.")

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError("Domain checkpoint model_state_dict must be a mapping.")

    normalized: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if isinstance(key, str) and isinstance(value, torch.Tensor):
            normalized[key.removeprefix("module.").removeprefix("model.")] = value
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_domain_pretrained_checkpoint(
        model: nn.Module,
        cfg: dict[str, Any],
        project_root: str | Path,
) -> dict[str, Any] | None:
    init_cfg = cfg.get("domain_pretrained")
    if init_cfg is None:
        return None
    if not isinstance(init_cfg, dict):
        raise TypeError("domain_pretrained must be a YAML mapping.")
    if bool(cfg.get("pretrained", True)):
        raise ValueError(
            "Set pretrained: false when domain_pretrained is configured so the "
            "initialization source is unambiguous."
        )

    configured_path = init_cfg.get("path")
    if not configured_path:
        raise ValueError("domain_pretrained.path is required when domain_pretrained is configured.")

    checkpoint_path = Path(str(configured_path)).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(project_root) / checkpoint_path
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Domain checkpoint not found: {checkpoint_path}")

    actual_sha256 = _sha256(checkpoint_path)
    expected_sha256 = init_cfg.get("sha256")
    if expected_sha256 and actual_sha256.lower() != str(expected_sha256).lower():
        raise ValueError(
            "Domain checkpoint SHA-256 does not match domain_pretrained.sha256. "
            f"Expected {expected_sha256}, got {actual_sha256}."
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_state = _checkpoint_state_dict(checkpoint)
    model_state = model.state_dict()
    backbone_only = bool(init_cfg.get("backbone_only", True))
    head_prefixes = (_architecture(cfg)[2] + ".",)

    compatible_state: dict[str, torch.Tensor] = {}
    skipped_head_keys: list[str] = []
    shape_mismatch_keys: list[str] = []
    unexpected_keys: list[str] = []
    for key, tensor in checkpoint_state.items():
        target = model_state.get(key)
        if (target is not None and tensor.ndim == 4 and tensor.shape[1] == 1 and target.shape[1] == 3
                and tensor.shape[0] == target.shape[0] and tensor.shape[2:] == target.shape[2:]):
            # Grayscale stem (e.g. TorchXRayVision): our images are grey replicated to RGB,
            # so spreading the weights over three channels gives the same response.
            tensor = tensor.repeat(1, 3, 1, 1) / 3
        if backbone_only and key.startswith(head_prefixes):
            skipped_head_keys.append(key)
        elif key not in model_state:
            unexpected_keys.append(key)
        elif model_state[key].shape != tensor.shape:
            shape_mismatch_keys.append(key)
        else:
            compatible_state[key] = tensor

    backbone_keys = [key for key in model_state if not key.startswith(head_prefixes)]
    matched_backbone_keys = [key for key in compatible_state if key in backbone_keys]
    match_ratio = len(matched_backbone_keys) / max(len(backbone_keys), 1)
    min_match_ratio = float(init_cfg.get("min_backbone_match_ratio", 0.95))
    if not 0.0 < min_match_ratio <= 1.0:
        raise ValueError("domain_pretrained.min_backbone_match_ratio must be in (0, 1].")
    if match_ratio < min_match_ratio:
        raise ValueError(
            "Domain checkpoint did not match enough backbone tensors: "
            f"{len(matched_backbone_keys)}/{len(backbone_keys)} ({match_ratio:.1%}); "
            f"required {min_match_ratio:.1%}."
        )

    incompatible = model.load_state_dict(compatible_state, strict=False)
    missing_backbone_keys = [key for key in incompatible.missing_keys if key in backbone_keys]
    if missing_backbone_keys:
        raise RuntimeError(
            "Domain checkpoint left backbone tensors uninitialized despite passing compatibility checks: "
            f"{missing_backbone_keys[:10]}"
        )

    return {
        "source": str(init_cfg.get("source", "unspecified")),
        "path": str(checkpoint_path),
        "sha256": actual_sha256,
        "backbone_only": backbone_only,
        "min_backbone_match_ratio": min_match_ratio,
        "matched_backbone_tensor_count": len(matched_backbone_keys),
        "total_backbone_tensor_count": len(backbone_keys),
        "matched_backbone_tensor_ratio": match_ratio,
        "matched_tensor_count": len(compatible_state),
        "skipped_head_tensor_count": len(skipped_head_keys),
        "shape_mismatch_keys": shape_mismatch_keys,
        "unexpected_checkpoint_keys": unexpected_keys,
        "missing_model_keys": list(incompatible.missing_keys),
    }
