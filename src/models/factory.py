# Model construction from model_name in the YAML config

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Tiny_Weights,
    DenseNet121_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_S_Weights,
    MaxVit_T_Weights,
    ResNet50_Weights,
    Swin_V2_B_Weights,
    ViT_B_16_Weights,
    convnext_base,
    convnext_small,
    convnext_tiny,
    densenet121,
    efficientnet_v2_m,
    efficientnet_v2_s,
    maxvit_t,
    resnet50,
    swin_v2_b,
    vit_b_16,
)
from torchvision.models.swin_transformer import (
    PatchMergingV2,
    ShiftedWindowAttentionV2,
    SwinTransformer,
    SwinTransformerBlockV2,
)

from torchvision.ops import StochasticDepth

from utils import PROJECT_ROOT

# Weights-only copy of EVA-X-B (huggingface.co/MapleF/eva_x, eva_x_base_patch16_merged520k_mim.pt,
# MIT/Apache-2.0); the original is a DeepSpeed checkpoint with optimizer state.
EVA_X_BASE_WEIGHTS = "weights/eva_x/eva_x_base_patch16_mim_weights.pt"


def eva_x_base(weights: str | None = None, image_size: int = 224, stochastic_depth_prob: float = 0.0) -> nn.Module:
    """EVA-X ViT-B/16 (Yao et al., npj Digit Med 2025): the EVA-02 architecture with masked image
    modelling on ~520k chest X-rays (NIH, CheXpert, MIMIC-CXR; the authors exclude NIH test images).
    Built with timm's EVA implementation, which converts the original checkpoint layout; the position
    embedding is resampled to image_size and RoPE interpolated from its 14x14 pretraining grid."""
    from timm.models.eva import Eva, checkpoint_filter_fn

    model = Eva(img_size=image_size, patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_fused=False,
                mlp_ratio=4 * 2 / 3, swiglu_mlp=True, scale_mlp=True, use_rot_pos_emb=True,
                ref_feat_shape=(14, 14), global_pool="avg", drop_path_rate=stochastic_depth_prob)
    if weights is not None:
        state = torch.load(PROJECT_ROOT / weights, map_location="cpu", weights_only=True)["model"]
        result = model.load_state_dict(checkpoint_filter_fn(state, model), strict=False)
        # Pretrained by masked image modelling, so there is no classifier; build_model replaces it anyway.
        if set(result.missing_keys) != {"head.weight", "head.bias"} or result.unexpected_keys:
            raise RuntimeError(f"EVA-X checkpoint mismatch: missing {result.missing_keys}, unexpected {result.unexpected_keys}")
    return model


# Swin attention buffers computed from the window size, not learned; rebuilt when it changes.
SWIN_WINDOW_BUFFERS = ("relative_coords_table", "relative_position_index")


def swin_v2_b_window(window_size: int, weights: Swin_V2_B_Weights | None = None,
                     pretrained_window_size: int = 8) -> SwinTransformer:
    """swin_v2_b with a window_size x window_size attention window (torchvision fixes 8x8). At 448px,
    8x8 windows don't tile stages 3-4 (28x28, 14x14 tokens), which torchvision pads with zero tokens
    that attention does not mask; 14x14 windows tile every stage. Swin-V2's position bias is an MLP of
    log-spaced relative coordinates, so it carries over to the new window when the coordinates are
    normalized by the pretraining window (as in the official code), not the new one."""
    model = SwinTransformer(patch_size=[4, 4], embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
                            window_size=[window_size, window_size], stochastic_depth_prob=0.5,
                            block=SwinTransformerBlockV2, downsample_layer=PatchMergingV2)
    coords = torch.arange(-(window_size - 1), window_size, dtype=torch.float32)
    table = torch.stack(torch.meshgrid([coords, coords], indexing="ij")).permute(1, 2, 0).unsqueeze(0)
    table = table / (pretrained_window_size - 1) * 8
    table = torch.sign(table) * torch.log2(table.abs() + 1.0) / 3.0
    for module in model.modules():
        if isinstance(module, ShiftedWindowAttentionV2):
            module.relative_coords_table.copy_(table)
    if weights is not None:
        state = {key: value for key, value in weights.get_state_dict(progress=True).items()
                 if not key.endswith(SWIN_WINDOW_BUFFERS)}
        result = model.load_state_dict(state, strict=False)
        if any(not key.endswith(SWIN_WINDOW_BUFFERS) for key in result.missing_keys) or result.unexpected_keys:
            raise RuntimeError(f"Swin checkpoint mismatch: missing {result.missing_keys}, unexpected {result.unexpected_keys}")
    return model


# model_name -> (constructor, pretrained weights, attribute holding the classifier head)
ARCHITECTURES = {
    "densenet121": (densenet121, DenseNet121_Weights.DEFAULT, "classifier"),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT, "fc"),
    "convnext_tiny": (convnext_tiny, ConvNeXt_Tiny_Weights.DEFAULT, "classifier"),
    "convnext_small": (convnext_small, ConvNeXt_Small_Weights.DEFAULT, "classifier"),
    "convnext_base": (convnext_base, ConvNeXt_Base_Weights.DEFAULT, "classifier"),
    "efficientnet_v2_s": (efficientnet_v2_s, EfficientNet_V2_S_Weights.DEFAULT, "classifier"),
    "efficientnet_v2_m": (efficientnet_v2_m, EfficientNet_V2_M_Weights.DEFAULT, "classifier"),
    "swin_v2_b": (swin_v2_b, Swin_V2_B_Weights.DEFAULT, "head"),
    # MaxViT's window partitions are fixed at construction: image_size must be a multiple of 224.
    "maxvit_t": (maxvit_t, MaxVit_T_Weights.DEFAULT, "classifier"),
    # SWAG end-to-end weights are fixed at 384px (set image_size: 384); no stochastic depth.
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1, "heads"),
    # Chest X-ray pretrained (not ImageNet); see eva_x_base.
    "eva_x_base": (eva_x_base, EVA_X_BASE_WEIGHTS, "head"),
}


def _architecture(cfg: dict[str, Any]):
    model_name = str(cfg["model_name"]).lower()
    if model_name not in ARCHITECTURES:
        raise ValueError(f"Unsupported model_name '{model_name}'. Available: {', '.join(ARCHITECTURES)}")
    return ARCHITECTURES[model_name]


def build_model(cfg: dict[str, Any], num_classes: int) -> nn.Module:
    constructor, weights, head_attr = _architecture(cfg)
    kwargs = {}
    if cfg.get("drop_path") is not None:  # stochastic depth; not torchvision's ViT, DenseNet or ResNet
        kwargs["stochastic_depth_prob"] = float(cfg["drop_path"])
    pretrained = cfg.get("pretrained", True)
    image_size = int(cfg.get("image_size", 224))
    if (constructor is vit_b_16 and not pretrained) or constructor is eva_x_base:
        # Size the position embedding for the configured input (torchvision's ViT defaults to 224px
        # without weights), so it matches a domain checkpoint trained at that size.
        kwargs["image_size"] = image_size
    if constructor is maxvit_t:
        kwargs["input_size"] = (image_size, image_size)
    fixed_drop_path = kwargs.pop("stochastic_depth_prob", None) if constructor in (swin_v2_b, maxvit_t) else None
    if constructor is maxvit_t:
        # torchvision only loads MaxViT weights at 224px, but no weight depends on the input size
        # (attention windows are 7x7), so build at image_size and load them directly.
        model = constructor(weights=None, **kwargs)
        if pretrained:
            model.load_state_dict(weights.get_state_dict(progress=True))
    elif constructor is swin_v2_b and cfg.get("swin_window_size"):
        model = swin_v2_b_window(int(cfg["swin_window_size"]), weights if pretrained else None)
    else:
        model = constructor(weights=weights if pretrained else None, **kwargs)
    if fixed_drop_path is not None:
        _rescale_drop_path(model, fixed_drop_path)

    head = getattr(model, head_attr)
    if isinstance(head, nn.Sequential):  # ConvNeXt: LayerNorm -> Flatten -> Linear; EfficientNet: Dropout -> Linear
        head[-1] = nn.Linear(head[-1].in_features, num_classes)
    else:
        setattr(model, head_attr, nn.Linear(head.in_features, num_classes))
    return model


def _rescale_drop_path(model: nn.Module, drop_path: float) -> None:
    """torchvision's swin_v2_b and maxvit_t fix their stochastic depth (0.5 and 0.2) and reject the
    argument; scale the builder's per-block schedule so its maximum is drop_path."""
    layers = [module for module in model.modules() if isinstance(module, StochasticDepth)]
    peak = max(layer.p for layer in layers)
    for layer in layers:
        layer.p = layer.p * drop_path / peak if peak > 0 else drop_path


def build_parameter_groups(model: nn.Module, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Backbone and head groups, head last. With optimizer.layer_decay (ViT-style models and Swin), the
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

    # Depth of each block's parameter-name prefix (1..num_layers) and the final-norm prefixes:
    # torchvision ViT, timm ViT, then torchvision Swin.
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        prefixes = {f"encoder.layers.{name}.": index + 1 for index, name in enumerate(model.encoder.layers._modules)}
        final_prefixes = ("encoder.ln",)
    elif hasattr(model, "blocks"):
        prefixes = {f"blocks.{index}.": index + 1 for index in range(len(model.blocks))}
        final_prefixes = ("norm.", "fc_norm.")
    elif isinstance(model, SwinTransformer):
        # features.0 is the patch embedding; stages of blocks alternate with patch merging, which
        # shares the depth of the block that follows it.
        prefixes, block = {}, 0
        for index, stage in enumerate(list(model.features)[1:], start=1):
            if isinstance(stage, nn.Sequential):
                for position in range(len(stage)):
                    block += 1
                    prefixes[f"features.{index}.{position}."] = block
            else:
                prefixes[f"features.{index}."] = block + 1
        final_prefixes = ("norm.",)
    else:
        raise ValueError("optimizer.layer_decay is only supported for ViT-style models and Swin.")
    num_layers = max(prefixes.values())

    def depth(name: str) -> int:  # 0 = patch embedding, 1..num_layers = blocks, num_layers + 1 = final norm
        for prefix, layer in prefixes.items():
            if name.startswith(prefix):
                return layer
        if name.startswith(final_prefixes):
            return num_layers + 1
        return 0

    groups: dict[tuple[int, bool], dict[str, Any]] = {}
    for name, param in model.named_parameters():
        if id(param) in head_ids:
            continue
        layer = depth(name)
        # Standard ViT fine-tuning: no weight decay on biases, norms, position embedding, class token
        # (nor Swin-V2's attention temperatures).
        decay = (param.ndim > 1 and name not in {"class_token", "encoder.pos_embedding", "cls_token", "pos_embed"}
                 and not name.endswith("logit_scale"))
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

    # model_state_dict: common external format; model: this project's checkpoints/best.pt
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
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
    rebuilt_buffer_keys: list[str] = []
    unexpected_keys: list[str] = []
    for key, tensor in checkpoint_state.items():
        target = model_state.get(key)
        if (target is not None and tensor.ndim == 4 and tensor.shape[1] == 1 and target.shape[1] == 3
                and tensor.shape[0] == target.shape[0] and tensor.shape[2:] == target.shape[2:]):
            # Grayscale stem (e.g. TorchXRayVision): our images are grey replicated to RGB,
            # so spreading the weights over three channels gives the same response.
            tensor = tensor.repeat(1, 3, 1, 1) / 3
        if key == "pos_embed" and target is not None and tensor.shape != target.shape:
            # timm ViT pretrained at another image size (e.g. CheXpert at 384px, NIH at 448px):
            # resample the patch-grid position embedding, keeping the class token's.
            from timm.layers import resample_abs_pos_embed
            prefix = int(getattr(model, "num_prefix_tokens", 1))
            grid = int(round((target.shape[1] - prefix) ** 0.5))
            tensor = resample_abs_pos_embed(tensor, new_size=(grid, grid), num_prefix_tokens=prefix)
        if backbone_only and key.startswith(head_prefixes):
            skipped_head_keys.append(key)
        elif key not in model_state:
            unexpected_keys.append(key)
        elif model_state[key].shape != tensor.shape and key.endswith(SWIN_WINDOW_BUFFERS):
            rebuilt_buffer_keys.append(key)  # pretrained with another window (swin_window_size); keep the model's
        elif model_state[key].shape != tensor.shape:
            shape_mismatch_keys.append(key)
        else:
            compatible_state[key] = tensor

    backbone_keys = [key for key in model_state if not key.startswith(head_prefixes) and key not in rebuilt_buffer_keys]
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
        "rebuilt_window_buffer_count": len(rebuilt_buffer_keys),
        "unexpected_checkpoint_keys": unexpected_keys,
        "missing_model_keys": list(incompatible.missing_keys),
    }
