from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re


SEED_SUFFIX_RE = re.compile(r"_seed\d+$")


@dataclass(frozen=True)
class RunLayout:
    output_root: Path
    run_group: str
    seed_dir: str
    run_name: str
    run_dir: Path


def _derive_run_name(
        cfg: Mapping[str, object],
        config_path: Path,
) -> str:
    configured_name = cfg.get("run_name")
    if configured_name:
        return str(configured_name)
    return config_path.stem


def _derive_run_group(
        run_name: str,
        cfg: Mapping[str, object],
) -> str:
    configured_group = cfg.get("run_group")
    if configured_group:
        return str(configured_group)

    derived_group = SEED_SUFFIX_RE.sub("", run_name)
    if not derived_group:
        raise ValueError(f"Could not derive run_group from run_name: {run_name}")

    return derived_group


def _derive_seed_dir(cfg: Mapping[str, object]) -> str:
    configured_seed_dir = cfg.get("seed_dir")
    if configured_seed_dir:
        return str(configured_seed_dir)

    if "train_seed" not in cfg:
        raise KeyError("Config must contain train_seed to derive seed_dir.")

    return f"seed{int(cfg['train_seed'])}"


def build_run_layout(
        *,
        cfg: Mapping[str, object],
        project_root: Path,
        config_path: str | Path,
) -> RunLayout:
    config_path = Path(config_path).resolve()

    output_root = Path(str(cfg["output_dir"]))
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()

    run_name = _derive_run_name(cfg, config_path)
    run_group = _derive_run_group(run_name, cfg)
    seed_dir = _derive_seed_dir(cfg)
    run_dir = (output_root / run_group / seed_dir).resolve()

    return RunLayout(
        output_root=output_root,
        run_group=run_group,
        seed_dir=seed_dir,
        run_name=run_name,
        run_dir=run_dir,
    )