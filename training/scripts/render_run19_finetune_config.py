#!/usr/bin/env python3
"""Render a Run19 finetune config from .env-managed local paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict


ROOT_DIR = Path(__file__).resolve().parents[2]


def _read_env_file(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    quote_chars = {chr(34), chr(39)}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in quote_chars:
            value = value[1:-1]
        if key:
            env[key] = value
    return env


def _env(env: Dict[str, str], key: str, default: str = "") -> str:
    return os.environ.get(key, env.get(key, default)).strip()


def _resolve(value: str) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    return str(path)


def _set_int(cfg: Dict, section: str, key: str, value: str) -> None:
    if value:
        cfg[section][key] = int(value)


def _set_float(cfg: Dict, section: str, key: str, value: str) -> None:
    if value:
        cfg[section][key] = float(value)


def _set_bool(cfg: Dict, section: str, key: str, value: str) -> None:
    if not value:
        return
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        cfg[section][key] = True
    elif normalized in {"0", "false", "no", "off"}:
        cfg[section][key] = False
    else:
        raise ValueError(f"Invalid boolean for {section}.{key}: {value}")


def _validate_existing(path: str, label: str, must_be_dir: bool = False) -> None:
    if not path:
        return
    p = Path(path)
    if must_be_dir and not p.is_dir():
        raise FileNotFoundError(f"{label} not found or not a directory: {path}")
    if not must_be_dir and not p.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def render(env_file: Path) -> Path:
    file_env = _read_env_file(env_file)

    base_config = Path(_resolve(_env(file_env, "RUN19_BASE_CONFIG", "training/configs/e2e_ppo_run19_finetune_from_best.json")))
    with base_config.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    data_file = _resolve(_env(file_env, "RUN19_DATA_FILE", cfg["data"]["data_file"]))
    repr_cache_dir = _resolve(_env(file_env, "RUN19_REPR_CACHE_DIR", cfg["teacher"]["repr_cache_dir"]))
    strength_cache_dir = _resolve(_env(file_env, "RUN19_STRENGTH_CACHE_DIR", cfg["teacher"]["cache_dir"]))
    init_checkpoint = _resolve(_env(file_env, "RUN19_INIT_CHECKPOINT", cfg["student_init"]["checkpoint"]))
    output_dir = _resolve(_env(file_env, "RUN19_OUTPUT_DIR", cfg["run"]["output_dir"]))
    resume = _resolve(_env(file_env, "RUN19_RESUME", cfg["run"].get("resume", "")))

    _validate_existing(data_file, "RUN19_DATA_FILE")
    _validate_existing(repr_cache_dir, "RUN19_REPR_CACHE_DIR", must_be_dir=True)
    _validate_existing(strength_cache_dir, "RUN19_STRENGTH_CACHE_DIR", must_be_dir=True)
    _validate_existing(init_checkpoint, "RUN19_INIT_CHECKPOINT")
    if resume:
        _validate_existing(resume, "RUN19_RESUME")

    cfg["data"]["data_file"] = data_file
    cfg["teacher"]["repr_cache_dir"] = repr_cache_dir
    cfg["teacher"]["cache_dir"] = strength_cache_dir
    cfg["student_init"]["checkpoint"] = init_checkpoint
    cfg["run"]["output_dir"] = output_dir
    cfg["run"]["resume"] = resume
    cfg["run"]["policy_device"] = _env(file_env, "RUN19_POLICY_DEVICE", cfg["run"]["policy_device"])
    cfg["run"]["value_device"] = _env(file_env, "RUN19_VALUE_DEVICE", cfg["run"]["value_device"])

    _set_int(cfg, "ppo", "max_updates", _env(file_env, "RUN19_MAX_UPDATES"))
    _set_int(cfg, "data", "seq_length", _env(file_env, "RUN19_SEQ_LENGTH"))
    _set_bool(cfg, "model", "use_asset_ids", _env(file_env, "RUN19_USE_ASSET_IDS"))
    _set_bool(cfg, "teacher", "enabled", _env(file_env, "RUN19_TEACHER_ENABLED"))
    _set_float(cfg, "ppo", "policy_lr", _env(file_env, "RUN19_POLICY_LR"))
    _set_float(cfg, "ppo", "value_lr", _env(file_env, "RUN19_VALUE_LR"))
    _set_int(cfg, "ppo", "mini_batch_size", _env(file_env, "RUN19_MINI_BATCH_SIZE"))
    _set_int(cfg, "ppo", "value_infer_batch_size", _env(file_env, "RUN19_VALUE_INFER_BATCH_SIZE"))
    _set_bool(cfg, "ppo", "parallel_value_policy_update", _env(file_env, "RUN19_PARALLEL_VALUE_POLICY_UPDATE"))
    _set_int(cfg, "ppo", "val_interval", _env(file_env, "RUN19_VAL_INTERVAL"))
    _set_int(cfg, "ppo", "save_interval", _env(file_env, "RUN19_SAVE_INTERVAL"))
    _set_int(cfg, "ppo", "log_interval", _env(file_env, "RUN19_LOG_INTERVAL"))

    cache_splits = _env(file_env, "RUN19_CACHE_SPLITS")
    if cache_splits:
        cfg["teacher"]["cache_splits"] = [s.strip() for s in cache_splits.split(",") if s.strip()]

    symbols = _env(file_env, "RUN19_SYMBOLS")
    if symbols:
        cfg["data"]["symbols"] = [s.strip() for s in symbols.split(",") if s.strip()]

    resolved_config = Path(_resolve(_env(file_env, "RUN19_RESOLVED_CONFIG", str(Path(output_dir) / "config.resolved.json"))))
    resolved_config.parent.mkdir(parents=True, exist_ok=True)
    with resolved_config.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return resolved_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=".env", help="Path to the .env file")
    args = parser.parse_args()
    print(render(Path(args.env_file).expanduser()))


if __name__ == "__main__":
    main()
