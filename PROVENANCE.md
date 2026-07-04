# Run19 Export Provenance

This directory is a clean Run19-only export from `/home/pythonuser/pythonproject/bigmodel`.

## What This Proves

- Final best model config source: `/home/pythonuser/pythonproject/bigmodel/checkpoints/ppo_e2e_run19_sl15_valrandom/config.json`
- Exported final model config: `/home/pythonuser/pythonproject/bigmodel/run19_export/model/run19_config.json`
- Effective config equality, ignoring only sanitized `email` and export-local `run.output_dir`: `True`
- Training config source: `/home/pythonuser/pythonproject/bigmodel/configs/e2e_ppo_run19_sl15_valrandom.json`
- Exported training config: `/home/pythonuser/pythonproject/bigmodel/run19_export/training/configs/e2e_ppo_run19_sl15_valrandom.json`
- Training config equality, ignoring only sanitized `email` and normalized `student_init.checkpoint`: `True`

## Final Run19 Best Config Highlights

```text
phase.mode = stage3_joint
data.seq_length = 300
model.hidden_dim = 256
model.num_layers = 4
model.use_asset_ids = False
teacher.enabled = True
teacher.distill_mode = representation
teacher.repr_cache_dir = data/cache/kron_repr
teacher.repr_dim = 512
teacher.repr_coef_start = 0.05
teacher.repr_coef_end = 0.0
teacher.repr_decay_updates = 1000
teacher.kl_coef_start = 0.0
teacher.strength_loss_weight = 0.0
teacher.teacher_logit_weight = 0.0
ppo.max_updates = 2000
ppo.trajectory_length = 150
ppo.envs_per_batch = 60
env.stop_loss_vol_multiple = 1.5
env.max_holding_steps = 40
env.entry_threshold = 0.2
env.position_scale = 1.3
```

## Included Source Areas

- `training/`: Run19 PPO training, teacher cache build, sanitized Run19 configs, start scripts.
- `core/`: helper modules imported by training (`EMA`, optional email notifier with empty defaults).
- `kronos_vendor/`: teacher model code needed to rebuild Kron representation cache.
- `ppo_trader/`: inference/session/Freqtrade adapter package source.
- `freqtrade/`: current `PPORun19SessionStrategy` and Run19 session strategy config templates.
- `evaluation/`: native Run19 and Freqtrade comparison/scan scripts.
- `docs/`: extracted Run19 training notes.

## Explicitly Excluded

- Market data parquet files.
- Teacher cache `.npy` files.
- Checkpoint binaries such as `best.pt` and `latest.pt`.
- Logs, CSV result files, SQLite databases, wheel files, pycache.

## Artifact Handling

The final effective model binary in the source workspace is:

```text
checkpoints/ppo_e2e_run19_sl15_valrandom/best.pt
```

It is intentionally not included because you requested code/config only, no data or large artifacts.
Use Git LFS or an external artifact store if this binary must be versioned.

## Hash Manifest

See `MANIFEST.sha256` for SHA256 checksums of exported files, excluding `.git` internals.
