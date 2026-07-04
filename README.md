# Run19 Export

Clean extraction of the Run19 training and runtime code from the original workspace.

Included:

- `training/`: Run19 PPO training code, teacher cache builder, sanitized configs, start scripts.
- `core/`: small runtime helpers used by training, especially EMA and optional email notifier.
- `kronos_vendor/`: Kronos teacher implementation needed when rebuilding teacher cache.
- `ppo_trader/`: wheel/package source for Run19 inference, session management, and Freqtrade adapter.
- `freqtrade/`: Run19 Freqtrade session strategy and sanitized config templates.
- `evaluation/`: native Run19 / Freqtrade comparison and stop-loss scan scripts.
- `docs/`: Run19 architecture and training notes.

Not included:

- Market data parquet files.
- Teacher cache `.npy` files.
- Checkpoint binaries such as `best.pt`.
- Logs, CSV results, SQLite databases, wheel build artifacts, pycache.

Important:

The original Run19 training config referenced a Run14 init checkpoint that is not present in the current workspace. The effective final model present locally was:

```text
checkpoints/ppo_e2e_run19_sl15_valrandom/best.pt
```

Use Git LFS or an external artifact store if you want to version that binary model.

The export intentionally keeps only `PPORun19SessionStrategy` Freqtrade configs; older `PPOStrategyV2` configs were excluded.
