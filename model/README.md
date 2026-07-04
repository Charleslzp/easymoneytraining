# Run19 Model Artifacts

Binary model files are intentionally not included in this git export.

Expected runtime layout if you want to run inference/backtests:

```text
checkpoints/ppo_e2e_run19_sl15_valrandom/best.pt
checkpoints/ppo_e2e_run19_sl15_valrandom/config.json
```

`run19_config.json` is a sanitized copy of the saved model config.
Use Git LFS or an external artifact store for `best.pt` if the model must be versioned.
