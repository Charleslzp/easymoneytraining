# Run19 Training

Main entry:

```bash
python training/train_e2e_ppo.py --config training/configs/e2e_ppo_run19_sl15_valrandom.json
```

Fine-tune from the trained Run19 best checkpoint:

```bash
bash training/scripts/start_run19_finetune.sh
```

The finetune launcher reads local paths/devices from `.env`, renders a resolved config, then starts training from `RUN19_INIT_CHECKPOINT` as `student_init` rather than `resume`. This creates a new run in `RUN19_OUTPUT_DIR`. The default finetune uses `RUN19_SEQ_LENGTH=300` and `RUN19_USE_ASSET_IDS=0`, matching the trained Run19 setup without feeding symbol identity into the model.

Cache build only:

```bash
python training/train_e2e_ppo.py --config training/configs/e2e_ppo_run19_sl15_valrandom.json --cache-only --cache-device cuda:0
```

Notes:

- Market data is not included. Set `data.data_file` in the config to your local parquet file.
- Teacher caches are not included. Set `teacher.repr_cache_dir` or rebuild from Kronos.
- Email settings were disabled and secrets removed for git safety.

Default local `.env` points `RUN19_REPR_CACHE_DIR` at the seq300 cache from `bigmodel`: `/home/pythonuser/pythonproject/bigmodel/data/cache/kron_repr_market_data_10M_run10`.
Default local `.env` sets `RUN19_CACHE_SPLITS=train,val,test` to match that cache meta.
For the default seq300 finetune, `.env` sets `RUN19_TEACHER_ENABLED=0` because the available seq300 repr cache does not match the selected 4h crypto data split.
The default local `.env` uses `RUN19_MINI_BATCH_SIZE=256` and disables parallel value/policy update so seq300 fits on the local 24GB GPU.
