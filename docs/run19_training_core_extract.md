# Run19 Training Core Extract

## Short Answer

Run19 is staged, but the first stage is better described as building an offline teacher cache, not building ordinary supervised labels.

In `configs/e2e_ppo_run19_sl15_valrandom.json`, Run19 uses:

- `phase.mode = "stage3_joint"`
- `teacher.enabled = true`
- `teacher.distill_mode = "representation"`
- `teacher.repr_cache_dir = "data/cache/kron_repr"`
- `teacher.repr_dim = 512`
- `teacher.repr_coef_start = 0.05`
- `teacher.repr_coef_end = 0.0`
- `teacher.repr_decay_updates = 1000`
- `teacher.kl_coef_start = 0.0`
- `teacher.strength_loss_weight = 0.0`
- `teacher.teacher_logit_weight = 0.0`

So the effective teacher signal for Run19 is a cached 512-dimensional Kron representation, trained through cosine distillation. It is not a buy/sell classification label. The student still learns the trading policy through PPO in the trading environment.

## Run19 Pipeline

1. Load 4h crypto market data.
2. Build or load Kron teacher representation cache for train/val windows.
3. Initialize the student from Run14 checkpoint.
4. Train policy and value networks jointly with PPO.
5. Add teacher representation distillation loss early in training.
6. Validate with random validation episodes.
7. Save `latest.pt` and `best.pt`.

## Entry Commands

Normal Run19 training:

```bash
python train_e2e_ppo.py --config configs/e2e_ppo_run19_sl15_valrandom.json
```

Resume Run19:

```bash
python train_e2e_ppo.py \
  --config configs/e2e_ppo_run19_sl15_valrandom.json \
  --resume checkpoints/ppo_e2e_run19_sl15_valrandom/latest.pt
```

Teacher cache only, split across two GPUs:

```bash
python train_e2e_ppo.py \
  --config configs/e2e_ppo_run19_sl15_valrandom.json \
  --cache-only --cache-device cuda:0 --num-shards 2 --shard-index 0

python train_e2e_ppo.py \
  --config configs/e2e_ppo_run19_sl15_valrandom.json \
  --cache-only --cache-device cuda:1 --num-shards 2 --shard-index 1

python train_e2e_ppo.py \
  --config configs/e2e_ppo_run19_sl15_valrandom.json \
  --cache-meta-only
```

There is also an orchestration script:

```bash
bash run_parallel_cache_then_train.sh configs/e2e_ppo_run19_sl15_valrandom.json
```

## Core Training Skeleton

This is the distilled shape of `train_e2e_ppo.py` for Run19:

```python
def run19_train(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg.run.seed)

    policy_device = choose_device(cfg.run.policy_device)
    value_device = choose_device(cfg.run.value_device)

    symbol_store = build_symbol_store(cfg)
    n_assets = len(symbol_store)

    policy = StudentPolicy(cfg, n_assets=n_assets).to(policy_device)
    value = StudentValue(cfg, n_assets=n_assets).to(value_device)

    teacher = None
    if cfg.teacher.enabled and cfg.teacher.use_offline_cache:
        teacher = load_kron_repr_cache(cfg, symbol_store, device=policy_device)
        if teacher is None:
            online_teacher = load_kron_teacher(cfg, policy_device)
            build_kron_repr_cache(cfg, symbol_store, online_teacher)
            teacher = load_kron_repr_cache(cfg, symbol_store, device=policy_device)

    load_student_init(
        policy,
        value,
        cfg.student_init.checkpoint,
        cfg.student_init.strict,
    )
    configure_policy_phase(policy, cfg.phase.mode)

    trainer = PPOTrainer(
        policy=policy,
        value=value,
        cfg=cfg,
        policy_device=policy_device,
        value_device=value_device,
        teacher_enabled=teacher is not None,
    )
    collector = Collector(cfg, symbol_store, policy_device, teacher=teacher)

    for update in range(start_update, cfg.ppo.max_updates):
        rollout, train_trades = collector.collect(policy, mode="train")
        stats = trainer.update(rollout)

        if update > 0 and update % cfg.ppo.val_interval == 0:
            val_trades = collector.collect(policy, mode="val", fixed_seed=None)
            val_metrics = compute_trade_metrics(val_trades)
            save_latest_or_best_checkpoint(stats, val_metrics)
```

## Stage 1: Teacher Representation Cache

The cache stores teacher outputs indexed by symbol and bar index.

For Run19, because `distill_mode = "representation"`, the cache path is `data/cache/kron_repr`, and each symbol gets an array shaped:

```python
(num_bars, 512)
```

The core cache loop is:

```python
idx = _cache_indices_for_symbol(
    data,
    seq_len=cfg.data.seq_length,
    split_names=["train", "val"],
    overlap_lookback=cfg.data.overlap_lookback,
)

for chunk_idx in chunks(idx):
    x = stack(data.raw_ohlcva[i - seq_len:i] for i in chunk_idx)
    x_stamp = stack(data.time_features[i - seq_len:i] for i in chunk_idx)
    y_stamp = stack(data.time_features[i:i + 1] for i in chunk_idx)

    reprs = frozen_kron_teacher.predict_repr(x, x_stamp, y_stamp)
    cache_array[chunk_idx] = reprs
```

`FrozenKronTeacher.predict_repr()` tokenizes the OHLCVA window, runs Kron decode stage 1, then uses the last context vector:

```python
pre_ids, post_ids = tokenizer.encode(x_in, half=True)
_, context = model.decode_s1(pre_ids, post_ids, stamp=x_stamp_in)
teacher_repr = context[:, -1, :]
```

## Stage 2: Student Init

Run19 is not trained from scratch. It initializes from:

```text
checkpoints/ppo_e2e_kron_mamba_daily_20260309_run14_noasset_sl15_email/best.pt
```

The loader accepts checkpoint keys like `policy`, `policy_state_dict`, or `model_state_dict`, and loads value weights too when present.

## Stage 3: Joint PPO Training

Run19 config uses:

```python
configure_policy_phase(policy, "stage3_joint")
```

That means all policy parameters are trainable.

The model has:

- Mamba backbone over normalized market sequence.
- Position-state projection.
- Optional asset embedding, disabled in Run19 with `use_asset_ids = false`.
- Flat-state heads: `WAIT/OPEN` gate plus signed entry size.
- Holding-state head: `CLOSE/HOLD`.
- Teacher representation projection head.
- Separate value network.

## Collector Core

The collector samples many trading environments in parallel, queries teacher cache by `(symbol, current_idx)`, then stores the teacher representation inside the rollout:

```python
teacher_repr = teacher.predict_repr_from_keys(teacher_symbols, teacher_indices)

actions, log_probs = policy(
    market,
    pos_info,
    asset_ids,
    position_states,
    action_masks,
)

rewards, dones, trade_ends, trades = batch_env.step_batch(actions, log_probs)

buffer.append_step(
    market=market,
    pos_info=pos_info,
    teacher_repr=teacher_repr,
    actions=actions,
    log_probs=log_probs,
    rewards=rewards,
    dones=dones,
    trade_ends=trade_ends,
)
```

## PPO Loss Core

The PPO update has three important parts.

Policy loss:

```python
ratio = exp(new_log_prob - old_log_prob)
surr1 = ratio * advantage
surr2 = clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
policy_loss = -mean(min(surr1, surr2))
```

Value loss:

```python
value_loss = smooth_l1_loss(value_network(obs), returns) * value_loss_coef
```

Run19 teacher representation distillation:

```python
student_repr = normalize(policy_outputs["teacher_repr"])
target_repr = normalize(cached_teacher_repr)
repr_loss = 1.0 - mean(sum(student_repr * target_repr))

loss = (
    policy_loss
    - entropy_coef * entropy
    + repr_coef * repr_loss
)
```

The representation coefficient linearly decays:

```python
repr_coef: 0.05 -> 0.0 over 1000 PPO updates
```

After this decay, the student mainly relies on PPO reward instead of teacher representation.

## Trading Environment Core

Run19 is not a one-bar prediction model. It trains inside a stateful trading environment.

Empty state:

- Wait until `min_empty_steps`.
- Decide `WAIT` or `OPEN`.
- If opening, output signed position size.
- Entry price is next bar open.

Holding state:

- Must hold for at least `min_holding_steps`.
- Must close after `max_holding_steps`.
- Can close by model action.
- Can close by ATR stop loss or take profit.

Reward is composed from:

- Open cost.
- Holding potential shaping.
- Close trade return.
- Empty wait reward or late empty penalty.

Run19 config highlights:

```text
seq_length = 180
trajectory_length = 150
envs_per_batch = 60
val_random_envs = 519
max_updates = 2000
stop_loss_vol_multiple = 1.5
max_holding_steps = 40
min_holding_steps = 3
entry_threshold = 0.2
position_scale = 1.3
```

## Source Files

- `train_e2e_ppo.py`: full training implementation.
- `configs/e2e_ppo_run19_sl15_valrandom.json`: Run19 training config.
- `start_run19.sh`: Run19 start command.
- `start_run19_resume.sh`: Run19 resume command.
- `run_parallel_cache_then_train.sh`: parallel cache build then train orchestration.
- `checkpoints/ppo_e2e_run19_sl15_valrandom/best.pt`: final effective model checkpoint.
- `checkpoints/ppo_e2e_run19_sl15_valrandom/config.json`: saved model config.

