# Run19 配置改动总结

## 相比 Run18 的主要变化

### 1. 验证策略改进（解决过拟合）
- **启用随机验证**: `val_random_enabled: true`
- **验证环境数**: 519 (Tier1全覆盖173个 * 3轨迹)
- **轨迹长度**: 150 (训练和验证都用150，加快速度)
- **禁用固定验证**: `val_fixed_envs: 0`

### 2. 强化持仓行为
- **max_holding_steps**: 30 → 40 (允许更长持仓)
- **max_empty_steps**: 30 → 25 (减少空仓时间)
- **min_holding_steps**: 3 → 5 (最少持仓5步)
- **reward_hold_norm_denom**: 2.0 → 1.5 (增强持仓奖励)
- **potential_scale**: 0.02 → 0.025 (增加潜在收益权重)
- **potential_drawdown_weight**: 0.6 → 0.5 (降低回撤惩罚)

### 3. 调整空仓惩罚
- **empty_first_step_bonus**: 0.0002 → 0.0001 (降低空仓初始奖励)
- **empty_step_penalty**: 0.0 → 0.0001 (增加空仓惩罚)
- **empty_late_penalty**: 0.0001 → 0.0002 (增加后期空仓惩罚)
- **empty_reward_steps**: 5 → 3 (缩短空仓奖励期)
- **empty_penalty_start**: 25 → 20 (提前开始空仓惩罚)

### 4. 训练效率提升
- **trajectory_length**: 200 → 150 (加快训练速度)
- **val_trajectory_length**: 300 → 150 (加快验证速度)

### 5. 初始化
- 从 run14 的 best checkpoint 开始训练
- 输出目录: `checkpoints/ppo_e2e_run19_sl15_valrandom`

## 预期效果
1. **解决 Tier1 过拟合**: 随机验证位置避免记住固定模式
2. **增加持仓时间**: 鼓励模型持有更长时间，捕捉更大趋势
3. **减少过度交易**: 通过空仓惩罚降低交易频率
4. **更快训练**: 轨迹长度从200/300降到150
5. **更好泛化**: ValRandom 确保模型在不同市场位置都能表现良好
