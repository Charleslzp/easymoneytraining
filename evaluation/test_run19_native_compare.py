import sys
import torch
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from datetime import datetime

from train_e2e_ppo import (
    Config, load_config, build_symbol_store,
    StudentPolicy, BatchTradingEnv, set_seed,
    TradingEnv
)

def _live_norm_get_market_features(self):
    """滚动窗口归一化，模拟实盘特征计算"""
    s = max(0, self.current_idx - self.seq_len)
    e = self.current_idx
    # 用 raw_ohlcva 的 ohlcv 部分（前5列）重新归一化
    raw = self.raw_ohlcva[s:e, :5].copy()  # (T, 5)
    out = np.zeros_like(raw, dtype=np.float32)
    # O/H/L/C → returns
    for i in range(4):
        p = raw[:, i]
        ret = np.zeros(len(p), dtype=np.float32)
        ret[1:] = (p[1:] - p[:-1]) / (p[:-1] + 1e-8)
        out[:, i] = ret
    # Volume → log z-score（仅用当前窗口）
    vol = raw[:, 4]
    lv = np.log(vol + 1.0)
    out[:, 4] = (lv - lv.mean()) / (lv.std() + 1e-8)
    return out

# Monkey-patch TradingEnv
TradingEnv.get_market_features = _live_norm_get_market_features

def main():
    # Load freqtrade config to get the 100 symbols
    print("Loading freqtrade config to get symbol list...")
    with open('config_backtest_multi100.json', 'r') as f:
        freqtrade_config = json.load(f)

    freqtrade_symbols = freqtrade_config['exchange']['pair_whitelist']
    print(f"Loaded {len(freqtrade_symbols)} symbols from freqtrade config")

    # Convert freqtrade format (BTC/USDT:USDT) to run19 format (BTCUSDT)
    target_symbols = set()
    for sym in freqtrade_symbols:
        base = sym.split('/')[0]
        target_symbols.add(base + 'USDT')

    print(f"Converted to {len(target_symbols)} base symbols")

    # Load run19 config with seq_length=180
    cfg = load_config('configs/e2e_ppo_run19_sl15_valrandom_seq180.json')

    print("\n⚠️ Using LIVE NORM + seq_length=180")
    print("=" * 80)
    print(f"seq_length: {cfg.data.seq_length}")
    print(f"overlap_lookback: {cfg.data.overlap_lookback}")
    print(f"stop_loss: {cfg.env.stop_loss}")
    print(f"stop_loss_vol_multiple: {cfg.env.stop_loss_vol_multiple}")
    print("=" * 80)

    set_seed(cfg.run.seed)

    print("\nLoading data...")
    symbol_store_full = build_symbol_store(cfg)

    # Filter to only target symbols
    symbol_store = {}
    for sym, data in symbol_store_full.items():
        if sym in target_symbols:
            symbol_store[sym] = data

    symbol_list = list(symbol_store.keys())
    print(f"Filtered to {len(symbol_list)} symbols that exist in run19 data")

    # Extract tier info
    tier_map = {}
    for sym, data in symbol_store.items():
        tier_str = data.tier
        tier_num = int(tier_str.replace('tier', '')) if tier_str.startswith('tier') else 3
        tier_map[sym] = tier_num

    # Load model
    print("Loading best checkpoint...")
    device = torch.device('cuda:0')
    n_assets = len(symbol_store_full)  # Model was trained on all symbols
    policy = StudentPolicy(cfg, n_assets=n_assets).to(device)

    ckpt = torch.load('checkpoints/ppo_e2e_run19_sl15_valrandom/best.pt', map_location=device)
    policy.load_state_dict(ckpt['policy'])
    policy.eval()

    # Run multiple rounds to cover all symbols
    print(f"\nRunning test on {len(symbol_list)} symbols (multiple rounds)...")
    all_trades = []
    symbols_tested = set()
    round_num = 0

    max_rounds = 20

    for round_num in range(max_rounds):
        print(f"\n=== Round {round_num + 1}/{max_rounds} ===")

        # Create test env
        test_env = BatchTradingEnv(
            symbol_store=symbol_store,
            cfg=cfg,
            mode='test',
            num_envs=min(60, len(symbol_list)),
            max_steps=300,
            fixed_episodes=None
        )

        # Track which symbols are in this round
        round_symbols = set()
        for env in test_env.envs:
            round_symbols.add(env.symbol)

        print(f"Testing {len(round_symbols)} symbols in this round")

        # Run test
        for step in range(300):
            market, teacher_symbols, teacher_indices, teacher_x, teacher_x_stamp, teacher_y_stamp, pos_info, action_masks, asset_ids = (
                test_env.get_batch_obs()
            )
            pos_states = test_env.get_batch_states()

            t_market = torch.from_numpy(market).to(device)
            t_pos = torch.from_numpy(pos_info).to(device)
            t_masks = torch.from_numpy(action_masks).to(device)
            t_assets = torch.from_numpy(asset_ids).to(device)
            t_states = torch.from_numpy(pos_states).to(device)

            with torch.no_grad():
                actions = policy.act_deterministic(
                    t_market, t_pos, t_assets, t_states, t_masks
                )

            actions_np = actions.cpu().numpy()

            # 调试：打印actions的形状和转换
            if step == 0:
                print(f"DEBUG: actions shape = {actions.shape}")
                print(f"DEBUG: actions_np shape = {actions_np.shape}")
                print(f"DEBUG: actions_np[0] = {actions_np[0]}")
                print(f"DEBUG: type(actions_np[0]) = {type(actions_np[0])}")
                print(f"DEBUG: int(actions_np[0]) will be called...")
                try:
                    test_int = int(actions_np[0])
                    print(f"DEBUG: int(actions_np[0]) = {test_int}")
                except Exception as e:
                    print(f"DEBUG: int(actions_np[0]) ERROR: {e}")

            zero_log_probs = np.zeros(actions_np.shape[0], dtype=np.float32)
            rewards, dones, trade_ends, ro, rh, rc, trades = test_env.step_batch(actions_np, zero_log_probs)
            all_trades.extend(trades)

            if dones.all():
                print(f"  All episodes done at step {step}")
                break

        symbols_tested.update(round_symbols)
        print(f"  Total unique symbols tested so far: {len(symbols_tested)}/{len(symbol_list)}")
        print(f"  Total trades so far: {len(all_trades)}")

        # Stop if we've tested all symbols
        if len(symbols_tested) >= len(symbol_list):
            print(f"\nAll {len(symbol_list)} symbols have been tested!")
            break

    print(f"\n=== Test Completed ===")
    print(f"Total rounds: {round_num + 1}")
    print(f"Total trades: {len(all_trades)}")
    print(f"Symbols tested: {len(symbols_tested)}/{len(symbol_list)}")

    # Aggregate by symbol
    symbol_results = defaultdict(lambda: {'pnl': 0.0, 'trades': 0, 'wins': 0, 'holding_steps': []})

    for trade in all_trades:
        sym = symbol_list[trade.asset_id]
        symbol_results[sym]['pnl'] += trade.pnl
        symbol_results[sym]['trades'] += 1
        symbol_results[sym]['holding_steps'].append(trade.holding_steps)
        if trade.pnl > 0:
            symbol_results[sym]['wins'] += 1

    # Convert to dataframe
    results = []
    for sym in symbol_list:
        tier = tier_map.get(sym, 3)
        pnl = symbol_results[sym]['pnl']
        trades = symbol_results[sym]['trades']
        wins = symbol_results[sym]['wins']
        holding_steps = symbol_results[sym]['holding_steps']
        avg_holding = np.mean(holding_steps) if holding_steps else 0.0
        results.append({
            'symbol': sym,
            'tier': tier,
            'pnl': pnl,
            'trades': trades,
            'wins': wins,
            'win_rate': wins/trades if trades > 0 else 0,
            'avg_holding_steps': avg_holding,
            'tested': sym in symbols_tested
        })

    df = pd.DataFrame(results)

    # Overall stats
    print("\n=== Overall Test Results (Run19 Native, 100 symbols, seq_length=180) ===")
    print(f"Total PNL: {df['pnl'].sum():.2f}")
    print(f"Total Trades: {df['trades'].sum()}")
    total_wins = df['wins'].sum()
    total_trades = df['trades'].sum()
    print(f"Win Rate: {total_wins/total_trades*100:.1f}%" if total_trades > 0 else "Win Rate: N/A")
    print(f"Avg PNL per symbol: {df['pnl'].mean():.4f}")
    print(f"Symbols with trades: {(df['trades'] > 0).sum()}/{len(df)}")

    # Tier stats
    print("\n=== Tier Breakdown ===")
    for tier in [1, 2, 3]:
        tier_df = df[df['tier'] == tier]
        if len(tier_df) > 0:
            total_pnl = tier_df['pnl'].sum()
            total_trades = tier_df['trades'].sum()
            total_wins = tier_df['wins'].sum()
            win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
            symbols_with_trades = (tier_df['trades'] > 0).sum()
            print(f"Tier {tier}: pnl={total_pnl:+.2f}, trades={total_trades}, win={win_rate:.1f}%, symbols={symbols_with_trades}/{len(tier_df)}")

    # Calculate weighted pnl
    tier1_pnl = df[df['tier'] == 1]['pnl'].sum()
    tier2_pnl = df[df['tier'] == 2]['pnl'].sum()
    tier3_pnl = df[df['tier'] == 3]['pnl'].sum()
    weighted_pnl = tier1_pnl * 0.6 + tier2_pnl * 0.25 + tier3_pnl * 0.15
    print(f"\nWeighted PNL: {weighted_pnl:+.2f}")

    # Calculate annualized return
    test_days = 447  # 2025-02-21 to 2026-05-14
    print(f"\n=== Annualized Returns (test period: {test_days} days) ===")
    for tier in [1, 2, 3]:
        tier_pnl = df[df['tier'] == tier]['pnl'].sum()
        annual = (tier_pnl / test_days) * 365
        print(f"Tier {tier}: {annual:.1f}%")
    annual_weighted = (weighted_pnl / test_days) * 365
    print(f"Weighted: {annual_weighted:.1f}%")

    # 对比
    print(f"\n=== 对比分析 ===")
    print(f"Freqtrade (100 symbols, seq_length=180): 339.22% (年化277.6%, 2552笔交易, 83.4%胜率)")
    print(f"Run19 Native (100 symbols, seq_length=180): {df['pnl'].sum():+.2f} (年化{(df['pnl'].sum()/test_days)*365:.1f}%, {df['trades'].sum()}笔交易, {total_wins/total_trades*100:.1f}%胜率)")

    # Save results
    df.to_csv('test_results_run19_native_100symbols.csv', index=False)
    print("\nResults saved to test_results_run19_native_100symbols.csv")

if __name__ == '__main__':
    main()
