"""
Break-even Win Rate Analysis for buy97_sell99

Calculates the exact break-even win rate and what improvements are needed.
"""

import pandas as pd
import numpy as np

# Load test trades
df = pd.read_csv('buy97_sell99_trades_test.csv')

total = len(df)
wins = df[df['pnl'] > 0]
losses = df[df['pnl'] < 0]

# Stats
avg_win = wins['pnl'].mean()
avg_loss = abs(losses['pnl'].mean())
win_rate = len(wins) / total
loss_rate = len(losses) / total

# Break-even math
# EV = (WR * avg_win) - ((1-WR) * avg_loss)
# At breakeven: WR * avg_win = (1-WR) * avg_loss
# WR * avg_win = avg_loss - WR * avg_loss
# WR(avg_win + avg_loss) = avg_loss
# WR = avg_loss / (avg_win + avg_loss)

breakeven_wr = avg_loss / (avg_win + avg_loss)

print("=" * 70)
print("buy97_sell99 BREAK-EVEN ANALYSIS")
print("=" * 70)
print(f"\nTotal Trades:      {total}")
print(f"Wins:              {len(wins)} ({win_rate*100:.1f}%)")
print(f"Losses:            {len(losses)} ({loss_rate*100:.1f}%)")
print(f"\nAvg Win:           ${avg_win:.2f}")
print(f"Avg Loss (abs):    ${avg_loss:.2f}")
print(f"Payoff Ratio:      {avg_win / avg_loss:.2f}:1")
print(f"\nBREAK-EVEN WR:     {breakeven_wr*100:.2f}%")
print(f"Actual WR:         {win_rate*100:.2f}%")
print(f"WR Surplus:        {(win_rate - breakeven_wr)*100:.2f} percentage points")

# Current EV
if 'pnl' in df.columns:
    total_pnl = df['pnl'].sum()
    net_wr = win_rate - loss_rate
    print(f"\nNet PnL:           ${total_pnl:.2f}")
    print(f"Net Win Rate:      {net_wr*100:.2f}%")
    
    # What happens if WR drops?
    print(f"\n--- What if WR drops? ---")
    for wr in [0.85, 0.80, 0.75, 0.70, breakeven_wr]:
        if wr >= 0:
            ev = wr * avg_win - (1 - wr) * avg_loss
            print(f"  WR = {wr*100:.0f}%:  EV = ${ev:.2f}/trade  ({'PROFIT' if ev > 0 else 'BREAKEVEN' if abs(ev) < 0.01 else 'LOSS'})")
    
    # What payoff is needed at current WR?
    print(f"\n--- What if losses were smaller? ---")
    for loss_mult in [1.0, 0.8, 0.6, 0.4, 0.2]:
        smaller_loss = avg_loss * loss_mult
        ev = win_rate * avg_win - loss_rate * smaller_loss
        print(f"  Loss = ${smaller_loss:.2f}:  EV = ${ev:.2f}/trade")
    
    # What avg profit is needed at current WR?
    print(f"\n--- What if wins were bigger? ---")
    for win_mult in [1.0, 1.2, 1.5, 2.0]:
        bigger_win = avg_win * win_mult
        ev = win_rate * bigger_win - loss_rate * avg_loss
        print(f"  Win = ${bigger_win:.2f}:  EV = ${ev:.2f}/trade")

print(f"\n{'='*70}")
print("SOLUTIONS TO IMPROVE buy97_sell99")
print(f"{'='*70}")
print("1. INCREASE WIN RATE:")
print(f"   - Current:     {win_rate*100:.1f}%")
print(f"   - Breakeven:   {breakeven_wr*100:.2f}%")
print(f"   - Target:      >95% (safer margin)")
print("   - How:         Enter EVEN LATER (5s not 30s before settlement)")
print("   -             Filter out marginal setups (require 2%+ price move)")
print()
print("2. REDUCE AVS LOSS:")
print(f"   - Current avg loss: ${avg_loss:.2f}")
print("   - How:         SL at 50c (but tested: cuts winners too much)")
print("   -             Better: Hedge with OTM options on wrong side")
print("   -             Position sizing: reduce $ per trade on choppy days")
print()
print("3. INCREASE AVG WIN:")
print(f"   - Current: ${avg_win:.2f}")
print("   - How:         Partial TP: sell half at 99c, hold half to settlement")
print("   -             But tested: 99c rarely hits, no benefit")
print("   -             Better: Enter slightly earlier (15s vs 30s) for more upside")
print()
print("4. COMBINE STRATEGIES:")
print("   - Use buy97 as core (high freq, small edge)")
print("   - Add prev10_momentum on select windows (lower freq, confirmation)")
print("   - Correlation: buy97 late-entry vs prev10 early-entry = diversification")
print(f"\n{'='*70}")

# Show exact calculations
print("\nExact Math:")
print(f"  EV/trade = (WR × AvgWin) - (LR × AvgLoss)")
print(f"           = ({win_rate:.3f} × ${avg_win:.2f}) - ({loss_rate:.3f} × ${avg_loss:.2f})")
print(f"           = ${win_rate * avg_win:.2f} - ${loss_rate * avg_loss:.2f}")
print(f"           = ${win_rate * avg_win - loss_rate * avg_loss:.2f}")
