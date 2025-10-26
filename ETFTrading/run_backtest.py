"""
ETF Strategy Backtesting Script

Run backtests for all 3 strategies and generate comparison report.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import our modules
from ETFTrading.strategy.asset_allocation import GlobalAssetAllocationStrategy
from ETFTrading.strategy.momentum_rotation import MomentumSectorRotationStrategy
from ETFTrading.strategy.dividend_growth import DividendGrowthMixStrategy
from ETFTrading.backtesting.engine import ETFBacktestEngine

print("="*80)
print("ETF Investment Strategy Backtesting")
print("="*80)
print()

# Generate realistic sample data
def generate_realistic_etf_data(tickers, start_date, end_date):
    """
    Generate realistic ETF price data based on historical characteristics.

    Different ETF types have different return/volatility profiles:
    - Equity ETFs: Higher return, higher volatility
    - Bond ETFs: Lower return, lower volatility
    - Gold: Moderate return, moderate volatility
    - Sector ETFs: Varied performance
    """
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # ETF characteristics (annual return, annual volatility)
    etf_profiles = {
        # Strategy 1: Global Asset Allocation
        "069500": (0.08, 0.20),   # KODEX 200: 8% return, 20% vol
        "360750": (0.12, 0.18),   # TIGER S&P500: 12% return, 18% vol
        "152380": (0.02, 0.05),   # KODEX 국고채: 2% return, 5% vol
        "132030": (0.05, 0.15),   # KODEX 골드: 5% return, 15% vol

        # Strategy 2: Momentum Sectors
        "091180": (0.15, 0.35),   # KODEX 반도체: High return, high vol
        "157450": (0.18, 0.40),   # TIGER 2차전지: Very high
        "227540": (0.10, 0.25),   # TIGER IT
        "139230": (0.05, 0.30),   # TIGER 건설
        "139260": (0.07, 0.28),   # TIGER 에너지
        "139250": (0.06, 0.22),   # TIGER 금융
        "228790": (0.09, 0.26),   # TIGER 헬스케어

        # Strategy 3: Dividend + Growth
        "458730": (0.08, 0.14),   # TIGER 배당: Lower vol
        "133690": (0.14, 0.22),   # TIGER 나스닥: Higher growth
    }

    result = {}

    for ticker in tickers:
        if ticker not in etf_profiles:
            # Default profile
            annual_return, annual_vol = 0.08, 0.20
        else:
            annual_return, annual_vol = etf_profiles[ticker]

        # Convert to daily
        daily_return = annual_return / 252
        daily_vol = annual_vol / np.sqrt(252)

        # Generate returns with some autocorrelation (momentum)
        np.random.seed(hash(ticker) % (2**32))

        returns = np.zeros(len(dates))
        returns[0] = daily_return

        for i in range(1, len(dates)):
            # Add momentum effect (0.3 autocorrelation)
            momentum = 0.3 * returns[i-1]
            noise = np.random.normal(0, daily_vol)
            returns[i] = daily_return + momentum + noise

        # Generate price series
        initial_price = 10000.0
        prices = initial_price * np.cumprod(1 + returns)

        # Add realistic OHLC variation
        df = pd.DataFrame({
            'date': dates,
            'close': prices,
        })

        df['open'] = df['close'] * (1 + np.random.uniform(-0.005, 0.005, len(dates)))
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, len(dates)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, len(dates)))
        df['volume'] = np.random.randint(100000, 1000000, len(dates))

        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

        result[ticker] = df

    return result


# Configuration
START_DATE = "2020-01-01"
END_DATE = "2024-10-26"

print(f"📅 Backtest Period: {START_DATE} to {END_DATE}")
print()

# Initialize strategies
print("📊 Initializing strategies...")
strategy1 = GlobalAssetAllocationStrategy()
strategy2 = MomentumSectorRotationStrategy(top_n=3)
strategy3 = DividendGrowthMixStrategy()

strategies = [
    (strategy1, "Strategy 1: Global Asset Allocation"),
    (strategy2, "Strategy 2: Momentum Sector Rotation"),
    (strategy3, "Strategy 3: Dividend + Growth Mix")
]

print(f"✓ Strategy 1: {strategy1.name}")
print(f"✓ Strategy 2: {strategy2.name}")
print(f"✓ Strategy 3: {strategy3.name}")
print()

# Collect all tickers
all_tickers = set()
for strategy, _ in strategies:
    all_tickers.update(strategy.get_universe())

print(f"📦 Loading data for {len(all_tickers)} ETFs...")
print(f"   Tickers: {sorted(all_tickers)}")
print()

# Generate data
data = generate_realistic_etf_data(
    tickers=list(all_tickers),
    start_date=START_DATE,
    end_date=END_DATE
)

print(f"✓ Generated {len(data)} ETF price histories")
print(f"✓ {len(data[list(data.keys())[0]])} trading days")
print()

# Initialize backtesting engine
print("⚙️  Initializing backtesting engine...")
engine = ETFBacktestEngine(
    initial_capital=4_200_000,
    monthly_deposit=300_000,
    commission_rate=0.00015,
    tax_rate=0.0023,
    rebalance_frequency="monthly"
)

print(f"✓ Initial capital: {engine.initial_capital:,} KRW")
print(f"✓ Monthly deposit: {engine.monthly_deposit:,} KRW")
print(f"✓ Commission: {engine.commission_rate:.4%}")
print(f"✓ Tax: {engine.tax_rate:.3%}")
print()

# Run backtests
results = []

print("="*80)
print("RUNNING BACKTESTS")
print("="*80)
print()

for i, (strategy, name) in enumerate(strategies, 1):
    print(f"[{i}/3] {name}...")
    print("-" * 60)

    equity = engine.run(strategy, data, START_DATE, END_DATE)
    stats = engine.get_summary_stats(equity)
    trades = engine.get_trade_log()

    results.append({
        'strategy': strategy,
        'name': name,
        'equity': equity,
        'stats': stats,
        'trades': trades
    })

    print(f"✓ Completed")
    print(f"  - Trading days: {len(equity)}")
    print(f"  - Total trades: {len(trades)}")
    print(f"  - Final value: {stats['final_value']:,.0f} KRW")
    print(f"  - Total return: {stats['total_return']:.2%}")
    print()

# Generate comparison report
print("="*80)
print("PERFORMANCE COMPARISON")
print("="*80)
print()

comparison_data = []
for r in results:
    stats = r['stats']
    comparison_data.append({
        'Strategy': r['name'].replace('Strategy ', 'S').replace(': ', ':\n'),
        'Initial (KRW)': f"{stats['initial_value']:,.0f}",
        'Final (KRW)': f"{stats['final_value']:,.0f}",
        'Return': f"{stats['total_return']:.2%}",
        'CAGR': f"{stats['cagr']:.2%}",
        'Volatility': f"{stats['volatility']:.2%}",
        'Sharpe': f"{stats['sharpe_ratio']:.2f}",
        'Max DD': f"{stats['max_drawdown']:.2%}",
        'Win Rate': f"{stats['win_rate']:.2%}",
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))
print()

# Detailed statistics
print("="*80)
print("DETAILED STATISTICS")
print("="*80)
print()

for r in results:
    print(f"\n{r['name']}")
    print("-" * 60)
    stats = r['stats']
    trades = r['trades']

    print(f"Portfolio Performance:")
    print(f"  Initial Value:     {stats['initial_value']:>15,.0f} KRW")
    print(f"  Final Value:       {stats['final_value']:>15,.0f} KRW")
    print(f"  Total Return:      {stats['total_return']:>15.2%}")
    print(f"  CAGR:              {stats['cagr']:>15.2%}")
    print()
    print(f"Risk Metrics:")
    print(f"  Volatility (Ann.): {stats['volatility']:>15.2%}")
    print(f"  Sharpe Ratio:      {stats['sharpe_ratio']:>15.2f}")
    print(f"  Maximum Drawdown:  {stats['max_drawdown']:>15.2%}")
    print(f"  Win Rate:          {stats['win_rate']:>15.2%}")
    print()
    print(f"Trading Activity:")
    print(f"  Total Trades:      {len(trades):>15,}")
    print(f"  Buy Orders:        {(trades['side'] == 'buy').sum():>15,}")
    print(f"  Sell Orders:       {(trades['side'] == 'sell').sum():>15,}")
    print(f"  Total Commission:  {trades['commission'].sum():>15,.0f} KRW")
    print(f"  Total Tax:         {trades['tax'].sum():>15,.0f} KRW")
    print()

# Create visualizations
print("="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)
print()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('ETF Investment Strategy Comparison', fontsize=16, fontweight='bold')

# 1. Equity curves
ax = axes[0, 0]
for r in results:
    equity = r['equity']
    label = r['name'].replace('Strategy ', 'S')
    ax.plot(equity['date'], equity['portfolio_value'] / 1_000_000,
            label=label, linewidth=2)

ax.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value (Million KRW)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 2. Drawdown
ax = axes[0, 1]
for r in results:
    equity = r['equity']
    cummax = equity['portfolio_value'].cummax()
    drawdown = (equity['portfolio_value'] - cummax) / cummax
    label = r['name'].replace('Strategy ', 'S')
    ax.plot(equity['date'], drawdown * 100, label=label, linewidth=2)

ax.set_title('Drawdown (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 3. Performance metrics comparison
ax = axes[1, 0]
metrics = ['CAGR\n(%)', 'Sharpe\nRatio', 'Max DD\n(%)']
x = np.arange(len(metrics))
width = 0.25

for i, r in enumerate(results):
    stats = r['stats']
    values = [
        stats['cagr'] * 100,
        stats['sharpe_ratio'],
        abs(stats['max_drawdown']) * 100
    ]
    offset = (i - 1) * width
    label = r['name'].replace('Strategy ', 'S').split(':')[0]
    ax.bar(x + offset, values, width, label=label)

ax.set_title('Performance Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Value')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# 4. Rolling 60-day returns
ax = axes[1, 1]
window = 60
for r in results:
    equity = r['equity']
    returns = equity['portfolio_value'].pct_change()
    rolling_ret = returns.rolling(window).sum() * 100
    label = r['name'].replace('Strategy ', 'S')
    ax.plot(equity['date'], rolling_ret, label=label, linewidth=2, alpha=0.8)

ax.set_title(f'{window}-Day Rolling Returns', fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Return (%)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

plt.tight_layout()

# Save figure
output_path = 'ETFTrading/backtest_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualization to: {output_path}")
print()

# Generate summary report
print("="*80)
print("SUMMARY & RECOMMENDATIONS")
print("="*80)
print()

# Find best performing strategy
best_return_idx = max(range(len(results)), key=lambda i: results[i]['stats']['total_return'])
best_sharpe_idx = max(range(len(results)), key=lambda i: results[i]['stats']['sharpe_ratio'])
lowest_dd_idx = max(range(len(results)), key=lambda i: results[i]['stats']['max_drawdown'])

print(f"🏆 Best Total Return:")
print(f"   {results[best_return_idx]['name']}")
print(f"   Return: {results[best_return_idx]['stats']['total_return']:.2%}")
print()

print(f"📈 Best Risk-Adjusted Return (Sharpe):")
print(f"   {results[best_sharpe_idx]['name']}")
print(f"   Sharpe Ratio: {results[best_sharpe_idx]['stats']['sharpe_ratio']:.2f}")
print()

print(f"🛡️  Lowest Drawdown:")
print(f"   {results[lowest_dd_idx]['name']}")
print(f"   Max Drawdown: {results[lowest_dd_idx]['stats']['max_drawdown']:.2%}")
print()

print("💡 Strategy Recommendations:")
print()
print("1. Conservative Investors (Risk-Averse):")
print("   → Strategy 1: Global Asset Allocation")
print("   → Stable, diversified, lower volatility")
print()
print("2. Aggressive Investors (High Risk Tolerance):")
print("   → Strategy 2: Momentum Sector Rotation")
print("   → Higher potential returns, higher volatility")
print()
print("3. Balanced Investors (Moderate Risk):")
print("   → Strategy 3: Dividend + Growth Mix")
print("   → Income + growth, moderate risk")
print()

print("="*80)
print("BACKTEST COMPLETED SUCCESSFULLY")
print("="*80)

# Save results to CSV
for r in results:
    strategy_name = r['name'].replace('Strategy ', 'S').replace(': ', '_').replace(' ', '_')

    # Save equity curve
    equity_file = f"ETFTrading/{strategy_name}_equity.csv"
    r['equity'].to_csv(equity_file, index=False)

    # Save trades
    trades_file = f"ETFTrading/{strategy_name}_trades.csv"
    r['trades'].to_csv(trades_file, index=False)

print(f"\n✓ Results saved to CSV files in ETFTrading/ directory")
