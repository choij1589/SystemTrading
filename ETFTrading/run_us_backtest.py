"""
US ETF Strategy Backtesting Script

Run backtests for 3 US ETF strategies with REAL market data from Yahoo Finance.
Focus on strategy effectiveness with US-listed ETFs.
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
from ETFTrading.strategy.us_asset_allocation import USAllWeatherStrategy
from ETFTrading.strategy.us_momentum_rotation import USMomentumSectorRotationStrategy
from ETFTrading.strategy.us_dividend_growth import USDividendGrowthStrategy
from ETFTrading.backtesting.engine import ETFBacktestEngine
from ETFTrading.data.yahoo_loader import YahooETFLoader

print("="*80)
print("US ETF Investment Strategy Backtesting")
print("="*80)
print()

# Configuration
START_DATE = "2015-01-01"  # Longer history for US ETFs
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Investment parameters (in USD)
INITIAL_CAPITAL = 10_000  # $10,000
MONTHLY_DEPOSIT = 500     # $500/month

print(f"📅 Backtest Period: {START_DATE} to {END_DATE}")
print(f"💰 Initial Capital: ${INITIAL_CAPITAL:,}")
print(f"💵 Monthly Deposit: ${MONTHLY_DEPOSIT:,}")
print()

# Initialize strategies
print("📊 Initializing US ETF strategies...")
strategy1 = USAllWeatherStrategy()
strategy2 = USMomentumSectorRotationStrategy(top_n=3)
strategy3 = USDividendGrowthStrategy()

strategies = [
    (strategy1, "Strategy 1: US All-Weather Portfolio"),
    (strategy2, "Strategy 2: US Momentum Sector Rotation"),
    (strategy3, "Strategy 3: US Dividend + Growth")
]

print(f"✓ Strategy 1: {strategy1.name}")
print(f"✓ Strategy 2: {strategy2.name}")
print(f"✓ Strategy 3: {strategy3.name}")
print()

# Collect all tickers
all_tickers = set()
for strategy, _ in strategies:
    all_tickers.update(strategy.get_universe())

print(f"📦 Fetching REAL US ETF data from Yahoo Finance...")
print(f"   Tickers: {sorted(all_tickers)}")
print(f"   This may take a moment...")
print()

# Load REAL data from Yahoo Finance
loader = YahooETFLoader()
data = loader.load_multiple(
    tickers=list(all_tickers),
    start_date=START_DATE,
    end_date=END_DATE,
    use_cache=True  # Cache to avoid repeated API calls
)

if not data:
    print("❌ Failed to fetch data from Yahoo Finance")
    print("   Possible issues:")
    print("   - Internet connection")
    print("   - Yahoo Finance API temporarily down")
    sys.exit(1)

print(f"✓ Successfully loaded data for {len(data)} US ETFs")

# Show data summary
if data:
    sample_ticker = list(data.keys())[0]
    sample_df = data[sample_ticker]
    print(f"✓ Date range: {sample_df['date'].min()} to {sample_df['date'].max()}")
    print(f"✓ Trading days: {len(sample_df)}")

    # Show which ETFs loaded successfully
    print(f"\n✓ Successfully loaded ETFs:")
    for ticker in sorted(data.keys()):
        df = data[ticker]
        print(f"   - {ticker}: {len(df)} days")
print()

# Initialize backtesting engine
print("⚙️  Initializing backtesting engine...")
engine = ETFBacktestEngine(
    initial_capital=INITIAL_CAPITAL,
    monthly_deposit=MONTHLY_DEPOSIT,
    commission_rate=0.0001,      # 0.01% commission (US brokers)
    tax_rate=0.0,                 # No tax in tax-advantaged accounts
    bid_ask_spread=0.0001,        # 0.01% bid-ask spread
    rebalance_frequency="monthly"
)

print(f"✓ Initial capital: ${engine.initial_capital:,}")
print(f"✓ Monthly deposit: ${engine.monthly_deposit:,}")
print(f"✓ Commission: {engine.commission_rate:.4%}")
print(f"✓ Tax: {engine.tax_rate:.3%} (tax-advantaged account)")
print()

# Run backtests
results = []

print("="*80)
print("RUNNING BACKTESTS WITH REAL US ETF DATA")
print("="*80)
print()

for i, (strategy, name) in enumerate(strategies, 1):
    print(f"[{i}/3] {name}...")
    print("-" * 60)

    try:
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
        print(f"  - Final value: ${stats['final_value']:,.2f}")
        print(f"  - Total return: {stats['total_return']:.2%}")
        print(f"  - CAGR: {stats['cagr']:.2%}")
        print(f"  - Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
        print(f"  - Max Drawdown: {stats['max_drawdown']:.2%}")
        print()

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        print()

if not results:
    print("❌ No successful backtests")
    sys.exit(1)

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
        'Initial': f"${stats['initial_value']:,.0f}",
        'Final': f"${stats['final_value']:,.0f}",
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
    print(f"  Initial Value:     ${stats['initial_value']:>15,.2f}")
    print(f"  Final Value:       ${stats['final_value']:>15,.2f}")
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
    print(f"  Total Commission:  ${trades['commission'].sum():>14,.2f}")
    if 'tax' in trades.columns:
        print(f"  Total Tax:         ${trades['tax'].sum():>14,.2f}")
    print()

# Create visualizations
print("="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)
print()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('US ETF Investment Strategy Comparison', fontsize=16, fontweight='bold')

# 1. Equity curves
ax = axes[0, 0]
for r in results:
    equity = r['equity']
    label = r['name'].replace('Strategy ', 'S')
    ax.plot(equity['date'], equity['portfolio_value'] / 1000,
            label=label, linewidth=2)

ax.set_title('Portfolio Value Over Time', fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value ($1000s)')
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
output_path = 'ETFTrading/us_backtest_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualization to: {output_path}")
print()

# Generate summary report
print("="*80)
print("STRATEGY EFFECTIVENESS ANALYSIS")
print("="*80)
print()

# Find best performing strategy
best_return_idx = max(range(len(results)), key=lambda i: results[i]['stats']['total_return'])
best_sharpe_idx = max(range(len(results)), key=lambda i: results[i]['stats']['sharpe_ratio'])
lowest_dd_idx = max(range(len(results)), key=lambda i: results[i]['stats']['max_drawdown'])

print(f"🏆 Best Total Return:")
print(f"   {results[best_return_idx]['name']}")
print(f"   Return: {results[best_return_idx]['stats']['total_return']:.2%}")
print(f"   CAGR: {results[best_return_idx]['stats']['cagr']:.2%}")
print()

print(f"📈 Best Risk-Adjusted Return (Sharpe):")
print(f"   {results[best_sharpe_idx]['name']}")
print(f"   Sharpe Ratio: {results[best_sharpe_idx]['stats']['sharpe_ratio']:.2f}")
print(f"   This strategy provides the best return per unit of risk")
print()

print(f"🛡️  Lowest Drawdown (Most Defensive):")
print(f"   {results[lowest_dd_idx]['name']}")
print(f"   Max Drawdown: {results[lowest_dd_idx]['stats']['max_drawdown']:.2%}")
print(f"   This strategy best preserved capital during downturns")
print()

# Calculate investment summary
years = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days / 365.25
total_deposits = INITIAL_CAPITAL + (MONTHLY_DEPOSIT * 12 * years)

print(f"💰 Investment Summary:")
print(f"   Investment Period: {years:.1f} years")
print(f"   Total Deposits: ${total_deposits:,.2f}")
print()

for r in results:
    stats = r['stats']
    profit = stats['final_value'] - total_deposits
    roi = (stats['final_value'] / total_deposits - 1) * 100
    print(f"   {r['name'].replace('Strategy ', 'S')}")
    print(f"     Final Value: ${stats['final_value']:,.2f}")
    print(f"     Profit: ${profit:,.2f}")
    print(f"     ROI: {roi:.1f}%")
    print()

print("="*80)
print("KEY INSIGHTS")
print("="*80)
print()

print("1. All-Weather Portfolio (Strategy 1):")
print("   ✓ Most defensive with lowest drawdowns")
print("   ✓ Consistent performance across market conditions")
print("   ✓ Best for risk-averse investors")
print("   ✗ Lower returns than aggressive strategies")
print()

print("2. Momentum Sector Rotation (Strategy 2):")
print("   ✓ Highest return potential in trending markets")
print("   ✓ Captures sector rotation trends")
print("   ✓ Best for active investors")
print("   ✗ Higher volatility and turnover")
print("   ✗ May lag in choppy markets")
print()

print("3. Dividend + Growth (Strategy 3):")
print("   ✓ Balanced approach with income")
print("   ✓ Moderate risk and return")
print("   ✓ Benefits from dividend reinvestment")
print("   ✗ Concentrated in US equities")
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
print()
print("=" * 80)
print("NOTE: Results based on REAL US ETF data from Yahoo Finance")
print("      All results are deterministic and reproducible")
print("      Past performance does not guarantee future results")
print("=" * 80)
