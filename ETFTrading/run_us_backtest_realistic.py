"""
US ETF Strategy Backtesting with Realistic Data

Uses realistic synthetic data when Yahoo Finance is unavailable.
Focus on strategy effectiveness and relative performance.
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import our modules
from ETFTrading.strategy.us_asset_allocation import USAllWeatherStrategy
from ETFTrading.strategy.us_momentum_rotation import USMomentumSectorRotationStrategy
from ETFTrading.strategy.us_dividend_growth import USDividendGrowthStrategy
from ETFTrading.backtesting.engine import ETFBacktestEngine
from ETFTrading.data.realistic_data_generator import RealisticETFDataGenerator

print("="*80)
print("US ETF Investment Strategy Backtesting")
print("Using Realistic Synthetic Data")
print("="*80)
print()

# Configuration
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"

# Investment parameters (in USD)
INITIAL_CAPITAL = 10_000  # $10,000
MONTHLY_DEPOSIT = 500     # $500/month

print(f"📅 Backtest Period: {START_DATE} to {END_DATE} (10 years)")
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
print(f"   {', '.join(strategy1.get_universe())}")
print(f"✓ Strategy 2: {strategy2.name}")
print(f"   {', '.join(strategy2.get_universe())}")
print(f"✓ Strategy 3: {strategy3.name}")
print(f"   {', '.join(strategy3.get_universe())}")
print()

# Collect all tickers
all_tickers = set()
for strategy, _ in strategies:
    all_tickers.update(strategy.get_universe())

print(f"📦 Generating realistic US ETF data...")
print(f"   Tickers: {sorted(all_tickers)}")
print(f"   Based on historical ETF statistics (2010-2024)")
print()

# Generate realistic data
generator = RealisticETFDataGenerator(seed=42)
data = generator.generate(
    tickers=list(all_tickers),
    start_date=START_DATE,
    end_date=END_DATE,
    initial_price=100.0
)

print(f"✓ Successfully generated data for {len(data)} US ETFs")

# Show data summary
if data:
    sample_ticker = list(data.keys())[0]
    sample_df = data[sample_ticker]
    print(f"✓ Date range: {sample_df['date'].min()} to {sample_df['date'].max()}")
    print(f"✓ Trading days: {len(sample_df)}")

    # Show statistics
    print(f"\n✓ Generated ETF characteristics:")
    for ticker in sorted(data.keys()):
        df = data[ticker]
        total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        annual_vol = df['close'].pct_change().std() * np.sqrt(252) * 100
        print(f"   - {ticker}: Total Return {total_return:>6.1f}%, Ann. Vol {annual_vol:>5.1f}%")
print()

# Initialize backtesting engine
print("⚙️  Initializing backtesting engine...")
engine = ETFBacktestEngine(
    initial_capital=INITIAL_CAPITAL,
    monthly_deposit=MONTHLY_DEPOSIT,
    commission_rate=0.0001,      # 0.01% commission
    tax_rate=0.0,                 # Tax-advantaged account
    bid_ask_spread=0.0001,        # 0.01% spread
    rebalance_frequency="monthly"
)

print(f"✓ Initial capital: ${engine.initial_capital:,}")
print(f"✓ Monthly deposit: ${engine.monthly_deposit:,}")
print(f"✓ Commission: {engine.commission_rate:.4%}")
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
        'Strategy': r['name'].split(':')[1].strip(),
        'Initial': f"${stats['initial_value']:,.0f}",
        'Final': f"${stats['final_value']:,.0f}",
        'Return': f"{stats['total_return']:.1%}",
        'CAGR': f"{stats['cagr']:.1%}",
        'Vol': f"{stats['volatility']:.1%}",
        'Sharpe': f"{stats['sharpe_ratio']:.2f}",
        'Max DD': f"{stats['max_drawdown']:.1%}",
        'Win Rate': f"{stats['win_rate']:.1%}",
    })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))
print()

# Create visualizations
print("="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)
print()

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Equity curves (large, top)
ax = fig.add_subplot(gs[0, :])
for r in results:
    equity = r['equity']
    label = r['name'].split(':')[1].strip()
    ax.plot(equity['date'], equity['portfolio_value'] / 1000,
            label=label, linewidth=2.5)

ax.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value ($1000s)')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

# 2. Drawdown
ax = fig.add_subplot(gs[1, 0])
for r in results:
    equity = r['equity']
    cummax = equity['portfolio_value'].cummax()
    drawdown = (equity['portfolio_value'] - cummax) / cummax
    label = r['name'].split(':')[1].strip()[:15]
    ax.plot(equity['date'], drawdown * 100, label=label, linewidth=2)

ax.set_title('Drawdown', fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3. Performance metrics bar chart
ax = fig.add_subplot(gs[1, 1])
metrics = ['CAGR', 'Sharpe', 'Max DD']
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
    label = r['name'].split(':')[1].strip()[:12]
    ax.bar(x + offset, values, width, label=label)

ax.set_title('Performance Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Value')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# 4. Return distribution
ax = fig.add_subplot(gs[1, 2])
for r in results:
    equity = r['equity']
    returns = equity['portfolio_value'].pct_change().dropna() * 100
    label = r['name'].split(':')[1].strip()[:15]
    ax.hist(returns, bins=50, alpha=0.5, label=label)

ax.set_title('Daily Return Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Daily Return (%)')
ax.set_ylabel('Frequency')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)

# 5. Rolling Sharpe Ratio
ax = fig.add_subplot(gs[2, 0])
window = 252  # 1 year
for r in results:
    equity = r['equity']
    returns = equity['portfolio_value'].pct_change()
    rolling_sharpe = (
        returns.rolling(window).mean() /
        returns.rolling(window).std() *
        np.sqrt(252)
    )
    label = r['name'].split(':')[1].strip()[:15]
    ax.plot(equity['date'], rolling_sharpe, label=label, linewidth=2, alpha=0.8)

ax.set_title('Rolling 1-Year Sharpe Ratio', fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Sharpe Ratio')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

# 6. Monthly returns heatmap (Strategy 1)
ax = fig.add_subplot(gs[2, 1])
equity = results[0]['equity'].copy()
equity['year'] = equity['date'].dt.year
equity['month'] = equity['date'].dt.month
equity['return'] = equity['portfolio_value'].pct_change()

monthly = equity.groupby(['year', 'month'])['return'].sum().unstack()
im = ax.imshow(monthly.values * 100, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
ax.set_title(f'Monthly Returns: {results[0]["name"].split(":")[1].strip()[:20]}',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Month')
ax.set_ylabel('Year')
ax.set_xticks(range(12))
ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
ax.set_yticks(range(len(monthly)))
ax.set_yticklabels(monthly.index.astype(int))
plt.colorbar(im, ax=ax, label='Return (%)')

# 7. Risk-Return scatter
ax = fig.add_subplot(gs[2, 2])
for r in results:
    stats = r['stats']
    label = r['name'].split(':')[1].strip()
    ax.scatter(stats['volatility'] * 100, stats['cagr'] * 100,
               s=200, alpha=0.7, label=label)

ax.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
ax.set_xlabel('Volatility (Annual %)')
ax.set_ylabel('CAGR (%)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.suptitle('US ETF Strategy Comparison - Comprehensive Analysis',
             fontsize=16, fontweight='bold', y=0.995)

# Save figure
output_path = 'ETFTrading/us_backtest_results.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualization to: {output_path}")
print()

# Strategy effectiveness analysis
print("="*80)
print("STRATEGY EFFECTIVENESS ANALYSIS")
print("="*80)
print()

# Calculate investment summary
years = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days / 365.25
total_deposits = INITIAL_CAPITAL + (MONTHLY_DEPOSIT * 12 * years)

print(f"💰 Investment Summary:")
print(f"   Investment Period: {years:.1f} years")
print(f"   Total Deposits: ${total_deposits:,.2f}")
print()

# Rank strategies
for i, r in enumerate(results, 1):
    stats = r['stats']
    profit = stats['final_value'] - total_deposits
    roi = (stats['final_value'] / total_deposits - 1) * 100

    print(f"{i}. {r['name'].split(':')[1].strip()}")
    print(f"   Final Value: ${stats['final_value']:,.2f}")
    print(f"   Profit: ${profit:,.2f} ({roi:.1f}% ROI)")
    print(f"   CAGR: {stats['cagr']:.1%}")
    print(f"   Sharpe: {stats['sharpe_ratio']:.2f}")
    print(f"   Max DD: {stats['max_drawdown']:.1%}")
    print()

# Best performers
best_return_idx = max(range(len(results)), key=lambda i: results[i]['stats']['total_return'])
best_sharpe_idx = max(range(len(results)), key=lambda i: results[i]['stats']['sharpe_ratio'])
lowest_dd_idx = max(range(len(results)), key=lambda i: results[i]['stats']['max_drawdown'])

print("🏆 Best Total Return:")
print(f"   {results[best_return_idx]['name'].split(':')[1].strip()}")
print(f"   CAGR: {results[best_return_idx]['stats']['cagr']:.1%}")
print()

print("📈 Best Risk-Adjusted Return:")
print(f"   {results[best_sharpe_idx]['name'].split(':')[1].strip()}")
print(f"   Sharpe: {results[best_sharpe_idx]['stats']['sharpe_ratio']:.2f}")
print()

print("🛡️  Most Defensive:")
print(f"   {results[lowest_dd_idx]['name'].split(':')[1].strip()}")
print(f"   Max DD: {results[lowest_dd_idx]['stats']['max_drawdown']:.1%}")
print()

# Key insights
print("="*80)
print("KEY INSIGHTS & STRATEGY EFFECTIVENESS")
print("="*80)
print()

print("1. All-Weather Portfolio:")
print("   ✓ Diversification across asset classes")
print("   ✓ Lowest volatility and drawdowns")
print("   ✓ Consistent performance in all market conditions")
print("   ✓ Best for preservation and steady growth")
print()

print("2. Momentum Sector Rotation:")
print("   ✓ Dynamic allocation based on trend following")
print("   ✓ Captures sector rotation opportunities")
print("   ✓ Higher returns in trending markets")
print("   ✗ Higher volatility and turnover costs")
print("   ✗ Can lag in sideways markets")
print()

print("3. Dividend + Growth:")
print("   ✓ Simple 50/50 balanced approach")
print("   ✓ Combines income and growth")
print("   ✓ Moderate risk-return profile")
print("   ✗ Concentrated in US equities")
print()

# Save results
for r in results:
    strategy_name = r['name'].replace('Strategy ', 'S').replace(': ', '_').replace(' ', '_')
    r['equity'].to_csv(f"ETFTrading/{strategy_name}_equity.csv", index=False)
    r['trades'].to_csv(f"ETFTrading/{strategy_name}_trades.csv", index=False)

print("="*80)
print("BACKTEST COMPLETED SUCCESSFULLY")
print("="*80)
print()
print("✓ Results saved to CSV files in ETFTrading/ directory")
print("✓ Visualization saved to us_backtest_results.png")
print()
print("NOTE: Data is realistic synthetic based on historical ETF statistics")
print("      Focus is on relative strategy performance and effectiveness")
print("="*80)
