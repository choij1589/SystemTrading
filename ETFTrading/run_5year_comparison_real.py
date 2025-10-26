"""
5-Year US ETF Strategy Comparison (2020-2024) - REAL DATA

Compare active strategies vs simple buy-and-hold benchmarks.
Uses REAL market data from Yahoo Finance (not synthetic).
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

# Import strategies
from ETFTrading.strategy.us_asset_allocation import USAllWeatherStrategy
from ETFTrading.strategy.us_momentum_rotation import USMomentumSectorRotationStrategy
from ETFTrading.strategy.us_dividend_growth import USDividendGrowthStrategy
from ETFTrading.strategy.benchmarks import (
    BuyHoldSPYStrategy,
    BuyHoldQQQStrategy,
    BuyHoldTLTStrategy,
    Classic6040Strategy
)
from ETFTrading.backtesting.engine import ETFBacktestEngine
from ETFTrading.data.yahoo_loader import YahooETFLoader  # REAL DATA

print("="*80)
print("5-YEAR US ETF STRATEGY COMPARISON (2020-2024)")
print("Active Strategies vs Buy-and-Hold Benchmarks")
print("📊 DATA SOURCE: REAL MARKET DATA (Yahoo Finance)")
print("="*80)
print()

# Configuration - LAST 5 YEARS
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

# Investment parameters (in USD)
INITIAL_CAPITAL = 10_000  # $10,000
MONTHLY_DEPOSIT = 500     # $500/month

print(f"📅 Backtest Period: {START_DATE} to {END_DATE} (5 years)")
print(f"💰 Initial Capital: ${INITIAL_CAPITAL:,}")
print(f"💵 Monthly Deposit: ${MONTHLY_DEPOSIT:,}")
print()

# Initialize ALL strategies (active + benchmarks)
print("📊 Initializing strategies...")
print()

# Active strategies
strategy1 = USAllWeatherStrategy()
strategy2 = USMomentumSectorRotationStrategy(top_n=3)
strategy3 = USDividendGrowthStrategy()

# Benchmark strategies
benchmark1 = BuyHoldSPYStrategy()
benchmark2 = BuyHoldQQQStrategy()
benchmark3 = BuyHoldTLTStrategy()
benchmark4 = Classic6040Strategy()

strategies = [
    # Active Strategies
    (strategy1, "Active 1: All-Weather", "active"),
    (strategy2, "Active 2: Momentum Rotation", "active"),
    (strategy3, "Active 3: Dividend + Growth", "active"),

    # Benchmarks
    (benchmark1, "Benchmark: 100% SPY", "benchmark"),
    (benchmark2, "Benchmark: 100% QQQ", "benchmark"),
    (benchmark3, "Benchmark: 100% TLT", "benchmark"),
    (benchmark4, "Benchmark: 60/40", "benchmark"),
]

print("🎯 ACTIVE STRATEGIES:")
print(f"  1. {strategy1.name}")
print(f"     → {', '.join(strategy1.get_universe())}")
print(f"  2. {strategy2.name}")
print(f"     → {', '.join(strategy2.get_universe())}")
print(f"  3. {strategy3.name}")
print(f"     → {', '.join(strategy3.get_universe())}")
print()

print("📊 BENCHMARK STRATEGIES:")
print(f"  1. {benchmark1.name} (S&P 500)")
print(f"  2. {benchmark2.name} (Nasdaq 100)")
print(f"  3. {benchmark3.name} (Long-term Bonds)")
print(f"  4. {benchmark4.name} (Classic Portfolio)")
print()

# Collect all tickers
all_tickers = set()
for strategy, _, _ in strategies:
    all_tickers.update(strategy.get_universe())

# Add BND for 60/40 strategy
all_tickers.add("BND")

print(f"📦 Fetching REAL market data from Yahoo Finance...")
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

print(f"✓ Successfully loaded REAL data for {len(data)} ETFs")

# Show key ETF statistics
print(f"\n✓ Key ETF Performance (5-year REAL data):")
key_etfs = ["SPY", "QQQ", "TLT", "SCHD", "BND"]
for ticker in key_etfs:
    if ticker in data:
        df = data[ticker]
        total_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        annual_vol = df['close'].pct_change().std() * np.sqrt(252) * 100
        print(f"   {ticker}: Return {total_return:>7.1f}%, Vol {annual_vol:>5.1f}%")
print()

# Initialize backtesting engine
engine = ETFBacktestEngine(
    initial_capital=INITIAL_CAPITAL,
    monthly_deposit=MONTHLY_DEPOSIT,
    commission_rate=0.0001,
    tax_rate=0.0,
    bid_ask_spread=0.0001,
    rebalance_frequency="monthly"
)

# Run backtests
results = []

print("="*80)
print("RUNNING BACKTESTS")
print("="*80)
print()

for i, (strategy, name, strategy_type) in enumerate(strategies, 1):
    print(f"[{i}/{len(strategies)}] {name}...")
    print("-" * 60)

    try:
        equity = engine.run(strategy, data, START_DATE, END_DATE)
        stats = engine.get_summary_stats(equity)
        trades = engine.get_trade_log()

        results.append({
            'strategy': strategy,
            'name': name,
            'type': strategy_type,
            'equity': equity,
            'stats': stats,
            'trades': trades
        })

        print(f"✓ Completed")
        print(f"  Final: ${stats['final_value']:,.0f} | CAGR: {stats['cagr']:.1%} | "
              f"Sharpe: {stats['sharpe_ratio']:.2f} | DD: {stats['max_drawdown']:.1%}")
        print()

    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        print()

# Separate active and benchmark results
active_results = [r for r in results if r['type'] == 'active']
benchmark_results = [r for r in results if r['type'] == 'benchmark']

# Performance comparison
print("="*80)
print("PERFORMANCE COMPARISON (REAL DATA)")
print("="*80)
print()

print("🎯 ACTIVE STRATEGIES:")
print("-" * 80)
comparison_data = []
for r in active_results:
    stats = r['stats']
    comparison_data.append({
        'Strategy': r['name'].replace('Active ', '').replace(': ', ':\n'),
        'Final ($)': f"${stats['final_value']:,.0f}",
        'CAGR': f"{stats['cagr']:.1%}",
        'Sharpe': f"{stats['sharpe_ratio']:.2f}",
        'Max DD': f"{stats['max_drawdown']:.1%}",
        'Vol': f"{stats['volatility']:.1%}",
    })

df = pd.DataFrame(comparison_data)
print(df.to_string(index=False))
print()

print("📊 BENCHMARK STRATEGIES:")
print("-" * 80)
comparison_data = []
for r in benchmark_results:
    stats = r['stats']
    comparison_data.append({
        'Strategy': r['name'].replace('Benchmark: ', ''),
        'Final ($)': f"${stats['final_value']:,.0f}",
        'CAGR': f"{stats['cagr']:.1%}",
        'Sharpe': f"{stats['sharpe_ratio']:.2f}",
        'Max DD': f"{stats['max_drawdown']:.1%}",
        'Vol': f"{stats['volatility']:.1%}",
    })

df = pd.DataFrame(comparison_data)
print(df.to_string(index=False))
print()

# Calculate investment summary
years = 5.0
total_deposits = INITIAL_CAPITAL + (MONTHLY_DEPOSIT * 12 * years)

print("="*80)
print("STRATEGY vs BENCHMARK ANALYSIS")
print("="*80)
print()

print(f"💰 Total Investment: ${total_deposits:,.0f} over {years:.0f} years")
print()

# Rank all strategies
all_by_return = sorted(results, key=lambda x: x['stats']['total_return'], reverse=True)
all_by_sharpe = sorted(results, key=lambda x: x['stats']['sharpe_ratio'], reverse=True)
all_by_dd = sorted(results, key=lambda x: x['stats']['max_drawdown'], reverse=True)

print("🏆 RANKING BY TOTAL RETURN:")
for i, r in enumerate(all_by_return[:5], 1):
    stats = r['stats']
    type_icon = "🎯" if r['type'] == 'active' else "📊"
    print(f"{i}. {type_icon} {r['name']}")
    print(f"   Final: ${stats['final_value']:,.0f} | CAGR: {stats['cagr']:.1%}")
print()

print("📈 RANKING BY SHARPE RATIO (Risk-Adjusted):")
for i, r in enumerate(all_by_sharpe[:5], 1):
    stats = r['stats']
    type_icon = "🎯" if r['type'] == 'active' else "📊"
    print(f"{i}. {type_icon} {r['name']}")
    print(f"   Sharpe: {stats['sharpe_ratio']:.2f} | CAGR: {stats['cagr']:.1%} | DD: {stats['max_drawdown']:.1%}")
print()

print("🛡️  RANKING BY DRAWDOWN (Most Defensive):")
for i, r in enumerate(all_by_dd[:5], 1):
    stats = r['stats']
    type_icon = "🎯" if r['type'] == 'active' else "📊"
    print(f"{i}. {type_icon} {r['name']}")
    print(f"   Max DD: {stats['max_drawdown']:.1%} | CAGR: {stats['cagr']:.1%}")
print()

# Alpha calculation (vs SPY)
spy_result = [r for r in results if 'SPY' in r['name'] and r['type'] == 'benchmark'][0]
spy_cagr = spy_result['stats']['cagr']

print("="*80)
print("ALPHA ANALYSIS (vs SPY Benchmark)")
print("="*80)
print()

for r in active_results:
    alpha = r['stats']['cagr'] - spy_cagr
    alpha_sign = "+" if alpha >= 0 else ""
    print(f"{r['name'].replace('Active ', '')}:")
    print(f"  CAGR: {r['stats']['cagr']:.1%} vs SPY {spy_cagr:.1%}")
    print(f"  Alpha: {alpha_sign}{alpha:.1%}")
    print()

# Create comprehensive visualization
print("="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)
print()

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1. Equity curves comparison (top, spanning 2 columns)
ax = fig.add_subplot(gs[0, :2])

# Plot active strategies
for r in active_results:
    equity = r['equity']
    label = r['name'].replace('Active ', '').replace(': ', ' ')
    ax.plot(equity['date'], equity['portfolio_value'] / 1000,
            label=label, linewidth=2.5, alpha=0.9)

# Plot benchmarks
for r in benchmark_results:
    equity = r['equity']
    label = r['name'].replace('Benchmark: ', '')
    ax.plot(equity['date'], equity['portfolio_value'] / 1000,
            label=label, linewidth=2, linestyle='--', alpha=0.7)

ax.set_title('Portfolio Value: Active Strategies vs Benchmarks (2020-2024) - REAL DATA',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Portfolio Value ($1000s)')
ax.legend(fontsize=9, loc='upper left', ncol=2)
ax.grid(True, alpha=0.3)

# 2. CAGR Comparison Bar Chart
ax = fig.add_subplot(gs[0, 2])
cagrs = [r['stats']['cagr'] * 100 for r in results]
names = [r['name'].replace('Active ', 'A:').replace('Benchmark: ', 'B:').replace(': ', '\n')[:20]
         for r in results]
colors = ['#2E86AB' if r['type'] == 'active' else '#A23B72' for r in results]

bars = ax.barh(range(len(names)), cagrs, color=colors, alpha=0.8)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('CAGR (%)')
ax.set_title('CAGR Comparison', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(x=spy_cagr * 100, color='red', linestyle='--', alpha=0.5, label='SPY')

# 3. Sharpe Ratio Comparison
ax = fig.add_subplot(gs[1, 0])
sharpes = [r['stats']['sharpe_ratio'] for r in results]

bars = ax.barh(range(len(names)), sharpes, color=colors, alpha=0.8)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('Sharpe Ratio')
ax.set_title('Risk-Adjusted Returns', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# 4. Max Drawdown Comparison
ax = fig.add_subplot(gs[1, 1])
dds = [abs(r['stats']['max_drawdown']) * 100 for r in results]

bars = ax.barh(range(len(names)), dds, color=colors, alpha=0.8)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('Max Drawdown (%)')
ax.set_title('Maximum Drawdown', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_xaxis()  # Lower is better

# 5. Risk-Return Scatter
ax = fig.add_subplot(gs[1, 2])

for r in active_results:
    stats = r['stats']
    ax.scatter(stats['volatility'] * 100, stats['cagr'] * 100,
               s=300, alpha=0.7, marker='o',
               label=r['name'].replace('Active ', '').replace(': ', ' ')[:15])

for r in benchmark_results:
    stats = r['stats']
    ax.scatter(stats['volatility'] * 100, stats['cagr'] * 100,
               s=200, alpha=0.6, marker='s',
               label=r['name'].replace('Benchmark: ', '')[:15])

ax.set_xlabel('Volatility (Annual %)')
ax.set_ylabel('CAGR (%)')
ax.set_title('Risk-Return Profile', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='best')
ax.grid(True, alpha=0.3)

# 6. Drawdown over time (Active strategies)
ax = fig.add_subplot(gs[2, 0])
for r in active_results:
    equity = r['equity']
    cummax = equity['portfolio_value'].cummax()
    drawdown = (equity['portfolio_value'] - cummax) / cummax
    label = r['name'].replace('Active ', '').replace(': ', ' ')[:15]
    ax.plot(equity['date'], drawdown * 100, label=label, linewidth=2)

ax.set_title('Drawdown: Active Strategies', fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 7. Drawdown over time (Benchmarks)
ax = fig.add_subplot(gs[2, 1])
for r in benchmark_results:
    equity = r['equity']
    cummax = equity['portfolio_value'].cummax()
    drawdown = (equity['portfolio_value'] - cummax) / cummax
    label = r['name'].replace('Benchmark: ', '')
    ax.plot(equity['date'], drawdown * 100, label=label, linewidth=2)

ax.set_title('Drawdown: Benchmarks', fontsize=12, fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Drawdown (%)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 8. Final Value Comparison
ax = fig.add_subplot(gs[2, 2])
final_values = [r['stats']['final_value'] / 1000 for r in results]

bars = ax.barh(range(len(names)), final_values, color=colors, alpha=0.8)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('Final Value ($1000s)')
ax.set_title('Final Portfolio Value', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.suptitle('5-Year Strategy Comparison (2020-2024): REAL MARKET DATA',
             fontsize=16, fontweight='bold', y=0.995)

# Save
output_path = 'ETFTrading/us_5year_comparison_REAL.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualization to: {output_path}")
print()

# Save detailed results
for r in results:
    name = r['name'].replace(' ', '_').replace(':', '').replace('/', '')
    r['equity'].to_csv(f"ETFTrading/5yr_REAL_{name}_equity.csv", index=False)

print("="*80)
print("KEY INSIGHTS (5-Year Period - REAL DATA)")
print("="*80)
print()

print("1. ACTIVE vs PASSIVE:")
active_avg_cagr = np.mean([r['stats']['cagr'] for r in active_results])
bench_avg_cagr = np.mean([r['stats']['cagr'] for r in benchmark_results])
print(f"   Active Strategies Avg CAGR: {active_avg_cagr:.1%}")
print(f"   Benchmarks Avg CAGR: {bench_avg_cagr:.1%}")
print()

print("2. SIMPLICITY TEST:")
qqq_result = [r for r in results if '100% QQQ' in r['name']][0]
div_result = [r for r in results if 'Dividend + Growth' in r['name']][0]
print(f"   100% QQQ: {qqq_result['stats']['cagr']:.1%} CAGR")
print(f"   Div+Growth (50% SCHD + 50% QQQ): {div_result['stats']['cagr']:.1%} CAGR")
diff_pct = ((div_result['stats']['cagr'] - qqq_result['stats']['cagr']) / qqq_result['stats']['cagr']) * 100
print(f"   → Rebalancing impact: {diff_pct:+.1f}%")
print()

print("3. DIVERSIFICATION VALUE:")
spy_dd = spy_result['stats']['max_drawdown']
aw_result = [r for r in results if 'All-Weather' in r['name']][0]
aw_dd = aw_result['stats']['max_drawdown']
print(f"   100% SPY Max DD: {spy_dd:.1%}")
print(f"   All-Weather Max DD: {aw_dd:.1%}")
print(f"   → Diversification reduces drawdown by: {abs(spy_dd - aw_dd):.1%}pp")
print()

print("="*80)
print("COMPLETED SUCCESSFULLY")
print("="*80)
print()
print("✓ 5-year backtest complete with REAL Yahoo Finance data")
print("✓ Active strategies vs benchmarks analyzed")
print("✓ Results saved to CSV and PNG")
print("="*80)
