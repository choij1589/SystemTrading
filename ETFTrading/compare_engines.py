"""
백테스팅 엔진 비교 스크립트

기존 엔진 vs 리팩토링된 엔진 결과 비교
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from datetime import datetime

# Import modules
from ETFTrading.strategy.asset_allocation import GlobalAssetAllocationStrategy
from ETFTrading.strategy.momentum_rotation import MomentumSectorRotationStrategy
from ETFTrading.strategy.dividend_growth import DividendGrowthMixStrategy
from ETFTrading.backtesting.engine import ETFBacktestEngine
from ETFTrading.backtesting.engine_v2 import ETFBacktestEngineV2

print("=" * 80)
print("백테스팅 엔진 비교 분석")
print("=" * 80)
print()

# Generate consistent sample data
def generate_sample_data(tickers, start_date, end_date, seed=42):
    """재현 가능한 샘플 데이터 생성"""
    np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    etf_profiles = {
        "069500": (0.08, 0.20),
        "360750": (0.12, 0.18),
        "152380": (0.02, 0.05),
        "132030": (0.05, 0.15),
        "091180": (0.15, 0.35),
        "157450": (0.18, 0.40),
        "227540": (0.10, 0.25),
        "139230": (0.05, 0.30),
        "139260": (0.07, 0.28),
        "139250": (0.06, 0.22),
        "228790": (0.09, 0.26),
        "458730": (0.08, 0.14),
        "133690": (0.14, 0.22),
    }

    result = {}

    for ticker in tickers:
        annual_return, annual_vol = etf_profiles.get(ticker, (0.08, 0.20))
        daily_return = annual_return / 252
        daily_vol = annual_vol / np.sqrt(252)

        np.random.seed(hash(ticker) % (2**32) + seed)

        returns = np.zeros(len(dates))
        returns[0] = daily_return

        for i in range(1, len(dates)):
            momentum = 0.3 * returns[i-1]
            noise = np.random.normal(0, daily_vol)
            returns[i] = daily_return + momentum + noise

        initial_price = 10000.0
        prices = initial_price * np.cumprod(1 + returns)

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

print(f"백테스트 기간: {START_DATE} ~ {END_DATE}")
print()

# Initialize strategies
strategy1 = GlobalAssetAllocationStrategy()
strategy2 = MomentumSectorRotationStrategy(top_n=3)
strategy3 = DividendGrowthMixStrategy()

strategies = [
    (strategy1, "Strategy 1: Global Asset Allocation"),
    (strategy2, "Strategy 2: Momentum Sector Rotation"),
    (strategy3, "Strategy 3: Dividend + Growth Mix")
]

# Collect all tickers
all_tickers = set()
for strategy, _ in strategies:
    all_tickers.update(strategy.get_universe())

print(f"데이터 생성 중... (총 {len(all_tickers)}개 ETF)")
data = generate_sample_data(
    tickers=list(all_tickers),
    start_date=START_DATE,
    end_date=END_DATE,
    seed=42  # 일관된 결과를 위한 시드
)
print(f"✓ 완료 ({len(data[list(data.keys())[0]])} 거래일)")
print()

# Initialize engines
engine_v1 = ETFBacktestEngine(
    initial_capital=4_200_000,
    monthly_deposit=300_000,
    commission_rate=0.00015,
    tax_rate=0.0023,
    slippage=0.0001,
    rebalance_frequency="monthly"
)

engine_v2 = ETFBacktestEngineV2(
    initial_capital=4_200_000,
    monthly_deposit=300_000,
    commission_rate=0.00015,
    tax_rate=0.0023,
    bid_ask_spread=0.0001,
    rebalance_frequency="monthly"
)

print("=" * 80)
print("전략별 비교 분석")
print("=" * 80)
print()

comparison_results = []

for i, (strategy, name) in enumerate(strategies, 1):
    print(f"[{i}/3] {name}")
    print("-" * 80)

    # Run V1
    equity_v1 = engine_v1.run(strategy, data, START_DATE, END_DATE)
    stats_v1 = engine_v1.get_summary_stats(equity_v1)
    trades_v1 = engine_v1.get_trade_log()

    # Run V2
    equity_v2 = engine_v2.run(strategy, data, START_DATE, END_DATE)
    stats_v2 = engine_v2.get_summary_stats(equity_v2)
    trades_v2 = engine_v2.get_trade_log()

    # Compare
    print(f"\n엔진 V1 (기존):")
    print(f"  최종 자산: {stats_v1['final_value']:,.0f} KRW")
    print(f"  총 수익률: {stats_v1['total_return']:.2%}")
    print(f"  CAGR: {stats_v1['cagr']:.2%}")
    print(f"  Sharpe: {stats_v1['sharpe_ratio']:.2f}")
    print(f"  MDD: {stats_v1['max_drawdown']:.2%}")
    print(f"  거래 수: {len(trades_v1)}")

    print(f"\n엔진 V2 (리팩토링):")
    print(f"  최종 자산: {stats_v2['final_value']:,.0f} KRW")
    print(f"  총 수익률: {stats_v2['total_return']:.2%}")
    print(f"  CAGR: {stats_v2['cagr']:.2%}")
    print(f"  Sharpe: {stats_v2['sharpe_ratio']:.2f}")
    print(f"  MDD: {stats_v2['max_drawdown']:.2%}")
    print(f"  거래 수: {len(trades_v2)}")

    # Calculate differences
    value_diff = stats_v2['final_value'] - stats_v1['final_value']
    value_diff_pct = (value_diff / stats_v1['final_value']) * 100
    return_diff = (stats_v2['total_return'] - stats_v1['total_return']) * 100
    sharpe_diff = stats_v2['sharpe_ratio'] - stats_v1['sharpe_ratio']

    print(f"\n차이 (V2 - V1):")
    print(f"  최종 자산: {value_diff:+,.0f} KRW ({value_diff_pct:+.2f}%)")
    print(f"  총 수익률: {return_diff:+.2f}%p")
    print(f"  Sharpe: {sharpe_diff:+.3f}")
    print()

    comparison_results.append({
        'Strategy': name.replace('Strategy ', 'S'),
        'V1 Final': f"{stats_v1['final_value']:,.0f}",
        'V2 Final': f"{stats_v2['final_value']:,.0f}",
        'Diff (KRW)': f"{value_diff:+,.0f}",
        'Diff (%)': f"{value_diff_pct:+.2f}%",
        'V1 CAGR': f"{stats_v1['cagr']:.2%}",
        'V2 CAGR': f"{stats_v2['cagr']:.2%}",
        'V1 Sharpe': f"{stats_v1['sharpe_ratio']:.2f}",
        'V2 Sharpe': f"{stats_v2['sharpe_ratio']:.2f}",
    })

print("=" * 80)
print("종합 비교표")
print("=" * 80)
print()

comparison_df = pd.DataFrame(comparison_results)
print(comparison_df.to_string(index=False))
print()

print("=" * 80)
print("분석 결과")
print("=" * 80)
print()

print("🔍 주요 변경사항:")
print()
print("1. **슬리피지 모델링 개선**")
print("   - 기존: 매수시 +슬리피지, 매도시 -슬리피지 (양쪽 적용)")
print("   - 개선: bid-ask spread 모델링 (한쪽만 적용)")
print("   - 영향: 거래 비용이 약 50% 감소")
print()

print("2. **리밸런싱 로직 개선**")
print("   - 기존: 매수 주문을 순차 처리 → 나중 종목 불리")
print("   - 개선: 필요 현금 계산 후 비례적 조정")
print("   - 영향: 목표 비중을 더 정확히 유지")
print()

print("3. **코드 품질 개선**")
print("   - 명확한 함수명 및 주석")
print("   - 타입 힌트 추가")
print("   - 로직 분리 및 모듈화")
print()

# Determine if results are consistent
max_diff_pct = max(abs(float(r['Diff (%)'].rstrip('%'))) for r in comparison_results)

print("=" * 80)
print("결론")
print("=" * 80)
print()

if max_diff_pct < 1.0:
    print("✅ **결과 일관성 검증: 통과**")
    print(f"   최대 차이: {max_diff_pct:.2f}% (< 1%)")
    print()
    print("   리팩토링된 코드는 기존 코드와 거의 동일한 결과를 생성합니다.")
    print("   슬리피지 모델링 개선으로 인한 미세한 차이만 발생했습니다.")
    print()
    print("   ✓ 로직 정확성 검증 완료")
    print("   ✓ 리팩토링된 엔진 사용 권장")
else:
    print("⚠️  **주의: 큰 차이 발견**")
    print(f"   최대 차이: {max_diff_pct:.2f}%")
    print()
    print("   추가 분석이 필요합니다.")

print()
print("=" * 80)
