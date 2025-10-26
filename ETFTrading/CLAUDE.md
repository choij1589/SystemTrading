# CLAUDE.md - ETF Trading System

This file provides guidance to Claude Code when working with the ETF Trading System.

## **CRITICAL RULE: NO RANDOMNESS**

**모든 로직에서 임의성(randomness)을 완전히 배제해야 합니다.**

### Deterministic Data Requirement

1. **NEVER generate random data** for backtesting or analysis
2. **ALWAYS use real market data** from approved API sources
3. **All backtesting results must be reproducible** - running the same backtest twice should produce identical results
4. **No random seeds, no simulated data** - except for explicit unit tests marked as such

### Approved Data Sources

**Primary (Production):**
- Korea Investment Securities API (`data/kis_client.py`)
  - Real-time and historical Korean ETF data
  - Requires API credentials in `config/secrets.yaml`

**Secondary (Development/Testing):**
- Yahoo Finance (`data/yahoo_loader.py`)
  - **WARNING**: Korean ETF tickers (.KS suffix) often have access issues on Yahoo Finance
  - Use only for US-listed ETFs or testing framework
  - **Not suitable for Korean ETF production backtests**

**NEVER use:**
- Random price generators
- Simulated returns with `np.random`
- Any non-deterministic data generation

### Data Caching

To ensure reproducibility:
- Use `use_cache=True` when loading data
- Cached data in `data/cache/` directory ensures identical results across runs
- Cache files use parquet format for efficiency and precision

### Example - CORRECT Approach

```python
# CORRECT: Load real market data
from ETFTrading.data.kis_client import KISClient

client = KISClient()
data = client.load_multiple(
    tickers=["069500", "360750"],
    start_date="2020-01-01",
    end_date="2024-12-31",
    use_cache=True  # Ensures reproducibility
)
```

### Example - INCORRECT Approach

```python
# WRONG: Never do this in production code
np.random.seed(42)
prices = initial_price * np.cumprod(1 + np.random.normal(0, 0.02, 1000))
```

## Project Overview

### Directory Structure

```
ETFTrading/
├── data/
│   ├── kis_client.py          # Korea Investment Securities API (PRIMARY)
│   ├── yahoo_loader.py        # Yahoo Finance (SECONDARY - limited support)
│   └── cache/                 # Cached market data (parquet files)
│
├── strategy/
│   ├── base.py                # Base strategy interface
│   ├── asset_allocation.py    # Strategy 1: Global Asset Allocation
│   ├── momentum_rotation.py   # Strategy 2: Momentum Sector Rotation
│   └── dividend_growth.py     # Strategy 3: Dividend + Growth Mix
│
├── backtesting/
│   └── engine.py              # Backtesting engine with DCA and rebalancing
│
├── config/
│   ├── config.yaml            # Investment parameters
│   ├── etf_universe.yaml      # ETF ticker universe
│   └── secrets.yaml.example   # API credentials template
│
├── run_backtest.py            # Main backtest execution script
├── CLAUDE.md                  # This file - development guidelines
├── README.md                  # User documentation
├── STRATEGY_LOGIC.md          # Strategy logic details
└── BACKTEST_REPORT.md         # Latest backtest results
```

## Investment Parameters

### Core Settings
- **Initial Capital**: 4,200,000 KRW
- **Monthly Deposit**: 300,000 KRW (Dollar Cost Averaging)
- **Rebalancing**: Monthly
- **Commission**: 0.015% (per trade)
- **Tax**: 0.23% (on sells only)
- **Bid-Ask Spread**: 0.01%

### Strategies

#### Strategy 1: Global Asset Allocation (안정형)
- Fixed allocation: 30% Korean stocks, 40% US stocks, 20% bonds, 10% gold
- Low volatility, all-weather portfolio
- Implementation: `strategy/asset_allocation.py`

#### Strategy 2: Momentum Sector Rotation (공격형)
- Dynamic allocation based on 3-month and 6-month momentum
- Selects top 3 sectors with positive momentum
- Equal weight distribution (33.33% each)
- High risk, trend-following approach
- Implementation: `strategy/momentum_rotation.py`

#### Strategy 3: Dividend + Growth Mix (균형형)
- Fixed allocation: 50% dividend stocks, 50% growth stocks
- Balanced income and capital gains
- Implementation: `strategy/dividend_growth.py`

## Backtesting Engine

### Key Features

1. **Monthly Deposits (DCA)**
   - Adds 300,000 KRW at the start of each month
   - Simulates regular savings plan

2. **Rebalancing**
   - Monthly rebalancing to target weights
   - Proportional cash allocation when funds are insufficient
   - Integer share constraints (no fractional shares)

3. **Transaction Costs**
   - Commission: 0.015% on both buy and sell
   - Tax: 0.23% on sell orders only
   - Bid-ask spread modeling (not double slippage)

4. **Performance Metrics**
   - CAGR: Compound Annual Growth Rate
   - Sharpe Ratio: Risk-adjusted returns
   - Maximum Drawdown (MDD): Largest peak-to-trough decline
   - Win Rate: Percentage of profitable trades

### Transaction Cost Modeling

The engine uses **bid-ask spread modeling**, not slippage:

```python
# Buy at ask price (higher)
execution_price = price * (1 + bid_ask_spread / 2)

# Sell at bid price (lower)
execution_price = price * (1 - bid_ask_spread / 2)
```

This is more realistic than applying "slippage" to both sides.

### Proportional Cash Allocation

When rebalancing with insufficient cash, the engine distributes cash proportionally:

```python
# Calculate total required cash
required = sum(buy_costs)

# Scale all buys proportionally if needed
if required > available_cash:
    scale = available_cash / required
    for each_buy:
        actual_amount = target_amount * scale
```

This ensures fair allocation across all positions.

## Development Guidelines

### Code Quality

1. **Type hints required** for all functions
2. **Docstrings required** for public methods
3. **No magic numbers** - use config files or named constants
4. **Vectorized operations** - prefer pandas/numpy over loops

### Testing

1. **Unit tests** should use mock data (clearly marked)
2. **Integration tests** must use real cached data
3. **Backtest validation** - verify results are reproducible

### Configuration

All parameters should be in `config/config.yaml`:

```yaml
investment:
  initial_capital: 4_200_000
  monthly_deposit: 300_000

rebalancing:
  frequency: "monthly"

transaction:
  commission_rate: 0.00015
  tax_rate: 0.0023
  bid_ask_spread: 0.0001
```

### API Credentials

Store in `config/secrets.yaml` (NEVER commit):

```yaml
kis:
  app_key: "YOUR_APP_KEY"
  app_secret: "YOUR_APP_SECRET"
  account_number: "YOUR_ACCOUNT"
  is_real: false  # true for real trading, false for paper trading
```

## Running Backtests

### Command

```bash
cd /home/user/SystemTrading
python ETFTrading/run_backtest.py
```

### Output

- **Terminal**: Summary statistics and comparison table
- **Files**:
  - `ETFTrading/backtest_results.png` - Visualization charts
  - `ETFTrading/S*_equity.csv` - Equity curves for each strategy
  - `ETFTrading/S*_trades.csv` - Trade logs for each strategy

### Validation

After each backtest run:
1. Verify results are reproducible (run twice, compare)
2. Check that data source is real market data (not random)
3. Review transaction costs are applied correctly
4. Confirm cash management is realistic (no negative cash)

## Common Issues

### Yahoo Finance Access Errors

**Problem**: Korean ETF tickers fail with HTTP 403 or "no data available"

**Solution**: Use Korea Investment Securities API instead:
```python
from ETFTrading.data.kis_client import KISClient
client = KISClient()
# ... use client to load data
```

### Import Errors

**Problem**: Module not found errors when running scripts

**Solution**: Run from project root:
```bash
cd /home/user/SystemTrading
python ETFTrading/run_backtest.py
```

### Data Cache Issues

**Problem**: Stale or corrupted cache data

**Solution**: Clear cache directory:
```bash
rm -rf ETFTrading/data/cache/*.parquet
```

## Future Enhancements

### Planned Features

1. **Walk-forward optimization**
   - Rolling window backtests
   - Out-of-sample validation

2. **Risk management**
   - Maximum drawdown limits
   - Position size limits
   - Stop-loss rules

3. **Live trading**
   - Paper trading mode (test with real prices, fake orders)
   - Live trading mode (real orders via KIS API)

4. **Portfolio analytics**
   - Factor exposure analysis
   - Correlation matrices
   - Monte Carlo simulations (using historical bootstrapping, not random)

### Enhancement Guidelines

When adding features:
1. **Maintain determinism** - no random data generation
2. **Add configuration** - make parameters adjustable
3. **Write tests** - verify correctness and reproducibility
4. **Update documentation** - keep CLAUDE.md current

## References

- **README.md**: User-facing documentation
- **STRATEGY_LOGIC.md**: Detailed strategy algorithms
- **BACKTEST_REPORT.md**: Latest performance results
- **config/etf_universe.yaml**: Available ETF tickers and metadata
