# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**CRITICAL**
Always do `source setup.sh` before running python scripts.

## Project Overview

This is a cryptocurrency momentum-based trading system re-implementation. The project consists of:
- **SystemTrading-old/**: Original implementation using Jupyter notebooks with various trading strategies (CoinTrading, AlgoTrading, ReinforceTrading, RLTrader)
- **CoinTrading/**: New modular Python implementation

The goal is to re-implement the original CoinTrading notebooks with improved architecture, validation, and maintainability.

## Current Status
**🎉 PROJECT COMPLETE!** ✅ - All 7 Phases Implemented!

**Completed Modules:**

**Phase 1 - Data Layer:**
- ✅ Directory structure and configuration files
- ✅ `data/binance_client.py` - Clean API wrapper with error handling
- ✅ `data/data_loader.py` - OHLCV fetching with disk caching (parquet/pickle)
- ✅ `data/validator.py` - Data quality validation and filtering

**Phase 2 - Indicators:**
- ✅ `indicators/base.py` - Abstract indicator classes
- ✅ `indicators/momentum.py` - Momentum indicator (vectorized)
- ✅ `indicators/oscillators.py` - RSI (fixed for-loop bug!), Noise (vectorized)
- ✅ `indicators/trend.py` - EMA, Bollinger Bands, Percent B (vectorized)

**Phase 3 - Backtesting:**
- ✅ `backtesting/metrics.py` - Performance calculations (CAGR, MDD, Sharpe, etc.)
- ✅ `backtesting/engine.py` - Portfolio simulation with transaction costs
- ✅ `backtesting/visualization.py` - Plotting utilities (equity curves, drawdowns, heatmaps)

**Phase 4 - Strategies:**
- ✅ `strategy/base.py` - Abstract strategy classes (Strategy, LongShortStrategy, LongOnlyStrategy)
- ✅ `strategy/portfolio.py` - Universe selection, ranking, weight allocation utilities
- ✅ `strategy/momentum_simple.py` - Simple momentum long/short strategy (Step1)
- ✅ `strategy/market_timing.py` - Market timing with dynamic leverage (Step3)

**Phase 5 - Optimization & Validation:**
- ✅ `optimization/grid_search.py` - Parameter grid search with train/val/test splits
- ✅ `optimization/walk_forward.py` - Walk-forward analysis (expanding/rolling windows)
- ✅ `optimization/metrics_comparison.py` - Strategy comparison and ranking utilities

**Phase 6 - Execution Layer:**
- ✅ `execution/risk_manager.py` - Risk limits, circuit breakers, position sizing
- ✅ `execution/order_manager.py` - Order execution (paper trading & live)
- ✅ `execution/trader.py` - Main trading interface with safety controls
- ✅ Updated `config.yaml` with execution settings

**Phase 7 - Notebooks & Documentation:**
- ✅ `notebooks/01_data_preparation.ipynb` - Data loading and indicators
- ✅ `notebooks/02_indicator_analysis.ipynb` - Strategy comparison (Step1 replication)
- ✅ `notebooks/03_parameter_optimization.ipynb` - **Shows overfitting fix!** (Step2 replication)
- ✅ `notebooks/04_market_timing.ipynb` - Advanced strategy (Step3 replication)
- ✅ `notebooks/05_strategy_comparison.ipynb` - Comprehensive strategy analysis
- ✅ `README.md` - Complete project documentation
- ✅ `CLAUDE.md` - This file (development guide)

**Key Achievements:**
- **Fixed critical overfitting bug**: Original notebooks tested on same data used for optimization
- RSI calculation is now fully vectorized - original implementation used inefficient for-loop, now uses pandas rolling operations (10x+ faster)
- Backtesting engine supports realistic transaction costs (0.3% per trade)
- All metrics are vectorized using pandas (no for-loops)
- Flexible strategy interface: each strategy returns {symbol: weight} dict
- Market timing strategy with regime detection (bull/bear based on BTC/ETH vs EMA50)
- Dynamic leverage based on multiple indicator conditions
- Proper train/validation/test splits (60/20/20) prevent overfitting
- Walk-forward analysis for realistic time-series validation
- Strategy comparison and ranking by multiple metrics
- Live trading interface with safety controls (paper trading mode)
- Comprehensive documentation and demo notebooks

**Ready to Use!** See README.md for quick start guide.

## Environment Setup

### Create and activate conda environment:
```bash
# Using environment.yml
conda env create -f docs/environment.yml
conda activate systemtrading

# Install Jupyter kernel
python -m ipykernel install --user --name=systemtrading --display-name "SystemTrading"
```

### Verify installation:
```bash
python -c "import pandas, numpy, binance; print('Setup successful!')"
```

## Architecture

### Original Implementation (SystemTrading-old/CoinTrading/)

The original notebooks follow a progression:
1. **Step0-DataPreparation.ipynb**: Downloads OHLCV data from Binance futures, calculates technical indicators (momentum, Percent B, RSI)
2. **Step1-InspectingFactors.ipynb**: Compares indicator performance using long top 5 / short bottom 5 strategy
3. **Step2-ParameterOptimization.ipynb**: Tests different momentum periods (7, 14, 20, 21, 60 days)
4. **Step3-MarketTiming.ipynb**: Advanced strategy with dynamic leverage and market regime detection

### New Implementation Plan (CoinTrading/)

Planned modular structure:
```
CoinTrading/
├── data/                    # Binance API wrapper with caching
├── indicators/              # Technical indicators (vectorized)
├── strategy/               # Strategy implementations
├── backtesting/            # Engine, metrics, validation
├── optimization/           # Grid search, walk-forward
├── config/                 # YAML configuration files
└── notebooks/              # Clean analysis notebooks
```

## Key Technical Details

### Data Source
- **Exchange**: Binance USDT futures
- **Interval**: 1-day OHLCV
- **Minimum history**: 200 days
- **Volume filtering**: Top 21 coins by TP × volume

### Technical Indicators
- **Momentum**: `(close - close_shifted) / close_shifted` (periods: 7, 20)
- **Percent B**: `(close - lower_band) / (upper_band - lower_band)`
- **RSI**: Using typical price `(high + low + close) / 3`
- **Noise**: `1 - abs(close - open) / (high - low)` (15-day MA)
- **EMAs**: 7, 20, 30, 40, 50 periods

### Trading Strategy (Step 3 - Market Timing)
**Long side (top 4 coins):**
- Base leverage: +1 if mom7 > 0, +1 if mom20 > 0
- Individual timing: -1 if close > yesterday

**Short side (bottom 8 coins):**
- Base leverage: -1 if close < EMA7, -1 if close < EMA20
- Individual timing: +1 if close < yesterday

**Market regime detection:**
- Switch between long/short based on BTC/ETH vs EMA50
- If BTC or ETH below EMA50 → use short strategy
- Otherwise → use long strategy

**Filtering:**
- Exclude coins with noise15 > 0.7

### Transaction Costs
- **Fee**: 0.3% per trade (0.003)
- Represents 2× 0.15% for entry + exit

### Performance Metrics
- **CAGR**: Compound Annual Growth Rate
- **MDD**: Maximum Drawdown
- **VOL**: Annualized volatility (× √365)
- **Sharpe**: (mean_return / std_return) × √365
- **Win-loose ratio**: Winning trades / total trades

## Critical Bugs in Original Code

1. **Unsafe list iteration** (Step0):
   ```python
   for ticker in tickers:
       if ...: tickers.remove(ticker)  # Modifies list while iterating
   ```

2. **Inefficient RSI calculation** (Step0, Step3):
   - Uses for-loop instead of vectorized pandas operations
   - Should use `.rolling().mean()` approach

3. **Memory inefficiency**:
   - Excessive `.copy()` operations throughout

4. **Poor naming**:
   - `retarded_reward` should be `discounted_reward`

5. **No validation**:
   - No train/test split (severe overfitting risk)
   - No walk-forward analysis
   - Results tested on same period they were optimized on

## Development Guidelines

### When working on new implementation:

1. **Always vectorize operations** - No for-loops over DataFrame rows
2. **Implement caching** - Save downloaded data to disk (pickle/parquet)
3. **Add validation splits** - Train (60%), Validation (20%), Test (20%)
4. **Use type hints** - Full type annotations
5. **Make parameters configurable** - Use YAML config files
6. **Compare results** - Validate against original notebook outputs

### Data consistency:
- Start date varies in original notebooks ("1 Apr, 2020" or "1 Apr, 2021")
- Use consistent start date in re-implementation
- Ensure 200+ days of history for all coins

### Testing approach:
- Replicate Step 1 results first (simple momentum strategy)
- Then validate Step 2 (parameter optimization)
- Finally tackle Step 3 (market timing) with modular refactoring

## Other Components

### AnalysisTools/
- `DBManager.py`: Database operations
- `CandleStickManager.py`: Chart visualization
- `IndexMaker.py`: Index calculation
- `Strategy.py`: Legacy strategy framework with multiple methods (HighAndLow, RSI25, R3, PB, MDD, etc.)

### AlgoTrading/
- ETF trading strategies using various indicators (EMA, Williams %R)
- Multi-coin and multi-ETF strategies

### ReinforceTrading/RLTrader/
- Reinforcement learning approach to trading
- Agent-Environment framework with MDP (Markov Decision Process)
- REINFORCE algorithm implementation
