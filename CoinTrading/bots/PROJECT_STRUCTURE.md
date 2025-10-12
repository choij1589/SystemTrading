# Upbit Rebalancing Bot - Project Structure

## Directory Layout

```
SystemTrading/
├── CoinTrading/
│   ├── data/
│   │   ├── upbit_client.py              ← Upbit API wrapper (pyupbit)
│   │   ├── binance_client.py            (existing)
│   │   └── data_loader.py               (existing)
│   │
│   ├── execution/
│   │   ├── upbit_order_manager.py       ← Portfolio rebalancing logic
│   │   ├── order_manager.py             (existing - for Binance)
│   │   ├── risk_manager.py              (existing)
│   │   └── trader.py                    (existing)
│   │
│   ├── bots/
│   │   ├── upbit_rebalancer.py          ← Main bot script (executable)
│   │   ├── README.md                    ← Quick reference guide
│   │   ├── SETUP_GUIDE.md               ← Complete setup instructions
│   │   └── PROJECT_STRUCTURE.md         ← This file
│   │
│   ├── config/
│   │   ├── upbit_config.yaml            ← Bot configuration
│   │   ├── upbit_secrets.yaml           ← API keys (NOT in git)
│   │   ├── upbit_secrets.yaml.template  ← Secrets template
│   │   └── config.yaml                  (existing - for Binance)
│   │
│   ├── scripts/
│   │   ├── setup_cron.sh                ← Interactive cron setup
│   │   └── test_bot.sh      1            ← Quick test helper
│   │
│   ├── logs/
│   │   ├── upbit_rebalancer.log         ← Main bot log (auto-rotated)
│   │   ├── cron.log                     ← Cron execution log
│   │   └── rebalance_history.csv        ← Execution history
│   │
│   ├── strategy/                        (existing - not used by bot)
│   ├── backtesting/                     (existing - not used by bot)
│   └── indicators/                      (existing - not used by bot)
│
├── .gitignore                           ← Updated with upbit_secrets.yaml
├── setup.sh                             (existing - conda activation)
└── CLAUDE.md                            (existing - project docs)
```

## File Descriptions

### Core Bot Files

**[CoinTrading/bots/upbit_rebalancer.py](upbit_rebalancer.py)**
- Main executable script
- One-shot execution (no continuous loop)
- Loads config and secrets
- Orchestrates rebalancing workflow
- Handles logging and error reporting
- Saves execution history

**[CoinTrading/data/upbit_client.py](../data/upbit_client.py)**
- Wrapper around pyupbit library
- Methods for:
  - Fetching OHLCV data
  - Getting current prices
  - Getting account balances
  - Placing market orders (buy/sell)
- Supports both authenticated and public access

**[CoinTrading/execution/upbit_order_manager.py](../execution/upbit_order_manager.py)**
- Portfolio rebalancing logic
- Methods for:
  - Getting portfolio value and positions
  - Calculating current weights
  - Computing rebalance orders
  - Executing trades
- Supports dry-run and paper trading modes

### Configuration Files

**[CoinTrading/config/upbit_config.yaml](../config/upbit_config.yaml)**
```yaml
# Configuration settings:
- Strategy: target allocation (25/25/25/25)
- Execution: paper trading, dry run, min order size
- Risk: position limits, loss limits
- Logging: log level, rotation
- Notifications: email, telegram (optional)
```

**CoinTrading/config/upbit_secrets.yaml** (NOT in git)
```yaml
# API credentials:
- Upbit access key
- Upbit secret key
- Email/Telegram tokens (optional)
```

### Helper Scripts

**[CoinTrading/scripts/setup_cron.sh](../scripts/setup_cron.sh)**
- Interactive cron job setup
- Prompts for:
  - Schedule (weekly/custom/test)
  - Mode (dry-run/paper/live)
- Adds job to crontab
- Shows verification steps

**[CoinTrading/scripts/test_bot.sh](../scripts/test_bot.sh)**
```bash
# Quick test commands:
bash test_bot.sh           # Dry run
bash test_bot.sh paper     # Paper trading
bash test_bot.sh live      # Live trading (careful!)
```

### Documentation

**[README.md](README.md)**
- Quick start guide
- Feature overview
- Basic usage examples
- Safety checklist

**[SETUP_GUIDE.md](SETUP_GUIDE.md)**
- Complete step-by-step setup
- Testing procedures
- Going live checklist
- Troubleshooting

**[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**
- This file
- Project architecture
- File purposes
- Data flow

## Architecture Overview

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Cron Job (every 7 days)                                         │
│   → Executes: python upbit_rebalancer.py                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ UpbitRebalancer (main bot)                                      │
│   1. Load config (upbit_config.yaml)                            │
│   2. Load secrets (upbit_secrets.yaml)                          │
│   3. Initialize UpbitClient                                     │
│   4. Initialize UpbitOrderManager                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Get Portfolio State                                             │
│   UpbitClient.get_balances()                                    │
│   → Returns: KRW + all coin balances                            │
│                                                                  │
│   UpbitClient.get_current_prices()                              │
│   → Returns: Current prices for all coins                       │
│                                                                  │
│   Calculate:                                                     │
│   - Total portfolio value (KRW)                                 │
│   - Current positions (value per coin)                          │
│   - Current weights (%)                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Calculate Rebalance Orders                                      │
│   Target: 25% USDT, 25% BTC, 25% ETH, 25% LTC                  │
│                                                                  │
│   For each coin:                                                │
│   - Current value vs Target value                               │
│   - Difference → BUY or SELL                                    │
│   - Skip if diff < min_order_krw (5,000 KRW)                   │
│                                                                  │
│   Sort: SELLs first (free up KRW), then BUYs                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Execute Orders                                                   │
│   If dry_run: Log only                                          │
│   If paper_trading: Simulate                                    │
│   If live: Execute via UpbitClient                              │
│                                                                  │
│   For each order:                                               │
│   - SELL: client.sell_market_order(ticker, volume)              │
│   - BUY: client.buy_market_order(ticker, krw_amount)            │
│                                                                  │
│   Track:                                                         │
│   - Successful trades                                            │
│   - Failed trades                                                │
│   - Total fees                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Log & Save Results                                               │
│   - Write to upbit_rebalancer.log                               │
│   - Append to rebalance_history.csv                             │
│   - Send notifications (if enabled)                             │
│   - Exit cleanly                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Relationships

```
upbit_rebalancer.py (main)
    │
    ├─→ UpbitClient (data access)
    │      ├─ get_balances()
    │      ├─ get_current_prices()
    │      ├─ buy_market_order()
    │      └─ sell_market_order()
    │
    └─→ UpbitOrderManager (rebalancing logic)
           ├─ get_portfolio_value()
           ├─ get_current_weights()
           ├─ calculate_rebalance_orders()
           └─ execute_rebalance_orders()
```

## Execution Modes

### 1. Dry Run
```bash
python upbit_rebalancer.py --dry-run
```
- Fetches real portfolio data
- Calculates real trades
- **Does NOT execute**
- Logs everything
- Safe for testing

### 2. Paper Trading
```bash
python upbit_rebalancer.py --paper-trading
```
- Fetches real portfolio data
- Calculates real trades
- **Simulates execution**
- Updates simulated positions
- Logs as if real

### 3. Live Trading
```bash
python upbit_rebalancer.py
# (with paper_trading: false in config)
```
- Fetches real portfolio data
- Calculates real trades
- **Executes on Upbit**
- Real money at risk
- All trades logged

## Configuration Priority

Settings are applied in this order (later overrides earlier):

1. **Default values** (hardcoded in Python)
2. **upbit_config.yaml** (file configuration)
3. **Command-line flags** (--dry-run, --paper-trading)

Example:
```yaml
# upbit_config.yaml
execution:
  paper_trading: false  # Config says live trading

# But command line overrides:
$ python upbit_rebalancer.py --dry-run
# → Result: dry run mode (CLI wins)
```

## Cron Job Format

```bash
# Crontab entry format:
# ┌───────────── minute (0-59)
# │ ┌───────────── hour (0-23)
# │ │ ┌───────────── day of month (1-31)
# │ │ │ ┌───────────── month (1-12)
# │ │ │ │ ┌───────────── day of week (0-6, Sunday=0)
# │ │ │ │ │
# * * * * * command

# Example: Every Sunday at 9:00 AM
0 9 * * 0 cd /path/to/SystemTrading && source setup.sh && python CoinTrading/bots/upbit_rebalancer.py >> logs/cron.log 2>&1
```

## Log Files

### upbit_rebalancer.log
```
2025-10-12 09:00:01 - INFO - Starting Upbit Rebalancer
2025-10-12 09:00:02 - INFO - Total portfolio: 1,234,567 KRW
2025-10-12 09:00:03 - INFO - Rebalancing: 3 orders
2025-10-12 09:00:05 - INFO - Completed successfully
```

**Rotation**: 10MB per file, keeps 5 backups

### rebalance_history.csv
```csv
timestamp,total_value_krw,num_orders,successful_orders,failed_orders,total_fees_krw,mode
2025-10-12 09:00:05,1234567,3,3,0,123.45,live
2025-10-19 09:00:07,1245123,2,2,0,87.32,live
```

## Security

### Secrets Management
- `upbit_secrets.yaml` → **NOT in git** (.gitignore)
- File permissions: `chmod 600 upbit_secrets.yaml`
- Never commit API keys
- Use read-only API if possible

### API Permissions
✓ View assets
✓ Trading
✗ **Withdrawals** (NEVER enable)

### Safe Testing
1. Start with `--dry-run`
2. Move to `--paper-trading`
3. Test cron with dry-run schedule
4. Only then enable live trading
5. Start with small capital

## Monitoring

### Daily Checks
```bash
tail -20 CoinTrading/logs/upbit_rebalancer.log
```

### Weekly Review
```bash
cat CoinTrading/logs/rebalance_history.csv | column -t -s,
```

### Real-time Monitoring
```bash
tail -f CoinTrading/logs/upbit_rebalancer.log
```

## Maintenance

### Update Allocation
Edit `CoinTrading/config/upbit_config.yaml`:
```yaml
strategy:
  target_weights:
    KRW-USDT: 0.30  # Changed
    KRW-BTC: 0.30
    KRW-ETH: 0.25
    KRW-LTC: 0.15   # Changed
```

### Change Schedule
```bash
crontab -e
# Edit the schedule line
```

### Pause Trading
```bash
crontab -e
# Comment out the line with #
```

## Dependencies

### Python Packages
- `pyupbit` - Upbit API wrapper
- `pyyaml` - Configuration parsing
- `pandas` - Data handling
- `logging` - Built-in

### External Requirements
- Upbit account with API keys
- Cron (for scheduling)
- Internet connection

## Version History

- **v1.0** (2025-10-12): Initial implementation
  - One-shot cron execution
  - 25/25/25/25 allocation
  - Dry-run/paper/live modes
  - Comprehensive logging

## Support

See documentation:
- [README.md](README.md) - Quick start
- [SETUP_GUIDE.md](SETUP_GUIDE.md) - Full setup
- Check logs for errors
- Test with dry-run first
