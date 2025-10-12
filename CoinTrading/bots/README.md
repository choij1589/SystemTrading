# Upbit Rebalancing Bot

Automated portfolio rebalancing bot for Upbit exchange (KRW markets).

## Features

- **Periodic rebalancing**: Maintains 25/25/25/25 allocation across USDT/BTC/ETH/LTC
- **Cron-based execution**: Runs via crontab every 7 days (no continuous monitoring)
- **Multiple modes**: Dry-run, paper trading, and live trading
- **Risk management**: Position limits and minimum order validation
- **Comprehensive logging**: All actions logged with rotation
- **Execution tracking**: CSV history of all rebalancing events

## Quick Start

### 1. Install Dependencies

```bash
# Already installed in systemtrading conda environment
pip install pyupbit pyyaml
```

### 2. Configure API Keys

```bash
# Create secrets file from template
cp CoinTrading/config/upbit_secrets.yaml.template CoinTrading/config/upbit_secrets.yaml

# Edit with your Upbit API keys
nano CoinTrading/config/upbit_secrets.yaml
```

**To get Upbit API keys:**
1. Login to [Upbit](https://upbit.com)
2. Go to Settings > API Management
3. Create new API key with permissions:
   - ✓ View assets
   - ✓ Trading (buy/sell)
4. Copy access key and secret key to `upbit_secrets.yaml`

### 3. Test in Dry-Run Mode

```bash
# Test without executing any trades
source setup.sh
python CoinTrading/bots/upbit_rebalancer.py --dry-run
```

This will:
- Connect to Upbit API
- Fetch your current portfolio
- Calculate rebalancing orders
- **Log everything without executing trades**

### 4. Test in Paper Trading Mode

```bash
# Simulate trades (no real execution)
python CoinTrading/bots/upbit_rebalancer.py --paper-trading
```

### 5. Setup Cron Job

```bash
# Interactive setup script
bash CoinTrading/scripts/setup_cron.sh
```

This will guide you through:
- Choosing rebalancing schedule (weekly/custom)
- Selecting execution mode (dry-run/paper/live)
- Adding cron job to crontab

## Configuration

### Portfolio Allocation

Edit [CoinTrading/config/upbit_config.yaml](../config/upbit_config.yaml):

```yaml
strategy:
  target_weights:
    KRW-USDT: 0.25    # 25% USDT
    KRW-BTC: 0.25     # 25% Bitcoin
    KRW-ETH: 0.25     # 25% Ethereum
    KRW-LTC: 0.25     # 25% Litecoin
```

**Note**: KRW-USDT is treated as a tradeable coin (not cash).

### Risk Limits

```yaml
risk:
  max_position_pct: 0.40   # Maximum 40% in one coin
  max_daily_loss_pct: 0.10 # Stop if -10% loss
```

### Execution Settings

```yaml
execution:
  paper_trading: true      # Set false for live trading
  dry_run: false           # Set true to log without executing
  min_order_krw: 5000      # Minimum order (Upbit minimum)
  transaction_fee: 0.0005  # 0.05% fee
```

## Manual Execution

### Dry Run (Safe)
```bash
python CoinTrading/bots/upbit_rebalancer.py --dry-run
```

### Paper Trading (Simulated)
```bash
python CoinTrading/bots/upbit_rebalancer.py --paper-trading
```

### Live Trading (Real Money!)
```bash
# Make sure paper_trading: false in config
python CoinTrading/bots/upbit_rebalancer.py
```

## Monitoring

### View Logs

```bash
# Real-time log monitoring
tail -f CoinTrading/logs/upbit_rebalancer.log

# Cron execution log
tail -f CoinTrading/logs/cron.log
```

### View Execution History

```bash
# CSV history of all rebalancing events
cat CoinTrading/logs/rebalance_history.csv
```

### Check Cron Status

```bash
# View crontab
crontab -l

# Edit crontab
crontab -e

# Remove crontab
crontab -r
```

## Example Output

```
================================================================================
UPBIT REBALANCING BOT
================================================================================
Started at: 2025-10-12 09:00:01
Config: CoinTrading/config/upbit_config.yaml
Secrets: CoinTrading/config/upbit_secrets.yaml
✓ Upbit client initialized
✓ Order manager initialized (Mode: PAPER TRADING)

--------------------------------------------------------------------------------
FETCHING PORTFOLIO STATE
--------------------------------------------------------------------------------

Total Portfolio Value: 1,234,567 KRW

Current Positions:
  KRW-BTC      400,000 KRW ( 32.4%)
  KRW-ETH      300,000 KRW ( 24.3%)
  KRW-LTC      200,000 KRW ( 16.2%)
  KRW-USDT     334,567 KRW ( 27.1%)

--------------------------------------------------------------------------------
CALCULATING REBALANCING ORDERS
--------------------------------------------------------------------------------

====================================================================================================
REBALANCING PLAN
====================================================================================================
Total Portfolio Value: 1,234,567 KRW
Number of trades: 3

Ticker       Action Current      Target       Diff (KRW)      Current%  Target%
----------------------------------------------------------------------------------------------------
KRW-BTC      SELL    400,000     308,642        -91,358       32.4%     25.0%
KRW-ETH      BUY     300,000     308,642         +8,642       24.3%     25.0%
KRW-LTC      BUY     200,000     308,642       +108,642       16.2%     25.0%
====================================================================================================

--------------------------------------------------------------------------------
EXECUTING REBALANCING ORDERS
--------------------------------------------------------------------------------
[PAPER] SELL KRW-BTC: 0.000538 units @ 169,829,000 KRW (~91,358 KRW)
[PAPER] BUY KRW-ETH: 8,642 KRW (~0.001491 units @ 5,797,000 KRW)
[PAPER] BUY KRW-LTC: 108,642 KRW (~1.668 units @ 65,123 KRW)

================================================================================
REBALANCING SUMMARY
================================================================================
Total orders: 3
Successful: 3
Failed: 0
Total fees: 104 KRW (0.008%)
Completed at: 2025-10-12 09:00:05
================================================================================
```

## Safety Checklist

Before enabling live trading:

- [ ] Test thoroughly with `--dry-run` mode
- [ ] Verify calculations with `--paper-trading` mode
- [ ] Check Upbit API key permissions (view + trade only)
- [ ] Verify portfolio allocation in config
- [ ] Set appropriate risk limits
- [ ] Test cron schedule (use test mode first)
- [ ] Monitor logs for at least one week
- [ ] Start with small capital
- [ ] Enable notifications (optional)

## Troubleshooting

### "API keys not configured"
- Make sure you created `upbit_secrets.yaml` from template
- Verify API keys are correct (no quotes, no spaces)

### "No data returned"
- Check internet connection
- Verify ticker symbols in config (must be KRW-XXX format)

### "Order too small"
- Increase portfolio value (minimum 20,000 KRW for 4 coins)
- Check `min_order_krw` in config (must be >= 5000)

### Cron job not running
- Check crontab: `crontab -l`
- Verify computer is on at scheduled time
- Check cron logs: `tail -f CoinTrading/logs/cron.log`

## Architecture

```
CoinTrading/bots/upbit_rebalancer.py
├── UpbitClient (data/upbit_client.py)
│   ├── get_balances()
│   ├── get_current_prices()
│   └── buy/sell_market_order()
│
└── UpbitOrderManager (execution/upbit_order_manager.py)
    ├── get_portfolio_value()
    ├── calculate_rebalance_orders()
    └── execute_rebalance_orders()
```

## Support

For issues or questions:
1. Check logs: `tail -f CoinTrading/logs/upbit_rebalancer.log`
2. Review configuration files
3. Test with `--dry-run` first
4. Verify API key permissions on Upbit

## Disclaimer

**USE AT YOUR OWN RISK**

This bot trades real money. Always:
- Start with small amounts
- Test thoroughly before live trading
- Monitor regularly
- Understand the strategy
- Never invest more than you can afford to lose

The authors are not responsible for any financial losses.
