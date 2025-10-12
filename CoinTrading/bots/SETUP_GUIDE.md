# Upbit Rebalancing Bot - Complete Setup Guide

Step-by-step guide to set up and run the Upbit rebalancing bot.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Setup](#initial-setup)
3. [API Key Configuration](#api-key-configuration)
4. [Testing Phase](#testing-phase)
5. [Cron Job Setup](#cron-job-setup)
6. [Going Live](#going-live)
7. [Monitoring](#monitoring)
8. [Maintenance](#maintenance)

---

## Prerequisites

### System Requirements
- macOS, Linux, or Windows (with WSL)
- Python 3.8+
- Internet connection
- Upbit account with KRW balance

### Conda Environment
```bash
# Already set up if you're using the systemtrading environment
conda activate systemtrading

# Verify pyupbit is installed
python -c "import pyupbit; print('pyupbit installed')"
```

---

## Initial Setup

### Step 1: Navigate to Project Directory

```bash
cd /Users/choij/workspace/SystemTrading
```

### Step 2: Verify File Structure

```bash
# Check that all files are present
ls -l CoinTrading/bots/upbit_rebalancer.py
ls -l CoinTrading/config/upbit_config.yaml
ls -l CoinTrading/config/upbit_secrets.yaml.template
ls -l CoinTrading/scripts/setup_cron.sh
```

You should see all these files exist.

### Step 3: Create Logs Directory

```bash
mkdir -p CoinTrading/logs
```

---

## API Key Configuration

### Step 1: Get Upbit API Keys

1. Login to [Upbit](https://upbit.com)
2. Click your profile → **Settings**
3. Navigate to **API Management** (API 관리)
4. Click **Create API Key** (API 키 생성)
5. Configure permissions:
   ```
   ✓ View assets (자산 조회)
   ✓ Trading (거래)
   ✗ Withdrawals (출금) - DO NOT enable this!
   ```
6. Set IP whitelist (recommended for security)
7. Copy the **Access Key** and **Secret Key**

**⚠️ IMPORTANT**:
- Never share your API keys
- Never enable withdrawal permissions
- Store keys securely

### Step 2: Create Secrets File

```bash
# Copy template to secrets file
cp CoinTrading/config/upbit_secrets.yaml.template \
   CoinTrading/config/upbit_secrets.yaml
```

### Step 3: Edit Secrets File

```bash
# Edit with your preferred editor
nano CoinTrading/config/upbit_secrets.yaml
```

Replace placeholders with your actual keys:

```yaml
upbit:
  access_key: "YOUR_ACTUAL_ACCESS_KEY_HERE"
  secret_key: "YOUR_ACTUAL_SECRET_KEY_HERE"
```

Save and exit (`Ctrl+O`, `Enter`, `Ctrl+X` in nano).

### Step 4: Verify Secrets File

```bash
# Check file exists and is not tracked by git
ls -l CoinTrading/config/upbit_secrets.yaml
git status | grep upbit_secrets.yaml
# Should NOT appear in git status (it's in .gitignore)
```

---

## Testing Phase

### Test 1: Public Data Access

Test basic connectivity without authentication:

```bash
source setup.sh
python CoinTrading/data/upbit_client.py
```

Expected output:
```
Found 215 KRW markets
Current prices:
  KRW-BTC: 169,829,000 KRW
  KRW-ETH: 5,797,000 KRW
  ...
✓ UpbitClient ready!
```

### Test 2: Dry Run (No Execution)

Test the complete bot workflow without any real trades:

```bash
source setup.sh
python CoinTrading/bots/upbit_rebalancer.py --dry-run
```

This will:
- ✓ Connect to Upbit with your API keys
- ✓ Fetch your actual portfolio balances
- ✓ Calculate rebalancing orders
- ✓ Display the rebalancing plan
- ✗ **NOT execute any trades**

**Review the output carefully**:
- Check if balances are correct
- Verify target allocation (25/25/25/25)
- Review calculated trades
- Check for any errors

### Test 3: Paper Trading (Simulated Execution)

Simulate trade execution (still no real trades):

```bash
python CoinTrading/bots/upbit_rebalancer.py --paper-trading
```

This will:
- ✓ Calculate orders
- ✓ Simulate order execution
- ✓ Log trade details
- ✗ **NOT send orders to Upbit**

**Check the logs**:
```bash
tail -50 CoinTrading/logs/upbit_rebalancer.log
```

### Test 4: Verify Configuration

Review your configuration:

```bash
cat CoinTrading/config/upbit_config.yaml
```

**Key settings to verify**:
```yaml
strategy:
  target_weights:
    KRW-USDT: 0.25
    KRW-BTC: 0.25
    KRW-ETH: 0.25
    KRW-LTC: 0.25

execution:
  paper_trading: true      # Should be true for now
  dry_run: false
  min_order_krw: 5000

risk:
  max_position_pct: 0.40
```

---

## Cron Job Setup

### Step 1: Choose Schedule

Recommended schedules:
- **Weekly**: Every Sunday at 9:00 AM
- **Bi-weekly**: Every other Sunday
- **Test mode**: Every 30 minutes (for testing only)

### Step 2: Run Setup Script

```bash
bash CoinTrading/scripts/setup_cron.sh
```

Follow the interactive prompts:

```
Choose rebalancing schedule:
  1) Every Sunday at 9:00 AM (recommended)
  2) Every 7 days at 9:00 AM
  3) Custom schedule
  4) Test mode (every 30 minutes)

Enter choice (1-4): 1

Choose execution mode:
  1) Dry run (log only, no execution)
  2) Paper trading (simulate trades)
  3) Live trading (REAL MONEY - use with caution!)

Enter choice (1-3): 1
```

**For initial setup, choose**:
- Schedule: Option 1 (Every Sunday 9 AM)
- Mode: Option 1 (Dry run)

### Step 3: Verify Cron Job

```bash
# Check crontab
crontab -l

# Should see something like:
# Upbit Rebalancer - DRY RUN - Every Sunday at 9:00 AM
# 0 9 * * 0 cd /Users/choij/workspace/SystemTrading && source setup.sh && ...
```

### Step 4: Test Cron Execution

For testing, you can temporarily change the schedule to run every minute:

```bash
# Edit crontab
crontab -e

# Change schedule line to:
# */1 * * * * cd /Users/choij/workspace/SystemTrading && ...
```

Wait 1-2 minutes, then check logs:

```bash
tail -f CoinTrading/logs/cron.log
```

**After testing, change back to weekly schedule!**

---

## Going Live

### Before Enabling Live Trading

**Complete this checklist**:

- [ ] Tested with `--dry-run` successfully
- [ ] Tested with `--paper-trading` successfully
- [ ] Verified portfolio calculations are correct
- [ ] Reviewed at least 3 dry-run executions via cron
- [ ] Checked all log files for errors
- [ ] Verified API key permissions (no withdrawal)
- [ ] Understand the 25/25/25/25 allocation strategy
- [ ] Comfortable with transaction costs (~0.05% per trade)
- [ ] Portfolio value > 100,000 KRW (recommended minimum)

### Enable Paper Trading via Cron

First, switch cron job to paper trading mode:

```bash
bash CoinTrading/scripts/setup_cron.sh
```

Choose:
- Schedule: Weekly (Sunday 9 AM)
- Mode: **Paper trading**

Run for 1-2 weeks and monitor:

```bash
# Check logs regularly
tail -f CoinTrading/logs/upbit_rebalancer.log

# Review execution history
cat CoinTrading/logs/rebalance_history.csv
```

### Enable Live Trading

**⚠️ FINAL WARNING**: This will trade real money!

#### Step 1: Update Configuration

Edit config file:

```bash
nano CoinTrading/config/upbit_config.yaml
```

Change:
```yaml
execution:
  paper_trading: false    # Change from true to false
  dry_run: false
```

Save and exit.

#### Step 2: Update Cron Job

```bash
bash CoinTrading/scripts/setup_cron.sh
```

Choose:
- Schedule: Weekly (Sunday 9 AM)
- Mode: **Live trading**
- Confirm by typing "YES"

#### Step 3: Verify Live Setup

```bash
# Check crontab
crontab -l

# Should NOT see --dry-run or --paper-trading flags

# Check config
grep "paper_trading" CoinTrading/config/upbit_config.yaml
# Should show: paper_trading: false
```

#### Step 4: Monitor First Live Execution

On the first scheduled run:

```bash
# Watch logs in real-time
tail -f CoinTrading/logs/upbit_rebalancer.log

# Check Upbit app/website
# Verify trades were executed
# Check balances match expected allocation
```

---

## Monitoring

### Daily Checks

```bash
# Quick status check
tail -20 CoinTrading/logs/upbit_rebalancer.log
```

### Weekly Review

```bash
# Review all executions
cat CoinTrading/logs/rebalance_history.csv

# Check for errors
grep ERROR CoinTrading/logs/upbit_rebalancer.log

# Verify cron is running
crontab -l
```

### Log Files

- `upbit_rebalancer.log` - Main bot log (rotated at 10MB)
- `cron.log` - Cron execution output
- `rebalance_history.csv` - Execution history with timestamps

### Performance Tracking

Check execution history:

```bash
# View CSV in terminal
column -t -s, CoinTrading/logs/rebalance_history.csv | less -S

# Or open in Excel/Numbers
open CoinTrading/logs/rebalance_history.csv
```

---

## Maintenance

### Update Target Allocation

To change portfolio allocation:

```bash
nano CoinTrading/config/upbit_config.yaml
```

Edit target weights (must sum to 1.0):

```yaml
strategy:
  target_weights:
    KRW-USDT: 0.30    # Change allocations
    KRW-BTC: 0.30
    KRW-ETH: 0.25
    KRW-LTC: 0.15
```

### Change Rebalancing Schedule

```bash
# Re-run setup script
bash CoinTrading/scripts/setup_cron.sh

# Or manually edit crontab
crontab -e
```

### Pause Rebalancing

Temporarily disable:

```bash
# Comment out cron job
crontab -e
# Add # at start of line

# Or remove crontab entirely
crontab -r
```

### Resume Rebalancing

```bash
# Re-run setup script
bash CoinTrading/scripts/setup_cron.sh
```

### Clean Old Logs

```bash
# Keep last 30 days only
find CoinTrading/logs -name "*.log.*" -mtime +30 -delete
```

---

## Troubleshooting

### Common Issues

**Issue**: "API keys not configured"
```bash
# Solution: Check secrets file
cat CoinTrading/config/upbit_secrets.yaml
# Make sure keys don't have quotes or spaces
```

**Issue**: "Order too small"
```bash
# Solution: Portfolio may be too small
# Minimum: ~20,000 KRW for 4 coins at 5,000 KRW minimum each
```

**Issue**: "Module not found"
```bash
# Solution: Always use setup.sh
source setup.sh
python CoinTrading/bots/upbit_rebalancer.py
```

**Issue**: Cron not running
```bash
# Check cron logs
tail -f CoinTrading/logs/cron.log

# Verify crontab
crontab -l

# Test manual execution
source setup.sh && python CoinTrading/bots/upbit_rebalancer.py --dry-run
```

### Get Help

1. Check logs first
2. Review configuration
3. Test with `--dry-run`
4. Check Upbit API status

---

## Quick Reference

### Common Commands

```bash
# Manual dry run
python CoinTrading/bots/upbit_rebalancer.py --dry-run

# Manual paper trading
python CoinTrading/bots/upbit_rebalancer.py --paper-trading

# Manual live trading
python CoinTrading/bots/upbit_rebalancer.py

# View logs
tail -f CoinTrading/logs/upbit_rebalancer.log

# View cron
crontab -l

# Edit cron
crontab -e

# Setup cron
bash CoinTrading/scripts/setup_cron.sh
```

### File Locations

```
CoinTrading/
├── bots/
│   └── upbit_rebalancer.py        # Main bot script
├── config/
│   ├── upbit_config.yaml          # Configuration
│   └── upbit_secrets.yaml         # API keys (not in git)
├── logs/
│   ├── upbit_rebalancer.log       # Main log
│   ├── cron.log                   # Cron output
│   └── rebalance_history.csv      # Execution history
└── scripts/
    └── setup_cron.sh              # Cron setup helper
```

---

## Safety Reminders

✓ **Always test first** with dry-run mode
✓ **Monitor regularly** especially in first month
✓ **Start small** and increase capital gradually
✓ **Never enable withdrawals** on API keys
✓ **Review logs** before and after each rebalance
✓ **Understand the strategy** before going live
✓ **Set risk limits** appropriately

**Questions before going live?** Review this guide again!

Good luck! 🚀
