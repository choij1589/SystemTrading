#!/bin/bash
#
# Crontab Setup Script for Upbit Rebalancer
#
# This script helps you set up a cron job to run the rebalancer automatically.
#

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "======================================================================="
echo "Upbit Rebalancer - Crontab Setup"
echo "======================================================================="
echo

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo -e "${GREEN}Project root:${NC} $PROJECT_ROOT"
echo

# Check if upbit_secrets.yaml exists
SECRETS_FILE="$PROJECT_ROOT/CoinTrading/config/upbit_secrets.yaml"
if [ ! -f "$SECRETS_FILE" ]; then
    echo -e "${RED}ERROR:${NC} upbit_secrets.yaml not found!"
    echo "Please create it from the template:"
    echo "  cp $PROJECT_ROOT/CoinTrading/config/upbit_secrets.yaml.template $SECRETS_FILE"
    echo "  # Then edit $SECRETS_FILE with your API keys"
    exit 1
fi

echo -e "${GREEN}✓${NC} Found upbit_secrets.yaml"
echo

# Ask for cron schedule
echo "Choose rebalancing schedule:"
echo "  1) Every Sunday at 9:00 AM (recommended)"
echo "  2) Every 7 days at 9:00 AM"
echo "  3) Custom schedule"
echo "  4) Test mode (every 30 minutes)"
echo

read -p "Enter choice (1-4): " schedule_choice

case $schedule_choice in
    1)
        CRON_SCHEDULE="0 9 * * 0"
        DESCRIPTION="Every Sunday at 9:00 AM"
        ;;
    2)
        CRON_SCHEDULE="0 9 */7 * *"
        DESCRIPTION="Every 7 days at 9:00 AM"
        ;;
    3)
        echo
        echo "Enter custom cron schedule (e.g., '0 9 * * 0' for Sunday 9 AM)"
        echo "Format: minute hour day month weekday"
        read -p "Schedule: " CRON_SCHEDULE
        DESCRIPTION="Custom: $CRON_SCHEDULE"
        ;;
    4)
        CRON_SCHEDULE="*/30 * * * *"
        DESCRIPTION="Every 30 minutes (TEST MODE)"
        echo -e "${YELLOW}WARNING: This is for testing only!${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo
echo -e "${GREEN}Schedule:${NC} $DESCRIPTION"
echo -e "${GREEN}Cron:${NC} $CRON_SCHEDULE"
echo

# Ask for execution mode
echo "Choose execution mode:"
echo "  1) Dry run (log only, no execution)"
echo "  2) Paper trading (simulate trades)"
echo "  3) Live trading (REAL MONEY - use with caution!)"
echo

read -p "Enter choice (1-3): " mode_choice

case $mode_choice in
    1)
        MODE_FLAGS="--dry-run"
        MODE_DESC="DRY RUN"
        ;;
    2)
        MODE_FLAGS="--paper-trading"
        MODE_DESC="PAPER TRADING"
        ;;
    3)
        MODE_FLAGS=""
        MODE_DESC="LIVE TRADING"
        echo -e "${RED}WARNING: This will execute REAL trades with REAL money!${NC}"
        read -p "Are you sure? Type 'YES' to confirm: " confirm
        if [ "$confirm" != "YES" ]; then
            echo "Aborted."
            exit 1
        fi
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo
echo -e "${GREEN}Mode:${NC} $MODE_DESC"
echo

# Build cron command
LOG_FILE="$PROJECT_ROOT/CoinTrading/logs/cron.log"
CRON_COMMAND="cd $PROJECT_ROOT && source setup.sh && python CoinTrading/bots/upbit_rebalancer.py $MODE_FLAGS >> $LOG_FILE 2>&1"

# Create the cron entry
CRON_ENTRY="$CRON_SCHEDULE $CRON_COMMAND"

# Show summary
echo "======================================================================="
echo "Cron Job Summary"
echo "======================================================================="
echo -e "${GREEN}Schedule:${NC} $DESCRIPTION"
echo -e "${GREEN}Mode:${NC} $MODE_DESC"
echo -e "${GREEN}Log file:${NC} $LOG_FILE"
echo
echo "Full cron entry:"
echo "$CRON_ENTRY"
echo "======================================================================="
echo

# Ask for confirmation
read -p "Add this cron job? (y/n): " confirm

if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Aborted."
    exit 0
fi

# Backup current crontab
echo
echo "Backing up current crontab..."
crontab -l > /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).txt 2>/dev/null || true

# Add cron job
echo "Adding cron job..."
(crontab -l 2>/dev/null || true; echo ""; echo "# Upbit Rebalancer - $MODE_DESC - $DESCRIPTION"; echo "$CRON_ENTRY") | crontab -

echo -e "${GREEN}✓ Cron job added successfully!${NC}"
echo

# Show current crontab
echo "Current crontab:"
echo "-----------------------------------------------------------------------"
crontab -l | grep -A1 "Upbit Rebalancer" || crontab -l | tail -5
echo "-----------------------------------------------------------------------"
echo

# Instructions
echo "======================================================================="
echo "Setup Complete!"
echo "======================================================================="
echo
echo "The rebalancer will run automatically according to the schedule."
echo
echo "Useful commands:"
echo "  - View crontab:     crontab -l"
echo "  - Edit crontab:     crontab -e"
echo "  - Remove crontab:   crontab -r"
echo "  - View logs:        tail -f $LOG_FILE"
echo "  - Manual run:       cd $PROJECT_ROOT && source setup.sh && python CoinTrading/bots/upbit_rebalancer.py $MODE_FLAGS"
echo
echo -e "${YELLOW}IMPORTANT:${NC}"
echo "  - Make sure your computer is running at the scheduled time"
echo "  - Check logs regularly: tail -f $LOG_FILE"
echo "  - Test with dry-run mode first before using live trading"
echo
echo "======================================================================="
