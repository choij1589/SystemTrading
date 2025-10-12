#!/bin/bash
#
# Quick Test Script for Upbit Rebalancer
#
# Usage:
#   bash test_bot.sh           # Dry run test
#   bash test_bot.sh paper     # Paper trading test
#   bash test_bot.sh live      # Live trading test (careful!)
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================================="
echo "Upbit Rebalancer - Quick Test"
echo "======================================================================="
echo

# Check secrets file
if [ ! -f "CoinTrading/config/upbit_secrets.yaml" ]; then
    echo -e "${RED}ERROR:${NC} upbit_secrets.yaml not found!"
    echo
    echo "Create it from template:"
    echo "  cp CoinTrading/config/upbit_secrets.yaml.template \\"
    echo "     CoinTrading/config/upbit_secrets.yaml"
    echo
    echo "Then add your Upbit API keys to the file."
    exit 1
fi

# Determine mode
MODE="${1:-dry-run}"

case "$MODE" in
    dry-run|dry|d)
        FLAGS="--dry-run"
        MODE_DESC="DRY RUN (no execution)"
        COLOR=$GREEN
        ;;
    paper|p)
        FLAGS="--paper-trading"
        MODE_DESC="PAPER TRADING (simulated)"
        COLOR=$YELLOW
        ;;
    live|l)
        FLAGS=""
        MODE_DESC="LIVE TRADING (REAL MONEY!)"
        COLOR=$RED
        echo -e "${RED}WARNING: This will execute REAL trades!${NC}"
        read -p "Are you sure? Type 'YES' to continue: " confirm
        if [ "$confirm" != "YES" ]; then
            echo "Aborted."
            exit 0
        fi
        ;;
    *)
        echo "Usage: $0 [dry-run|paper|live]"
        echo
        echo "Modes:"
        echo "  dry-run (default) - Log trades without executing"
        echo "  paper             - Simulate trade execution"
        echo "  live              - Execute real trades (CAUTION!)"
        exit 1
        ;;
esac

echo -e "${COLOR}Mode: $MODE_DESC${NC}"
echo "======================================================================="
echo

# Activate environment and run
echo "Running bot..."
echo

source setup.sh
python CoinTrading/bots/upbit_rebalancer.py $FLAGS

EXIT_CODE=$?

echo
echo "======================================================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Test completed successfully${NC}"
else
    echo -e "${RED}✗ Test failed with exit code $EXIT_CODE${NC}"
fi
echo "======================================================================="
echo
echo "Next steps:"
echo "  - Review logs: tail -f CoinTrading/logs/upbit_rebalancer.log"
echo "  - Check history: cat CoinTrading/logs/rebalance_history.csv"
echo "  - Setup cron: bash CoinTrading/scripts/setup_cron.sh"
echo

exit $EXIT_CODE
