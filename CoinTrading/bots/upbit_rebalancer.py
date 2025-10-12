#!/usr/bin/env python3
"""
Upbit Rebalancing Bot

One-shot execution bot for periodic portfolio rebalancing on Upbit exchange.
Designed to be run via crontab every 7 days.

Usage:
    python upbit_rebalancer.py [--dry-run] [--paper-trading]

Options:
    --dry-run: Log all actions without executing any trades
    --paper-trading: Simulate trades (no real execution)
"""

import os
import sys
import argparse
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
import yaml
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from CoinTrading.data.upbit_client import UpbitClient
from CoinTrading.execution.upbit_order_manager import UpbitOrderManager


class UpbitRebalancer:
    """
    One-shot rebalancing bot for Upbit exchange.
    """

    def __init__(self, config_path: str, secrets_path: str):
        """
        Initialize rebalancer.

        Args:
            config_path: Path to config YAML file
            secrets_path: Path to secrets YAML file
        """
        # Load configuration
        self.config = self._load_yaml(config_path)
        self.secrets = self._load_yaml(secrets_path)

        # Setup logging
        self._setup_logging()

        # Log startup
        logger.info("=" * 100)
        logger.info("UPBIT REBALANCING BOT")
        logger.info("=" * 100)
        logger.info(f"Started at: {datetime.now()}")
        logger.info(f"Config: {config_path}")
        logger.info(f"Secrets: {secrets_path}")

        # Initialize Upbit client
        self._init_client()

        # Initialize order manager
        self._init_order_manager()

        # Get strategy config
        self.target_weights = self.config['strategy']['target_weights']
        logger.info(f"Target allocation: {self.target_weights}")

    def _load_yaml(self, path: str) -> dict:
        """Load YAML configuration file."""
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"ERROR: Configuration file not found: {path}")
            if 'secrets' in path:
                print("Please create upbit_secrets.yaml from the template:")
                print("  cp CoinTrading/config/upbit_secrets.yaml.template CoinTrading/config/upbit_secrets.yaml")
                print("  # Then edit upbit_secrets.yaml with your API keys")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to load {path}: {e}")
            sys.exit(1)

    def _setup_logging(self):
        """Setup logging configuration."""
        global logger

        log_config = self.config['logging']
        log_level = getattr(logging, log_config['level'])

        # Create logs directory
        log_dir = Path(log_config['file']).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        logger = logging.getLogger('UpbitRebalancer')
        logger.setLevel(log_level)

        # Formatter
        formatter = logging.Formatter(log_config['format'])

        # File handler with rotation
        file_handler = RotatingFileHandler(
            log_config['file'],
            maxBytes=log_config.get('max_bytes', 10485760),
            backupCount=log_config.get('backup_count', 5)
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        if log_config.get('console', True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    def _init_client(self):
        """Initialize Upbit client."""
        upbit_config = self.secrets['upbit']
        access_key = upbit_config.get('access_key')
        secret_key = upbit_config.get('secret_key')

        if not access_key or not secret_key or 'YOUR_' in access_key:
            logger.error("API keys not configured in upbit_secrets.yaml")
            logger.error("Please update CoinTrading/config/upbit_secrets.yaml with your Upbit API keys")
            sys.exit(1)

        cache_dir = self.config['upbit'].get('cache_dir', '.cache/upbit')

        self.client = UpbitClient(
            access_key=access_key,
            secret_key=secret_key,
            cache_dir=cache_dir
        )

        logger.info("✓ Upbit client initialized")

    def _init_order_manager(self):
        """Initialize order manager."""
        exec_config = self.config['execution']

        self.paper_trading = exec_config.get('paper_trading', True)
        self.dry_run = exec_config.get('dry_run', False)

        self.order_manager = UpbitOrderManager(
            upbit_client=self.client,
            paper_trading=self.paper_trading,
            min_order_krw=exec_config.get('min_order_krw', 5000.0),
            transaction_fee=exec_config.get('transaction_fee', 0.0005)
        )

        mode = "DRY RUN" if self.dry_run else ("PAPER TRADING" if self.paper_trading else "LIVE TRADING")
        logger.info(f"✓ Order manager initialized (Mode: {mode})")

    def get_portfolio_state(self):
        """
        Get current portfolio state.

        Returns:
            Tuple of (total_value, positions, weights)
        """
        logger.info("\n" + "-" * 100)
        logger.info("FETCHING PORTFOLIO STATE")
        logger.info("-" * 100)

        total_value, positions = self.order_manager.get_portfolio_value()

        if total_value == 0:
            logger.error("Portfolio value is zero - cannot proceed")
            return None, None, None

        weights = self.order_manager.get_current_weights(total_value, positions)

        logger.info(f"\nTotal Portfolio Value: {total_value:,.0f} KRW")
        logger.info("\nCurrent Positions:")
        for ticker in sorted(positions.keys()):
            value = positions[ticker]
            weight = weights[ticker]
            logger.info(f"  {ticker:<12} {value:>12,.0f} KRW ({weight:>6.1%})")

        return total_value, positions, weights

    def save_execution_record(
        self,
        total_value: float,
        current_weights: dict,
        orders: list,
        successful: int,
        failed: int,
        total_fees: float
    ):
        """
        Save execution record to CSV.

        Args:
            total_value: Portfolio value
            current_weights: Current weights before rebalancing
            orders: List of orders
            successful: Number of successful trades
            failed: Number of failed trades
            total_fees: Total fees paid
        """
        tracking_config = self.config.get('tracking', {})
        if not tracking_config.get('save_history', True):
            return

        history_file = tracking_config.get('history_file', 'logs/rebalance_history.csv')

        # Create record
        record = {
            'timestamp': datetime.now(),
            'total_value_krw': total_value,
            'num_orders': len(orders),
            'successful_orders': successful,
            'failed_orders': failed,
            'total_fees_krw': total_fees,
            'mode': 'dry_run' if self.dry_run else ('paper' if self.paper_trading else 'live')
        }

        # Add current weights
        for ticker, weight in current_weights.items():
            record[f'weight_{ticker}'] = weight

        # Save to CSV
        try:
            df_record = pd.DataFrame([record])

            if os.path.exists(history_file):
                df_existing = pd.read_csv(history_file)
                df_combined = pd.concat([df_existing, df_record], ignore_index=True)
            else:
                df_combined = df_record

            df_combined.to_csv(history_file, index=False)
            logger.info(f"✓ Execution record saved to {history_file}")

        except Exception as e:
            logger.error(f"Failed to save execution record: {e}")

    def run(self):
        """
        Execute one-shot rebalancing.
        """
        try:
            # Get current portfolio state
            total_value, positions, current_weights = self.get_portfolio_state()

            if total_value is None:
                logger.error("Failed to get portfolio state")
                return 1

            # Calculate rebalancing orders
            logger.info("\n" + "-" * 100)
            logger.info("CALCULATING REBALANCING ORDERS")
            logger.info("-" * 100)

            orders = self.order_manager.calculate_rebalance_orders(
                self.target_weights,
                total_value,
                positions
            )

            # Print rebalancing plan
            self.order_manager.print_rebalance_plan(orders, total_value)

            if not orders:
                logger.info("\n✓ No rebalancing needed - portfolio already at target allocation")
                return 0

            # Execute orders
            logger.info("\n" + "-" * 100)
            logger.info("EXECUTING REBALANCING ORDERS")
            logger.info("-" * 100)

            successful, failed, total_fees = self.order_manager.execute_rebalance_orders(
                orders,
                dry_run=self.dry_run
            )

            # Print summary
            logger.info("\n" + "=" * 100)
            logger.info("REBALANCING SUMMARY")
            logger.info("=" * 100)
            logger.info(f"Total orders: {len(orders)}")
            logger.info(f"Successful: {successful}")
            logger.info(f"Failed: {failed}")
            logger.info(f"Total fees: {total_fees:,.0f} KRW ({total_fees/total_value*100:.3f}%)")
            logger.info(f"Completed at: {datetime.now()}")
            logger.info("=" * 100)

            # Save execution record
            self.save_execution_record(
                total_value,
                current_weights,
                orders,
                successful,
                failed,
                total_fees
            )

            # Send notifications (if enabled)
            self._send_notifications(
                total_value,
                orders,
                successful,
                failed,
                total_fees
            )

            # Return success if no failures
            return 0 if failed == 0 else 1

        except KeyboardInterrupt:
            logger.info("\n\nRebalancing interrupted by user")
            return 1

        except Exception as e:
            logger.error(f"\n\nUnexpected error: {e}", exc_info=True)
            return 1

    def _send_notifications(
        self,
        total_value: float,
        orders: list,
        successful: int,
        failed: int,
        total_fees: float
    ):
        """Send notifications (email/telegram) if enabled."""
        notif_config = self.config.get('notifications', {})
        if not notif_config.get('enabled', False):
            return

        # TODO: Implement email/telegram notifications
        logger.debug("Notifications not yet implemented")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Upbit Rebalancing Bot")
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Log trades without executing (overrides config)'
    )
    parser.add_argument(
        '--paper-trading',
        action='store_true',
        help='Paper trading mode (overrides config)'
    )
    parser.add_argument(
        '--config',
        default='CoinTrading/config/upbit_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--secrets',
        default='CoinTrading/config/upbit_secrets.yaml',
        help='Path to secrets file'
    )

    args = parser.parse_args()

    # Initialize rebalancer
    rebalancer = UpbitRebalancer(args.config, args.secrets)

    # Override config with CLI args
    if args.dry_run:
        rebalancer.dry_run = True
        logger.info("CLI override: dry_run enabled")

    if args.paper_trading:
        rebalancer.paper_trading = True
        logger.info("CLI override: paper_trading enabled")

    # Run rebalancing
    exit_code = rebalancer.run()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
