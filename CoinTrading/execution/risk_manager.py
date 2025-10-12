"""
Risk Manager

Position limits and safety checks before order execution.
"""

from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """
    Risk management limits.

    Attributes:
        max_leverage_per_coin: Maximum leverage for a single coin
        max_portfolio_leverage: Maximum total portfolio leverage
        max_position_pct: Maximum percentage of portfolio in one coin
        max_daily_loss_pct: Maximum daily loss before circuit breaker
        max_drawdown_pct: Maximum drawdown before position reduction
        min_notional: Minimum order size (USDT)
    """
    max_leverage_per_coin: float = 2.0
    max_portfolio_leverage: float = 3.0
    max_position_pct: float = 0.30
    max_daily_loss_pct: float = 0.10
    max_drawdown_pct: float = 0.30
    min_notional: float = 10.0


class RiskManager:
    """
    Risk management for live trading.

    Validates trades before execution to prevent excessive risk.
    """

    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        initial_capital: float = 10000.0
    ):
        """
        Initialize risk manager.

        Args:
            limits: Risk limits (default: RiskLimits())
            initial_capital: Initial capital for tracking
        """
        self.limits = limits or RiskLimits()
        self.initial_capital = initial_capital

        # Tracking
        self.daily_pnl: float = 0.0
        self.peak_equity: float = initial_capital
        self.daily_reset_date: Optional[pd.Timestamp] = None

        # Circuit breaker
        self.circuit_breaker_active: bool = False

        logger.info(f"Initialized RiskManager: {self.limits}")

    def update_daily_pnl(
        self,
        current_equity: float,
        current_date: pd.Timestamp
    ) -> None:
        """
        Update daily P&L tracking.

        Args:
            current_equity: Current portfolio value
            current_date: Current date
        """
        # Reset daily P&L at start of new day
        if self.daily_reset_date is None or current_date.date() != self.daily_reset_date.date():
            self.daily_pnl = 0.0
            self.daily_reset_date = current_date
            logger.info(f"Reset daily P&L for {current_date.date()}")

        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

    def check_daily_loss_limit(
        self,
        current_equity: float,
        start_of_day_equity: float
    ) -> Tuple[bool, str]:
        """
        Check if daily loss limit is exceeded.

        Args:
            current_equity: Current portfolio value
            start_of_day_equity: Portfolio value at start of day

        Returns:
            Tuple of (is_valid, reason)
        """
        if start_of_day_equity <= 0:
            return True, ""

        daily_loss_pct = (current_equity - start_of_day_equity) / start_of_day_equity

        if daily_loss_pct < -self.limits.max_daily_loss_pct:
            self.circuit_breaker_active = True
            return False, f"Daily loss limit exceeded: {daily_loss_pct*100:.2f}% < {-self.limits.max_daily_loss_pct*100:.2f}%"

        return True, ""

    def check_drawdown_limit(
        self,
        current_equity: float
    ) -> Tuple[bool, str]:
        """
        Check if drawdown limit is exceeded.

        Args:
            current_equity: Current portfolio value

        Returns:
            Tuple of (is_valid, reason)
        """
        if self.peak_equity <= 0:
            return True, ""

        drawdown = (self.peak_equity - current_equity) / self.peak_equity

        if drawdown > self.limits.max_drawdown_pct:
            return False, f"Drawdown limit exceeded: {drawdown*100:.2f}% > {self.limits.max_drawdown_pct*100:.2f}%"

        return True, ""

    def validate_position_weights(
        self,
        target_weights: Dict[str, float],
        current_equity: float
    ) -> Tuple[bool, str, Dict[str, float]]:
        """
        Validate and adjust position weights.

        Args:
            target_weights: Dict of {symbol: weight}
            current_equity: Current portfolio value

        Returns:
            Tuple of (is_valid, reason, adjusted_weights)
        """
        adjusted_weights = {}
        warnings = []

        for symbol, weight in target_weights.items():
            # Check individual coin leverage
            if abs(weight) > self.limits.max_leverage_per_coin:
                warnings.append(
                    f"{symbol}: leverage {abs(weight):.2f} > max {self.limits.max_leverage_per_coin:.2f}, clamping"
                )
                weight = max(-self.limits.max_leverage_per_coin, min(self.limits.max_leverage_per_coin, weight))

            # Check position size
            position_pct = abs(weight) / sum(abs(w) for w in target_weights.values()) if target_weights else 0
            if position_pct > self.limits.max_position_pct:
                warnings.append(
                    f"{symbol}: position {position_pct*100:.1f}% > max {self.limits.max_position_pct*100:.1f}%"
                )

            # Check minimum notional
            notional = abs(weight) * current_equity
            if 0 < notional < self.limits.min_notional:
                warnings.append(
                    f"{symbol}: notional ${notional:.2f} < min ${self.limits.min_notional:.2f}, skipping"
                )
                continue

            adjusted_weights[symbol] = weight

        # Check total portfolio leverage
        total_leverage = sum(abs(w) for w in adjusted_weights.values())
        if total_leverage > self.limits.max_portfolio_leverage:
            scale_factor = self.limits.max_portfolio_leverage / total_leverage
            warnings.append(
                f"Total leverage {total_leverage:.2f} > max {self.limits.max_portfolio_leverage:.2f}, scaling by {scale_factor:.2f}"
            )
            adjusted_weights = {
                symbol: weight * scale_factor
                for symbol, weight in adjusted_weights.items()
            }

        # Log warnings
        if warnings:
            for warning in warnings:
                logger.warning(warning)

        return True, "; ".join(warnings), adjusted_weights

    def validate_trade(
        self,
        symbol: str,
        target_weight: float,
        current_weight: float,
        current_equity: float,
        current_price: float
    ) -> Tuple[bool, str]:
        """
        Validate a single trade before execution.

        Args:
            symbol: Trading symbol
            target_weight: Target position weight
            current_weight: Current position weight
            current_equity: Current portfolio value
            current_price: Current asset price

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check circuit breaker
        if self.circuit_breaker_active:
            return False, "Circuit breaker active - no new trades allowed"

        # Calculate trade size
        weight_change = target_weight - current_weight
        notional = abs(weight_change) * current_equity

        # Check minimum notional
        if 0 < notional < self.limits.min_notional:
            return False, f"Trade size ${notional:.2f} below minimum ${self.limits.min_notional:.2f}"

        # Check leverage
        if abs(target_weight) > self.limits.max_leverage_per_coin:
            return False, f"Target leverage {abs(target_weight):.2f} exceeds max {self.limits.max_leverage_per_coin:.2f}"

        return True, ""

    def reset_circuit_breaker(self) -> None:
        """Reset circuit breaker (manual intervention)."""
        self.circuit_breaker_active = False
        logger.info("Circuit breaker manually reset")

    def get_status(self) -> Dict:
        """
        Get current risk status.

        Returns:
            Dict with risk metrics
        """
        return {
            'circuit_breaker_active': self.circuit_breaker_active,
            'peak_equity': self.peak_equity,
            'daily_pnl': self.daily_pnl,
            'limits': {
                'max_leverage_per_coin': self.limits.max_leverage_per_coin,
                'max_portfolio_leverage': self.limits.max_portfolio_leverage,
                'max_position_pct': self.limits.max_position_pct,
                'max_daily_loss_pct': self.limits.max_daily_loss_pct,
                'max_drawdown_pct': self.limits.max_drawdown_pct,
                'min_notional': self.limits.min_notional,
            }
        }


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Risk Manager example")
    print()

    # Create risk manager
    limits = RiskLimits(
        max_leverage_per_coin=2.0,
        max_portfolio_leverage=3.0,
        max_position_pct=0.30,
        max_daily_loss_pct=0.10,
        max_drawdown_pct=0.30,
        min_notional=10.0
    )

    risk_manager = RiskManager(limits=limits, initial_capital=10000.0)

    # Example: Validate position weights
    target_weights = {
        'BTCUSDT': 1.5,
        'ETHUSDT': 1.2,
        'BNBUSDT': 0.8,
    }

    is_valid, reason, adjusted = risk_manager.validate_position_weights(
        target_weights,
        current_equity=10000.0
    )

    print(f"Original weights: {target_weights}")
    print(f"Adjusted weights: {adjusted}")
    print(f"Valid: {is_valid}, Reason: {reason}")
    print()

    print("✓ Risk manager ready!")
