"""
Data Validation Module

Validates OHLCV data quality and filters tickers based on criteria.
"""

from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates market data quality and completeness.

    Features:
    - Check minimum history requirements
    - Detect missing dates
    - Filter out invalid/incomplete data
    - Validate data freshness
    """

    def __init__(self, min_history_days: int = 200):
        """
        Initialize validator.

        Args:
            min_history_days: Minimum required days of history
        """
        self.min_history_days = min_history_days
        logger.info(f"DataValidator initialized (min_history: {min_history_days} days)")

    def check_impossible_data(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Tuple[bool, List[str], Dict]:
        """
        Check for technically impossible data only.

        This focuses on data that violates mathematical/physical constraints:
        - NaN, zero, negative prices
        - OHLC violations (high < low, etc.)
        - Invalid data types

        Does NOT flag:
        - Large price changes (crypto is volatile)
        - Date gaps (delisting, data issues are acceptable)
        - Low/zero volume (acceptable)

        Args:
            df: DataFrame to validate
            symbol: Symbol name (for logging)

        Returns:
            (is_valid, critical_issues_list, stats_dict) tuple
        """
        critical_issues = []
        stats = {
            'total_rows': len(df),
            'bad_rows': 0,
            'nan_rows': 0,
            'zero_negative_price_rows': 0,
            'ohlc_violation_rows': 0
        }

        if df.empty:
            critical_issues.append("DataFrame is empty")
            return False, critical_issues, stats

        price_cols = ['open', 'high', 'low', 'close']

        # Check 1: NaN values in OHLC columns
        ohlc_cols = [col for col in price_cols if col in df.columns]
        if ohlc_cols:
            nan_mask = df[ohlc_cols].isnull().any(axis=1)
            nan_count = nan_mask.sum()
            if nan_count > 0:
                stats['nan_rows'] = nan_count
                critical_issues.append(
                    f"Contains {nan_count} rows with NaN in OHLC columns"
                )

        # Check 2: Zero or negative prices (technically impossible)
        for col in price_cols:
            if col in df.columns:
                invalid_mask = df[col] <= 0
                invalid_count = invalid_mask.sum()
                if invalid_count > 0:
                    stats['zero_negative_price_rows'] += invalid_count
                    critical_issues.append(
                        f"Contains {invalid_count} rows with zero/negative {col}"
                    )

        # Check 3: OHLC relationship violations
        if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            # high must be >= low
            high_low_violation = df['high'] < df['low']

            # high must be >= open and close
            high_open_violation = df['high'] < df['open']
            high_close_violation = df['high'] < df['close']

            # low must be <= open and close
            low_open_violation = df['low'] > df['open']
            low_close_violation = df['low'] > df['close']

            ohlc_violations = (
                high_low_violation |
                high_open_violation |
                high_close_violation |
                low_open_violation |
                low_close_violation
            )

            violation_count = ohlc_violations.sum()
            if violation_count > 0:
                stats['ohlc_violation_rows'] = violation_count
                critical_issues.append(
                    f"Contains {violation_count} rows with OHLC violations "
                    "(high < low, close outside [low, high], etc.)"
                )

        # Check 4: NaN in volume (negative volume is checked but zero is OK)
        if 'volume' in df.columns:
            nan_volume = df['volume'].isnull().sum()
            if nan_volume > 0:
                critical_issues.append(f"Contains {nan_volume} rows with NaN volume")

            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                critical_issues.append(f"Contains {negative_volume} rows with negative volume")

        # Calculate total bad rows
        stats['bad_rows'] = max(
            stats['nan_rows'],
            stats['zero_negative_price_rows'],
            stats['ohlc_violation_rows']
        )

        is_valid = len(critical_issues) == 0

        if not is_valid:
            logger.debug(f"{symbol}: Found {len(critical_issues)} critical issues")

        return is_valid, critical_issues, stats

    def validate_all_symbols(
        self,
        data_dict: Dict[str, pd.DataFrame],
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Validate all symbols and return a detailed report.

        Args:
            data_dict: Dict mapping symbol to DataFrame
            verbose: Whether to log progress

        Returns:
            DataFrame with columns: symbol, start_date, end_date, days,
                                   status, issues, bad_rows, total_rows
        """
        results = []

        for symbol, df in data_dict.items():
            is_valid, issues, stats = self.check_impossible_data(df, symbol)

            start_date = df.index[0].date() if not df.empty else None
            end_date = df.index[-1].date() if not df.empty else None

            results.append({
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'days': len(df),
                'status': 'valid' if is_valid else 'invalid',
                'issues': '; '.join(issues) if issues else '',
                'bad_rows': stats.get('bad_rows', 0),
                'total_rows': stats.get('total_rows', 0)
            })

            if verbose and not is_valid:
                logger.info(f"{symbol}: INVALID - {'; '.join(issues)}")

        report_df = pd.DataFrame(results)

        valid_count = (report_df['status'] == 'valid').sum()
        invalid_count = (report_df['status'] == 'invalid').sum()

        logger.info(
            f"Validation complete: {valid_count} valid, {invalid_count} invalid "
            f"out of {len(data_dict)} total symbols"
        )

        return report_df

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        check_today: bool = True,
        check_missing: bool = True
    ) -> Tuple[bool, List[str]]:
        """
        Validate a single DataFrame.

        Args:
            df: DataFrame to validate
            symbol: Symbol name (for logging)
            check_today: Whether to verify last date is today
            check_missing: Whether to check for missing dates

        Returns:
            (is_valid, list_of_issues) tuple
        """
        issues = []

        # Check if empty
        if df.empty:
            issues.append("DataFrame is empty")
            return False, issues

        # Check minimum length
        if len(df) < self.min_history_days:
            issues.append(
                f"Insufficient history: {len(df)} < {self.min_history_days} days"
            )

        # Check for NaN values
        if df.isnull().any().any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            issues.append(f"Contains NaN values in columns: {nan_cols}")

        # Check for zero/negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns and (df[col] <= 0).any():
                issues.append(f"Contains zero/negative values in {col}")

        # Check OHLC validity (high >= low, etc.)
        if 'high' in df.columns and 'low' in df.columns:
            if (df['high'] < df['low']).any():
                issues.append("Invalid OHLC: high < low")

        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            )
            if invalid_ohlc.any():
                issues.append("Invalid OHLC relationships detected")

        # Check data freshness
        if check_today:
            last_date = df.index[-1].date()
            today = datetime.today().date()
            yesterday = (datetime.today() - timedelta(days=1)).date()

            if last_date not in [today, yesterday]:
                issues.append(
                    f"Data not up-to-date: last={last_date}, expected={today}"
                )

        # Check for missing dates (simple version: check consecutive days)
        if check_missing and len(df) > 1:
            date_diffs = df.index.to_series().diff()[1:]
            # Assuming daily data, diff should be ~1 day
            expected_diff = timedelta(days=1)
            large_gaps = date_diffs[date_diffs > expected_diff * 2]

            if len(large_gaps) > 5:  # Allow a few gaps (weekends, etc.)
                issues.append(
                    f"Found {len(large_gaps)} date gaps larger than 2 days"
                )

        is_valid = len(issues) == 0

        if not is_valid:
            logger.debug(f"{symbol} validation failed: {'; '.join(issues)}")

        return is_valid, issues

    def filter_valid_symbols(
        self,
        data_dict: Dict[str, pd.DataFrame],
        check_today: bool = True,
        check_missing: bool = True,
        verbose: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Filter dictionary of DataFrames to keep only valid ones.

        Args:
            data_dict: Dict mapping symbol to DataFrame
            check_today: Whether to verify last date is today
            check_missing: Whether to check for missing dates
            verbose: Whether to log filtered symbols

        Returns:
            Filtered dict with only valid DataFrames
        """
        valid_data = {}
        invalid_count = 0

        for symbol, df in data_dict.items():
            is_valid, issues = self.validate_dataframe(
                df, symbol, check_today, check_missing
            )

            if is_valid:
                valid_data[symbol] = df
            else:
                invalid_count += 1
                if verbose:
                    logger.info(f"Filtered out {symbol}: {'; '.join(issues)}")

        logger.info(
            f"Validation complete: {len(valid_data)} valid, "
            f"{invalid_count} invalid (filtered out)"
        )

        return valid_data

    def get_symbols_with_min_history(
        self,
        data_dict: Dict[str, pd.DataFrame],
        min_days: Optional[int] = None
    ) -> List[str]:
        """
        Get symbols that have sufficient history.

        Args:
            data_dict: Dict mapping symbol to DataFrame
            min_days: Minimum days (uses self.min_history_days if None)

        Returns:
            List of valid symbol names
        """
        min_days = min_days or self.min_history_days
        valid_symbols = [
            symbol
            for symbol, df in data_dict.items()
            if len(df) >= min_days
        ]

        logger.info(
            f"Found {len(valid_symbols)} symbols with >={min_days} days of history"
        )

        return valid_symbols

    def check_data_freshness(
        self,
        data_dict: Dict[str, pd.DataFrame],
        max_age_hours: int = 24
    ) -> Dict[str, bool]:
        """
        Check which symbols have fresh data.

        Args:
            data_dict: Dict mapping symbol to DataFrame
            max_age_hours: Maximum acceptable data age in hours

        Returns:
            Dict mapping symbol to is_fresh (bool)
        """
        now = datetime.now()
        freshness = {}

        for symbol, df in data_dict.items():
            if df.empty:
                freshness[symbol] = False
                continue

            last_timestamp = df.index[-1]
            age_hours = (now - last_timestamp).total_seconds() / 3600

            freshness[symbol] = age_hours <= max_age_hours

        stale_count = sum(1 for is_fresh in freshness.values() if not is_fresh)
        logger.info(
            f"Freshness check: {len(freshness) - stale_count} fresh, "
            f"{stale_count} stale (>{max_age_hours}h old)"
        )

        return freshness

    def validate_column_names(
        self,
        df: pd.DataFrame,
        required_columns: List[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Validate that DataFrame has required columns.

        Args:
            df: DataFrame to check
            required_columns: List of required column names

        Returns:
            (is_valid, missing_columns) tuple
        """
        if required_columns is None:
            required_columns = ['open', 'high', 'low', 'close', 'volume']

        missing = [col for col in required_columns if col not in df.columns]

        is_valid = len(missing) == 0

        return is_valid, missing


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    dates = pd.date_range('2024-01-01', periods=250, freq='D')
    sample_df = pd.DataFrame({
        'open': 40000.0,
        'high': 41000.0,
        'low': 39000.0,
        'close': 40500.0,
        'volume': 1000.0
    }, index=dates)

    # Initialize validator
    validator = DataValidator(min_history_days=200)

    # Validate sample data
    is_valid, issues = validator.validate_dataframe(sample_df, 'BTCUSDT', check_today=False)
    print(f"Valid: {is_valid}")
    print(f"Issues: {issues}")

    # Test with invalid data
    invalid_df = sample_df.copy()
    invalid_df.loc[dates[100], 'close'] = None  # Add NaN
    is_valid, issues = validator.validate_dataframe(invalid_df, 'ETHUSDT', check_today=False)
    print(f"\nInvalid data - Valid: {is_valid}")
    print(f"Issues: {issues}")

    # Test batch validation
    data_dict = {
        'BTCUSDT': sample_df,
        'ETHUSDT': invalid_df,
        'BNBUSDT': sample_df[:100]  # Too short
    }

    valid_data = validator.filter_valid_symbols(data_dict, check_today=False)
    print(f"\nFiltered: {list(valid_data.keys())}")
