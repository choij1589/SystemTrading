"""
ETF Backtesting Engine

백테스팅 엔진 주요 기능:
1. 월별 정기 입금 (Dollar Cost Averaging)
2. 자동 리밸런싱 (월간/주간)
3. 현실적인 거래 비용 (수수료 + 세금 + bid-ask spread)
4. 정수 주식 제약
5. 비례적 현금 배분
"""

import os
import yaml
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..strategy.base import BaseETFStrategy


class ETFBacktestEngine:
    """
    ETF 백테스팅 엔진

    주요 기능:
    - 월별 정기 입금 (Dollar Cost Averaging)
    - 주기적 리밸런싱 (월간/주간)
    - 현실적인 거래 비용 (수수료 + 세금 + bid-ask spread)
    - 정수 주식 제약
    - 비례적 현금 배분
    """

    def __init__(
        self,
        initial_capital: float = 4_200_000,
        monthly_deposit: float = 300_000,
        commission_rate: float = 0.00015,
        tax_rate: float = 0.0023,
        bid_ask_spread: float = 0.0001,  # 기존 slippage를 bid-ask spread로 명확화
        rebalance_frequency: str = "monthly",
        deposit_day: int = 1,
        config_path: Optional[str] = None
    ):
        """
        백테스팅 엔진 초기화

        Args:
            initial_capital: 초기 투자금 (원)
            monthly_deposit: 월별 입금액 (원)
            commission_rate: 거래 수수료율 (0.015%)
            tax_rate: 거래세율 - 매도시만 (0.23%)
            bid_ask_spread: 호가 스프레드 (0.01%)
            rebalance_frequency: "monthly" 또는 "weekly"
            deposit_day: 입금일 (1-28)
            config_path: 설정 파일 경로 (선택사항)
        """
        # 설정 로드
        if config_path:
            self._load_config(config_path)
        else:
            self.initial_capital = initial_capital
            self.monthly_deposit = monthly_deposit
            self.commission_rate = commission_rate
            self.tax_rate = tax_rate
            self.bid_ask_spread = bid_ask_spread
            self.rebalance_frequency = rebalance_frequency
            self.deposit_day = deposit_day

        # 포트폴리오 상태 초기화
        self.reset()

    def reset(self):
        """포트폴리오 상태 초기화"""
        self.cash = self.initial_capital
        self.holdings = {}  # {ticker: 주식수}
        self.equity_curve = []
        self.trade_log = []

    def _load_config(self, config_path: str):
        """YAML 설정 파일 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        inv = config['investment']
        self.initial_capital = inv['initial_capital']
        self.monthly_deposit = inv['monthly_deposit']
        self.deposit_day = inv['deposit_day']

        txn = config['transaction']
        self.commission_rate = txn['commission_rate']
        self.tax_rate = txn['tax_rate']
        self.bid_ask_spread = txn.get('slippage', 0.0001)  # 하위 호환성

        reb = config['rebalancing']
        self.rebalance_frequency = reb['frequency']

    def _is_month_start(self, date: pd.Timestamp, prev_date: pd.Timestamp) -> bool:
        """월 변경 여부 확인 (첫 거래일)"""
        return date.month != prev_date.month

    def _is_rebalance_day(self, date: pd.Timestamp, prev_date: pd.Timestamp) -> bool:
        """리밸런싱 일자 여부 확인"""
        if self.rebalance_frequency == "monthly":
            return self._is_month_start(date, prev_date)
        elif self.rebalance_frequency == "weekly":
            return date.weekday() == 0 and (date - prev_date).days >= 1
        return False

    def _get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        현재 포트폴리오 총 가치 계산

        Args:
            prices: {ticker: 현재가}

        Returns:
            현금 + 보유주식 가치
        """
        holdings_value = sum(
            shares * prices.get(ticker, 0)
            for ticker, shares in self.holdings.items()
        )
        return self.cash + holdings_value

    def _get_execution_price(self, price: float, side: str) -> float:
        """
        실제 체결 가격 계산 (bid-ask spread 반영)

        Args:
            price: 기준 가격 (중간가)
            side: "buy" 또는 "sell"

        Returns:
            실제 체결 가격

        Note:
            - 매수: ask price (중간가 + spread/2)
            - 매도: bid price (중간가 - spread/2)
            - 기존 구현은 spread를 양쪽에 적용했으나, 여기서는 한쪽만 적용
        """
        if side == "buy":
            return price * (1 + self.bid_ask_spread / 2)
        else:
            return price * (1 - self.bid_ask_spread / 2)

    def _calculate_target_shares(
        self,
        total_value: float,
        target_weights: Dict[str, float],
        prices: Dict[str, float]
    ) -> Dict[str, int]:
        """
        목표 비중에 따른 목표 주식 수 계산

        Args:
            total_value: 현재 총 자산
            target_weights: {ticker: 목표비중}
            prices: {ticker: 현재가}

        Returns:
            {ticker: 목표주식수}
        """
        target_shares = {}

        for ticker, weight in target_weights.items():
            if ticker not in prices or prices[ticker] <= 0:
                continue

            target_value = total_value * weight
            execution_price = self._get_execution_price(prices[ticker], "buy")
            shares = int(target_value / execution_price)  # 정수 주식만

            if shares > 0:
                target_shares[ticker] = shares

        return target_shares

    def _rebalance_with_cash_constraint(
        self,
        date: pd.Timestamp,
        target_weights: Dict[str, float],
        prices: Dict[str, float]
    ):
        """
        현금 제약 하에서 비례적 리밸런싱

        개선사항:
        - 현금 부족 시 모든 종목을 비례적으로 축소
        - 순차적 처리로 인한 불균형 방지

        Args:
            date: 리밸런싱 일자
            target_weights: 목표 비중
            prices: 현재 가격
        """
        # 1. 현재 포트폴리오 가치 계산
        total_value = self._get_portfolio_value(prices)

        if total_value <= 0:
            return

        # 2. 초기 목표 주식 수 계산
        target_shares = self._calculate_target_shares(total_value, target_weights, prices)

        # 3. 매도 주문 실행 (현금 확보)
        sells = self._determine_sells(target_shares)
        for ticker, shares in sells:
            self._execute_trade(date, ticker, shares, prices[ticker], "sell")

        # 4. 필요한 매수 금액 계산
        buys = self._determine_buys(target_shares)
        required_cash = self._calculate_required_cash(buys, prices)

        # 5. 현금 부족 시 비례적 조정
        if required_cash > self.cash:
            scale_factor = self.cash / required_cash
            buys = [(ticker, int(shares * scale_factor)) for ticker, shares in buys]

        # 6. 매수 주문 실행
        for ticker, shares in buys:
            if shares > 0:
                self._execute_trade(date, ticker, shares, prices[ticker], "buy")

    def _determine_sells(self, target_shares: Dict[str, int]) -> List[Tuple[str, int]]:
        """
        매도할 종목 및 수량 결정

        Args:
            target_shares: 목표 주식 수

        Returns:
            [(ticker, shares_to_sell), ...]
        """
        sells = []

        for ticker, current_shares in self.holdings.items():
            target = target_shares.get(ticker, 0)
            if target < current_shares:
                sells.append((ticker, current_shares - target))

        return sells

    def _determine_buys(self, target_shares: Dict[str, int]) -> List[Tuple[str, int]]:
        """
        매수할 종목 및 수량 결정

        Args:
            target_shares: 목표 주식 수

        Returns:
            [(ticker, shares_to_buy), ...]
        """
        buys = []

        for ticker, target in target_shares.items():
            current = self.holdings.get(ticker, 0)
            if target > current:
                buys.append((ticker, target - current))

        return buys

    def _calculate_required_cash(
        self,
        buys: List[Tuple[str, int]],
        prices: Dict[str, float]
    ) -> float:
        """
        매수에 필요한 총 현금 계산 (수수료 포함)

        Args:
            buys: [(ticker, shares), ...]
            prices: {ticker: price}

        Returns:
            필요한 총 현금
        """
        total_required = 0.0

        for ticker, shares in buys:
            if ticker not in prices:
                continue

            execution_price = self._get_execution_price(prices[ticker], "buy")
            trade_value = shares * execution_price
            commission = trade_value * self.commission_rate
            total_required += trade_value + commission

        return total_required

    def _execute_trade(
        self,
        date: pd.Timestamp,
        ticker: str,
        shares: int,
        market_price: float,
        side: str
    ):
        """
        거래 실행 (수수료, 세금, 슬리피지 반영)

        Args:
            date: 거래 일자
            ticker: 종목 코드
            shares: 주식 수
            market_price: 시장 가격 (중간가)
            side: "buy" 또는 "sell"
        """
        if shares <= 0:
            return

        # 실제 체결 가격 계산
        execution_price = self._get_execution_price(market_price, side)
        trade_value = shares * execution_price

        # 거래 비용 계산
        commission = trade_value * self.commission_rate

        if side == "buy":
            total_cost = trade_value + commission

            # 현금 부족 체크 (안전장치)
            if total_cost > self.cash:
                return  # 리밸런싱 로직에서 이미 처리되므로 도달하지 않아야 함

            # 매수 실행
            self.cash -= total_cost
            self.holdings[ticker] = self.holdings.get(ticker, 0) + shares

            # 거래 기록
            self.trade_log.append({
                'date': date,
                'ticker': ticker,
                'side': 'buy',
                'shares': shares,
                'market_price': market_price,
                'execution_price': execution_price,
                'value': trade_value,
                'commission': commission,
                'tax': 0,
                'total_cost': total_cost
            })

        else:  # sell
            tax = trade_value * self.tax_rate
            total_proceeds = trade_value - commission - tax

            # 매도 실행
            self.cash += total_proceeds
            self.holdings[ticker] = self.holdings.get(ticker, 0) - shares

            # 보유 수량이 0이면 제거
            if self.holdings[ticker] <= 0:
                del self.holdings[ticker]

            # 거래 기록
            self.trade_log.append({
                'date': date,
                'ticker': ticker,
                'side': 'sell',
                'shares': shares,
                'market_price': market_price,
                'execution_price': execution_price,
                'value': trade_value,
                'commission': commission,
                'tax': tax,
                'total_proceeds': total_proceeds
            })

    def run(
        self,
        strategy: BaseETFStrategy,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        백테스트 실행

        Args:
            strategy: 투자 전략
            data: {ticker: OHLCV DataFrame}
            start_date: 시작일 (None = 최초 가능일)
            end_date: 종료일 (None = 최종 가능일)
            verbose: 상세 로그 출력 여부

        Returns:
            자산 곡선 DataFrame
        """
        # 상태 초기화
        self.reset()

        # 날짜 목록 생성
        dates = self._get_trading_dates(data, start_date, end_date)

        if len(dates) < 2:
            raise ValueError("백테스트를 위해 최소 2개 거래일이 필요합니다")

        if verbose:
            print(f"백테스트 기간: {dates[0]} ~ {dates[-1]}")
            print(f"총 {len(dates)}일")

        # 첫날은 prev_date 개념이 없으므로 특별 처리
        first_date = dates[0]
        prices = self._get_prices_at_date(data, first_date)

        # 첫날 리밸런싱
        target_weights = strategy.get_weights(data, first_date)
        self._rebalance_with_cash_constraint(first_date, target_weights, prices)

        # 첫날 포트폴리오 가치 기록
        self._record_equity(first_date, prices)

        # 두번째 날부터 시뮬레이션
        prev_date = first_date

        for date in dates[1:]:
            prices = self._get_prices_at_date(data, date)

            if not prices:
                continue

            # 월초 입금
            if self._is_month_start(date, prev_date):
                self.cash += self.monthly_deposit

            # 리밸런싱
            if self._is_rebalance_day(date, prev_date):
                target_weights = strategy.get_weights(data, date)
                self._rebalance_with_cash_constraint(date, target_weights, prices)

            # 포트폴리오 가치 기록
            self._record_equity(date, prices)

            prev_date = date

        return pd.DataFrame(self.equity_curve)

    def _get_trading_dates(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> List[pd.Timestamp]:
        """거래일 목록 생성"""
        all_dates = set()
        for df in data.values():
            all_dates.update(df['date'].values)

        dates = sorted(pd.to_datetime(list(all_dates)))

        if start_date:
            dates = [d for d in dates if d >= pd.to_datetime(start_date)]
        if end_date:
            dates = [d for d in dates if d <= pd.to_datetime(end_date)]

        return dates

    def _get_prices_at_date(
        self,
        data: Dict[str, pd.DataFrame],
        date: pd.Timestamp
    ) -> Dict[str, float]:
        """특정 일자의 종가 가져오기"""
        prices = {}

        for ticker, df in data.items():
            ticker_data = df[df['date'] == date]
            if not ticker_data.empty:
                prices[ticker] = ticker_data['close'].values[0]

        return prices

    def _record_equity(self, date: pd.Timestamp, prices: Dict[str, float]):
        """자산 곡선 기록"""
        portfolio_value = self._get_portfolio_value(prices)

        self.equity_curve.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'holdings_value': portfolio_value - self.cash
        })

    def get_trade_log(self) -> pd.DataFrame:
        """거래 로그 DataFrame 반환"""
        return pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()

    def get_summary_stats(self, equity_df: pd.DataFrame) -> Dict:
        """
        성과 지표 계산

        Args:
            equity_df: 자산 곡선 DataFrame

        Returns:
            성과 지표 딕셔너리
        """
        if equity_df.empty:
            return {}

        # 수익률 계산
        equity_df = equity_df.copy()
        equity_df['returns'] = equity_df['portfolio_value'].pct_change()

        # 기본 지표
        initial_value = equity_df['portfolio_value'].iloc[0]
        final_value = equity_df['portfolio_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value

        # CAGR
        days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
        years = days / 365.25
        cagr = (final_value / initial_value) ** (1 / years) - 1 if years > 0 else 0

        # 변동성 (연환산)
        daily_returns = equity_df['returns'].dropna()
        volatility = daily_returns.std() * np.sqrt(252)

        # 샤프 비율 (무위험 수익률 = 0 가정)
        sharpe = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0

        # 최대 낙폭 (MDD)
        cummax = equity_df['portfolio_value'].cummax()
        drawdown = (equity_df['portfolio_value'] - cummax) / cummax
        max_drawdown = drawdown.min()

        # 승률
        winning_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        win_rate = winning_days / total_days if total_days > 0 else 0

        return {
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_days': total_days
        }


if __name__ == "__main__":
    print("ETF Backtesting Engine")
    print("=" * 60)
    print("주요 기능:")
    print("- 월별 정기 입금 (DCA)")
    print("- 자동 리밸런싱")
    print("- 현실적인 거래 비용 반영")
