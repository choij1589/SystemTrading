# ETF 투자 전략 로직 상세 설명

## 📊 전략 1: 글로벌 자산배분 (Global Asset Allocation)

### 전략 타입
- **카테고리**: 정적 배분 (Static Allocation)
- **리밸런싱**: 월 1회
- **위험도**: 낮음

### 포트폴리오 구성
| ETF 코드 | ETF 명 | 자산 클래스 | 목표 비중 |
|---------|--------|-----------|----------|
| 069500 | KODEX 200 | 국내 주식 | 30% |
| 360750 | TIGER 미국S&P500 | 미국 주식 | 40% |
| 152380 | KODEX 국고채3년 | 채권 | 20% |
| 132030 | KODEX 골드선물(H) | 금/원자재 | 10% |

### 로직 흐름도
```
매 리밸런싱 시점:
1. 목표 비중 반환: {069500: 0.30, 360750: 0.40, 152380: 0.20, 132030: 0.10}
2. 백테스팅 엔진이 현재 비중 계산
3. 목표 비중과 차이 계산
4. 매도/매수 주문 실행하여 목표 비중 달성
```

### 의사결정 코드
```python
def get_weights(self, data, date):
    # 시장 상황과 무관하게 고정 비중 반환
    return {
        "069500": 0.30,  # 항상 30%
        "360750": 0.40,  # 항상 40%
        "152380": 0.20,  # 항상 20%
        "132030": 0.10   # 항상 10%
    }
```

### 투자 철학
1. **분산 투자**: 상관관계가 낮은 자산에 분산
2. **리스크 패리티**: 각 자산의 위험 기여도 조정
3. **전천후 전략**: 모든 경제 환경에서 작동
4. **낮은 회전율**: 고정 비중으로 거래 비용 최소화

### 리밸런싱 예시
```
초기 (1,000만원):
- 069500: 300만원 (30주)
- 360750: 400만원 (40주)
- 152380: 200만원 (200주)
- 132030: 100만원 (10주)

1개월 후 (가격 변동):
- 069500: 330만원 (34%) ← 목표보다 높음
- 360750: 360만원 (37%) ← 목표보다 낮음
- 152380: 190만원 (20%)
- 132030: 90만원 (9%) ← 목표보다 낮음
합계: 970만원

리밸런싱 (목표: 30/40/20/10):
- 069500: 매도 39만원 → 291만원 (30%)
- 360750: 매수 28만원 → 388만원 (40%)
- 152380: 유지 → 194만원 (20%)
- 132030: 매수 7만원 → 97만원 (10%)
```

---

## 🚀 전략 2: 모멘텀 섹터 로테이션 (Momentum Sector Rotation)

### 전략 타입
- **카테고리**: 동적 배분 (Dynamic Allocation)
- **리밸런싱**: 월 1회
- **위험도**: 높음

### 투자 유니버스
| ETF 코드 | ETF 명 | 섹터 |
|---------|--------|------|
| 091180 | KODEX 반도체 | 반도체 |
| 157450 | TIGER 2차전지테마 | 2차전지 |
| 227540 | TIGER 200 IT | IT |
| 139230 | TIGER 200 건설 | 건설 |
| 139260 | TIGER 200 에너지화학 | 에너지/화학 |
| 139250 | TIGER 200 금융 | 금융 |
| 228790 | TIGER 200 헬스케어 | 헬스케어 |
| 360750 | TIGER 미국S&P500 | 미국 주식 |

### 로직 흐름도
```
매 리밸런싱 시점 (t):

1. 모든 섹터 ETF에 대해:

   a. 3개월 모멘텀 계산:
      momentum_3m = (price[t] - price[t-60]) / price[t-60]

   b. 6개월 모멘텀 계산:
      momentum_6m = (price[t] - price[t-120]) / price[t-120]

   c. 복합 모멘텀 점수:
      score = 0.5 × momentum_3m + 0.5 × momentum_6m

2. 모멘텀 점수로 섹터 순위 정렬 (내림차순)

3. 상위 3개 섹터 선택:
   - 조건: 모멘텀 점수 ≥ 0 (양수만)
   - 선택된 섹터가 없으면 → 100% 현금 보유

4. 선택된 섹터에 균등 비중 할당:
   - 3개 선택 → 각 33.33%
   - 2개 선택 → 각 50%
   - 1개 선택 → 100%
```

### 의사결정 코드
```python
def get_weights(self, data, date):
    momentum_scores = {}

    # 1. 모든 섹터의 모멘텀 계산
    for ticker in universe:
        df = data[ticker][data[ticker]['date'] <= date]

        if len(df) < 130:  # 최소 6개월 데이터 필요
            continue

        # 3개월 모멘텀 (60 거래일)
        mom_3m = (df['close'].iloc[-1] - df['close'].iloc[-60]) / df['close'].iloc[-60]

        # 6개월 모멘텀 (120 거래일)
        mom_6m = (df['close'].iloc[-1] - df['close'].iloc[-120]) / df['close'].iloc[-120]

        # 복합 점수 (50:50 가중)
        score = 0.5 * mom_3m + 0.5 * mom_6m
        momentum_scores[ticker] = score

    # 2. 순위 정렬
    ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)

    # 3. 상위 3개 중 양수 모멘텀만 선택
    selected = [ticker for ticker, score in ranked[:3] if score >= 0.0]

    # 4. 균등 비중 할당
    if len(selected) == 0:
        return {}  # 현금 100%

    weight = 1.0 / len(selected)
    return {ticker: weight for ticker in selected}
```

### 투자 철학
1. **모멘텀 효과**: 과거 상승세가 지속되는 경향
2. **섹터 로테이션**: 경기 사이클에 따른 섹터 순환
3. **트렌드 추종**: 상승 추세만 따라감
4. **손절 자동화**: 모멘텀 하락 시 자동 매도

### 리밸런싱 예시
```
2024년 1월 리밸런싱:

1. 모멘텀 계산:
   - 091180 (반도체): +45% (3개월), +80% (6개월) → 점수 62.5%
   - 157450 (2차전지): +30% (3개월), +50% (6개월) → 점수 40%
   - 227540 (IT): +25% (3개월), +40% (6개월) → 점수 32.5%
   - 139230 (건설): -5% (3개월), +10% (6개월) → 점수 2.5%
   - 139260 (에너지): -10% (3개월), -5% (6개월) → 점수 -7.5%
   - ...

2. 순위:
   1위: 091180 (반도체) - 62.5%
   2위: 157450 (2차전지) - 40%
   3위: 227540 (IT) - 32.5%

3. 포트폴리오 (1,000만원):
   - 091180: 333만원 (33.33%)
   - 157450: 333만원 (33.33%)
   - 227540: 333만원 (33.33%)
   - 나머지: 매도 또는 미보유

2024년 2월 리밸런싱:

1. 모멘텀 재계산 (순위 변동):
   1위: 157450 (2차전지) - 70%
   2위: 228790 (헬스케어) - 55%
   3위: 091180 (반도체) - 50%

2. 포트폴리오 조정:
   - 227540 (IT): 전량 매도
   - 228790 (헬스케어): 신규 매수
   - 비중 재조정: 각 33.33%
```

---

## ⚖️ 전략 3: 배당 + 성장 혼합 (Dividend + Growth Mix)

### 전략 타입
- **카테고리**: 정적 배분 (Static Allocation)
- **리밸런싱**: 월 1회
- **위험도**: 중간

### 포트폴리오 구성
| ETF 코드 | ETF 명 | 특성 | 목표 비중 |
|---------|--------|------|----------|
| 458730 | TIGER 미국배당다우존스 | 배당 수익 | 50% |
| 133690 | TIGER 미국나스닥100 | 성장 | 50% |

### 로직 흐름도
```
매 리밸런싱 시점:
1. 목표 비중 반환: {458730: 0.50, 133690: 0.50}
2. 백테스팅 엔진이 현재 비중 계산
3. 목표 비중과 차이 계산
4. 매도/매수 주문 실행하여 50/50 비중 달성
```

### 의사결정 코드
```python
def get_weights(self, data, date):
    # 시장 상황과 무관하게 50/50 비중 반환
    return {
        "458730": 0.50,  # 배당주 50%
        "133690": 0.50   # 성장주 50%
    }
```

### 투자 철학
1. **수익의 이원화**:
   - 배당 수익: 안정적 현금 흐름
   - 자본 이득: 성장 잠재력

2. **리스크 균형**:
   - 배당주: 낮은 변동성, 방어적
   - 성장주: 높은 변동성, 공격적

3. **미국 시장 집중**:
   - 세계 최대 경제
   - 우량 기업 집중

### 리밸런싱 예시
```
초기 (1,000만원):
- 458730 (배당): 500만원 (50%)
- 133690 (나스닥): 500만원 (50%)

1개월 후 (가격 변동):
- 458730: 510만원 (49%) ← 배당주는 소폭 상승
- 133690: 530만원 (51%) ← 성장주는 큰 폭 상승
합계: 1,040만원

리밸런싱 (목표: 50/50):
- 458730: 매수 10만원 → 520만원 (50%)
- 133690: 매도 10만원 → 520만원 (50%)
```

---

## 🔄 백테스팅 엔진 로직

### 시뮬레이션 흐름
```
초기화:
- 현금 = 4,200,000원
- 보유주식 = {}

매 거래일 (t):

  1. 입금 체크:
     if 이번달 첫 거래일:
       현금 += 300,000원

  2. 리밸런싱 체크:
     if 이번달 첫 거래일 or t == 0:
       a. 전략에서 목표 비중 가져오기
       b. 현재 포트폴리오 가치 계산
       c. 각 종목의 목표 주식 수 계산
       d. 매도 주문 실행 (현금 확보)
       e. 매수 주문 실행 (목표 비중 달성)

  3. 포트폴리오 가치 기록:
     - 현금 + 보유주식 가치

최종:
- 자산곡선, 거래로그, 성과지표 반환
```

### 리밸런싱 상세 로직
```python
def rebalance(date, target_weights, prices):
    # 1. 현재 포트폴리오 가치 계산
    total_value = cash + sum(holdings[ticker] * prices[ticker])

    # 2. 각 종목의 목표 주식 수 계산
    for ticker, weight in target_weights.items():
        target_value = total_value * weight
        target_shares[ticker] = int(target_value / prices[ticker])  # 정수만

    # 3. 매도 주문 (현금 확보)
    for ticker in holdings:
        current_shares = holdings[ticker]
        target_shares_for_ticker = target_shares.get(ticker, 0)

        if target_shares_for_ticker < current_shares:
            shares_to_sell = current_shares - target_shares_for_ticker
            execute_sell(ticker, shares_to_sell, prices[ticker])

    # 4. 매수 주문 (목표 비중 달성)
    for ticker in target_shares:
        current_shares = holdings.get(ticker, 0)
        target = target_shares[ticker]

        if target > current_shares:
            shares_to_buy = target - current_shares
            execute_buy(ticker, shares_to_buy, prices[ticker])
```

### 거래 실행 로직
```python
def execute_trade(ticker, shares, price, side):
    if side == "buy":
        # 매수 비용 계산
        trade_value = shares * price
        commission = trade_value * 0.00015
        total_cost = trade_value + commission

        # 현금 부족 시 주식 수 조정
        if total_cost > cash:
            affordable_shares = int(cash / (price * 1.00015))
            shares = affordable_shares
            total_cost = shares * price * 1.00015

        # 매수 실행
        cash -= total_cost
        holdings[ticker] += shares

    elif side == "sell":
        # 매도 수익 계산
        trade_value = shares * price
        commission = trade_value * 0.00015
        tax = trade_value * 0.0023
        total_proceeds = trade_value - commission - tax

        # 매도 실행
        cash += total_proceeds
        holdings[ticker] -= shares
```

---

## 📈 성과 지표 계산

### CAGR (연평균 복리 수익률)
```python
days = (end_date - start_date).days
years = days / 365.25
CAGR = (final_value / initial_value) ** (1 / years) - 1
```

### Sharpe Ratio (샤프 비율)
```python
daily_returns = portfolio_value.pct_change()
annual_return = daily_returns.mean() * 252
annual_volatility = daily_returns.std() * sqrt(252)
Sharpe = annual_return / annual_volatility
```

### Maximum Drawdown (최대 낙폭)
```python
cummax = portfolio_value.cummax()
drawdown = (portfolio_value - cummax) / cummax
MDD = drawdown.min()
```

### Win Rate (승률)
```python
daily_returns = portfolio_value.pct_change()
winning_days = (daily_returns > 0).sum()
Win_Rate = winning_days / len(daily_returns)
```
