# 코드 리뷰 및 발견된 문제점

## 🔍 발견된 주요 문제점

### 1. **슬리피지 계산 오류** (중요도: 높음)
**위치**: `backtesting/engine.py:246-249`
```python
if side == "buy":
    execution_price = price * (1 + self.slippage)
else:
    execution_price = price * (1 - self.slippage)
```

**문제점**:
- 슬리피지가 이중으로 적용됨
- 매수시 높은 가격, 매도시 낮은 가격 모두 불리하게 적용
- 실제로는 bid-ask spread를 모델링하는 것이므로 한쪽만 적용해야 함

**영향**: 거래 비용이 과대 계상됨

---

### 2. **리밸런싱 시 현금 부족 처리** (중요도: 중간)
**위치**: `backtesting/engine.py:261-271`

**문제점**:
- 매수 주문을 순차적으로 처리하므로 나중 종목이 불리
- 전체 포트폴리오 관점에서 비례적으로 조정하지 않음

**시나리오**:
```
목표: A 50%, B 50% (각 500만원씩)
현금: 800만원만 보유

현재 로직:
1. A 매수 시도 → 500만원 투자 (성공)
2. B 매수 시도 → 300만원만 투자 (실패)
결과: A 62.5%, B 37.5% (불균형)

이상적 로직:
1. 현금 부족 감지
2. 비례적 조정: A 400만원, B 400만원
결과: A 50%, B 50% (균형 유지)
```

---

### 3. **모멘텀 계산 인덱싱** (중요도: 낮음)
**위치**: `strategy/base.py:175-185`

```python
current_price = close_prices[-1]
past_price = close_prices[-lookback_days]
```

**잠재적 문제**:
- `-lookback_days`는 배열의 끝에서 lookback_days 위치를 가리킴
- 데이터가 정확히 일별이 아닌 경우 문제 발생 가능
- 하지만 현재 구현에서는 실제로 문제 없음 (일별 데이터 가정)

---

### 4. **데이터 필터링 성능** (중요도: 낮음)
**위치**: 여러 곳에서 `df[df['date'] <= date]` 사용

**문제점**:
- 매번 전체 DataFrame 스캔
- 백테스트가 길어질수록 느려짐

**개선안**:
- 인덱싱 활용
- 또는 미리 정렬된 데이터에서 binary search 활용

---

### 5. **초기 투자 시점 혼란** (중요도: 낮음)
**위치**: `backtesting/engine.py:376-378`

```python
if i == 0 or self._is_rebalance_day(date, prev_date):
    target_weights = strategy.get_weights(data, date)
    self._rebalance(date, target_weights, prices)
```

**문제점**:
- `i == 0`일 때 `prev_date`는 의미 없음
- 명확성 부족

---

## ✅ 검증된 정상 로직

### 1. **거래 순서** (✓ 정상)
- 매도 먼저 → 매수 나중 (현금 확보 후 투자)
- 올바른 구현

### 2. **정수 주식 제약** (✓ 정상)
- `int(target_value / prices[ticker])`로 정수 주식만 거래
- 올바른 구현

### 3. **월별 입금/리밸런싱** (✓ 정상)
- `date.month != prev_date.month`로 월 변경 감지
- 올바른 구현

### 4. **거래 비용** (✓ 정상)
- 수수료: 매수/매도 양쪽
- 세금: 매도만
- 올바른 구현

---

## 📋 전략 로직 검증

### 전략 1: Global Asset Allocation (✓ 정상)
- 고정 비중 반환
- 단순하고 명확

### 전략 2: Momentum Sector Rotation (✓ 정상)
- 3개월/6개월 모멘텀 계산
- 상위 N개 선택
- 균등 비중 할당

### 전략 3: Dividend + Growth Mix (✓ 정상)
- 50/50 고정 비중
- 단순하고 명확

---

## 🎯 리팩토링 우선순위

1. **높음**: 슬리피지 계산 수정
2. **중간**: 리밸런싱 로직 개선
3. **낮음**: 성능 최적화
4. **추가**: 단위 테스트 추가
5. **추가**: 로깅 및 디버깅 개선
