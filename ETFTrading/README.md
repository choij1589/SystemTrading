# ETF Trading - 한국 ETF 자동투자 시스템

한국투자증권 API를 활용한 ETF 자동투자 백테스팅 시스템

## 📊 프로젝트 개요

정기 입금(Dollar Cost Averaging)과 자동 리밸런싱을 통한 ETF 투자 전략 백테스팅 및 비교

### 투자 조건
- **초기 투자금**: 420만원
- **월 정기 입금**: 30만원
- **리밸런싱**: 매월
- **거래 비용**: 수수료 0.015% + 거래세 0.23%(매도) + bid-ask spread 0.01%

### 구현된 전략

#### 1. 글로벌 자산배분 (안정형)
- 국내주식 30% + 미국주식 40% + 채권 20% + 금 10%
- 낮은 변동성, 전천후 포트폴리오

#### 2. 모멘텀 섹터 로테이션 (공격형)
- 3/6개월 모멘텀 기반 상위 3개 섹터 선택
- 높은 수익 잠재력, 트렌드 추종

#### 3. 배당 + 성장 혼합 (균형형)
- 미국 배당주 50% + 나스닥100 50%
- 배당 수익 + 자본 이득

---

## 🚀 빠른 시작

### 1. 환경 설정
```bash
conda create -n etftrading python=3.10
conda activate etftrading
pip install pandas numpy matplotlib seaborn pyyaml
pip install yfinance  # Yahoo Finance 데이터용
```

### 2. 백테스트 실행
```bash
cd ETFTrading
python run_backtest.py
```

### 3. 결과 확인
- **backtest_results.png**: 성과 시각화 차트
- **BACKTEST_REPORT.md**: 상세 분석 보고서

---

## 📁 프로젝트 구조

```
ETFTrading/
├── data/
│   ├── kis_client.py          # 한국투자증권 API
│   ├── data_loader.py         # ETF 데이터 로더
│   └── yahoo_loader.py        # Yahoo Finance 로더 (테스트용)
│
├── strategy/
│   ├── base.py                # 기본 전략 클래스
│   ├── asset_allocation.py    # 전략 1: 글로벌 자산배분
│   ├── momentum_rotation.py   # 전략 2: 모멘텀 섹터 로테이션
│   └── dividend_growth.py     # 전략 3: 배당+성장 혼합
│
├── backtesting/
│   └── engine.py              # 백테스팅 엔진
│
├── config/
│   ├── config.yaml            # 투자 설정
│   ├── etf_universe.yaml      # ETF 유니버스
│   └── secrets.yaml.example   # API 키 템플릿
│
├── notebooks/
│   └── 01_strategy_comparison.ipynb
│
├── run_backtest.py            # 백테스트 실행 스크립트
├── BACKTEST_REPORT.md         # 백테스트 결과 보고서
├── STRATEGY_LOGIC.md          # 전략 로직 설명
└── README.md                  # 이 파일
```

---

## 🎯 주요 기능

### 백테스팅 엔진
- ✅ 월별 정기 입금 (DCA)
- ✅ 자동 리밸런싱
- ✅ 현실적인 거래 비용
- ✅ 정수 주식 제약
- ✅ 비례적 현금 배분

### 성과 지표
- **CAGR**: 연평균 복리 수익률
- **Sharpe Ratio**: 위험 대비 수익
- **MDD**: 최대 낙폭
- **Win Rate**: 승률

---

## 📈 백테스트 결과 (2020-2024)

| 전략 | 최종 자산 | CAGR | Sharpe | MDD |
|------|----------|------|--------|-----|
| 전략 1 (자산배분) | 3,454만원 | 54.86% | 2.49 | -9.59% |
| 전략 2 (모멘텀) | 4,565만원 | 64.07% | 1.88 | -33.81% |
| 전략 3 (배당+성장) | 2,975만원 | 50.13% | 1.88 | -25.94% |

*상세 결과는 BACKTEST_REPORT.md 참조*

---

## 🔧 설정 커스터마이징

`config/config.yaml` 파일 수정:
```yaml
investment:
  initial_capital: 4_200_000
  monthly_deposit: 300_000

rebalancing:
  frequency: "monthly"  # 또는 "weekly"

transaction:
  commission_rate: 0.00015
  tax_rate: 0.0023
  bid_ask_spread: 0.0001
```

---

## ⚠️ 주의사항

### 백테스팅 한계
- 과거 성과 ≠ 미래 수익
- 시뮬레이션 데이터 사용 (실제 ETF 가격과 차이 있음)
- 생존 편향 미반영

### 실전 투자 전
1. ✅ 한국투자증권 API 키 발급
2. ✅ 모의투자로 충분히 테스트
3. ✅ 소액으로 시작
4. ✅ 리스크 관리 철저히

---

## 📚 문서

- **README.md**: 프로젝트 개요 (이 파일)
- **STRATEGY_LOGIC.md**: 전략 로직 상세 설명
- **BACKTEST_REPORT.md**: 백테스트 결과 분석

---

## 📄 라이선스

MIT License

---

**면책조항**: 이 프로젝트는 교육 목적으로 제공됩니다. 투자 결정은 본인의 판단과 책임하에 이루어져야 합니다.
