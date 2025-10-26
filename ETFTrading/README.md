# ETF Trading - 한국 ETF 자동투자 시스템

한국투자증권 API를 활용한 ETF 자동투자 백테스팅 시스템입니다.

## 📊 프로젝트 개요

이 프로젝트는 정기 입금(Dollar Cost Averaging)과 자동 리밸런싱을 통한 ETF 투자 전략을 백테스트하고 비교하는 시스템입니다.

### 투자 조건
- **초기 투자금**: 420만원
- **월 정기 입금**: 30만원
- **리밸런싱**: 매월 또는 매주
- **거래 비용**: 수수료 0.015% + 거래세 0.23%(매도시)

### 구현된 전략

#### 1. 글로벌 자산배분 (Global Asset Allocation)
올웨더 포트폴리오 스타일의 분산투자 전략:
- KODEX 200 (국내주식): 30%
- TIGER 미국S&P500 (미국주식): 40%
- KODEX 국고채3년 (채권): 20%
- KODEX 골드선물 (금): 10%

**특징**: 안정적, 낮은 변동성, 전천후 포트폴리오

#### 2. 모멘텀 섹터 로테이션 (Momentum Sector Rotation)
상승 모멘텀이 강한 섹터를 선택하여 투자:
- 3개월/6개월 모멘텀 계산
- 상위 3개 섹터에 균등 투자
- 매월 리밸런싱

**대상 섹터**: 반도체, 2차전지, IT, 건설, 에너지화학, 금융, 헬스케어, 미국주식

**특징**: 높은 수익 잠재력, 트렌드 추종, 높은 변동성

#### 3. 배당 + 성장 혼합 (Dividend + Growth Mix)
배당주와 성장주의 균형 전략:
- TIGER 미국배당다우존스: 50%
- TIGER 미국나스닥100: 50%

**특징**: 배당 수익 + 자본 이득, 중간 리스크

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# Conda 환경 생성 및 활성화
conda create -n etftrading python=3.10
conda activate etftrading

# 필수 패키지 설치
pip install pandas numpy matplotlib seaborn
pip install yfinance  # Yahoo Finance 데이터용
pip install pyyaml requests

# Jupyter 설치 (노트북 실행용)
pip install jupyter ipykernel
python -m ipykernel install --user --name=etftrading
```

### 2. 설정 파일 (선택사항)

한국투자증권 API를 사용하려면:

```bash
cd ETFTrading/config
cp secrets.yaml.example secrets.yaml
# secrets.yaml 파일을 열어 API 키 입력
```

### 3. 백테스트 실행

```bash
cd notebooks
jupyter notebook 01_strategy_comparison.ipynb
```

노트북을 열고 셀을 순차적으로 실행하세요.

## 📁 프로젝트 구조

```
ETFTrading/
├── data/
│   ├── kis_client.py          # 한국투자증권 API 클라이언트
│   ├── data_loader.py         # ETF 데이터 로더 (KIS API)
│   └── yahoo_loader.py        # Yahoo Finance 데이터 로더
│
├── strategy/
│   ├── base.py                # 전략 기본 클래스
│   ├── asset_allocation.py    # 전략 1: 글로벌 자산배분
│   ├── momentum_rotation.py   # 전략 2: 모멘텀 섹터 로테이션
│   └── dividend_growth.py     # 전략 3: 배당 + 성장 혼합
│
├── backtesting/
│   └── engine.py              # 백테스팅 엔진
│
├── config/
│   ├── config.yaml            # 투자 설정
│   ├── etf_universe.yaml      # ETF 유니버스 정의
│   └── secrets.yaml.example   # API 키 템플릿
│
├── notebooks/
│   └── 01_strategy_comparison.ipynb  # 전략 비교 노트북
│
└── README.md                  # 이 파일
```

## 🎯 주요 기능

### 1. 데이터 수집
```python
from ETFTrading.data.yahoo_loader import YahooETFLoader

loader = YahooETFLoader()
data = loader.load_multiple(
    tickers=["069500", "360750"],
    start_date="2020-01-01"
)
```

### 2. 전략 정의
```python
from ETFTrading.strategy import GlobalAssetAllocationStrategy

strategy = GlobalAssetAllocationStrategy()
print(strategy.describe())
```

### 3. 백테스팅
```python
from ETFTrading.backtesting import ETFBacktestEngine

engine = ETFBacktestEngine(
    initial_capital=4_200_000,
    monthly_deposit=300_000
)

equity_curve = engine.run(strategy, data)
stats = engine.get_summary_stats(equity_curve)
```

## 📈 백테스팅 엔진 특징

- ✅ **정기 입금**: 매월 자동 입금 시뮬레이션
- ✅ **리밸런싱**: 주간/월간 자동 리밸런싱
- ✅ **거래 비용**: 수수료 + 거래세 + 슬리피지
- ✅ **정수 주식**: 실제 시장처럼 정수 주식만 거래
- ✅ **성과 지표**: CAGR, MDD, Sharpe, Win Rate 등

## 🔧 설정 커스터마이징

`config/config.yaml` 파일에서 설정 변경:

```yaml
investment:
  initial_capital: 4_200_000
  monthly_deposit: 300_000
  deposit_day: 1

rebalancing:
  frequency: "monthly"  # 또는 "weekly"

transaction:
  commission_rate: 0.00015  # 0.015%
  tax_rate: 0.0023          # 0.23%
```

## 📊 성과 측정 지표

- **Total Return**: 총 수익률
- **CAGR**: 연평균 복리 수익률
- **Volatility**: 연간 변동성
- **Sharpe Ratio**: 위험 대비 수익
- **Max Drawdown**: 최대 낙폭
- **Win Rate**: 승률

## ⚠️ 주의사항

### 백테스팅 한계
1. **과거 성과 ≠ 미래 수익**: 백테스트 결과가 미래 수익을 보장하지 않습니다
2. **생존 편향**: 상장폐지된 ETF는 포함되지 않음
3. **시장 충격**: 대량 거래시 가격 영향 미반영
4. **유동성**: ETF 유동성 제약 미반영

### 실전 투자 전
1. ✅ 모의투자로 충분히 테스트
2. ✅ 소액으로 시작
3. ✅ 리스크 관리 철저히
4. ✅ 정기적인 모니터링
5. ✅ 세금 영향 고려

## 🔄 업데이트 계획

- [ ] 실시간 모니터링 대시보드
- [ ] 텔레그램 알림 연동
- [ ] 더 많은 전략 추가
- [ ] 포트폴리오 최적화 (평균-분산, 리스크 패리티)
- [ ] 백테스팅 성능 개선

## 📚 참고 자료

- [한국투자증권 API 문서](https://apiportal.koreainvestment.com/)
- [ETF 정보 - 한국거래소](http://www.krx.co.kr/main/main.jsp)
- [포트폴리오 이론](https://en.wikipedia.org/wiki/Modern_portfolio_theory)

## 📄 라이선스

MIT License

## 💬 문의

이슈나 질문은 GitHub Issues를 통해 남겨주세요.

---

**면책조항**: 이 프로젝트는 교육 목적으로 제공됩니다. 투자 결정은 본인의 판단과 책임하에 이루어져야 합니다.
