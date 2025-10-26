# ETFTrading 데이터 소스 통합 가이드

## 📊 현재 상황 분석 (2025-10-26)

### 데이터 소스 테스트 결과

| 데이터 소스 | 인증 필요 | 상태 | 비고 |
|------------|----------|------|------|
| **KIS API** | ✓ (App Key + Secret) | 🔴 403 Forbidden | 실제 API 키 필요 |
| **Yahoo Finance** | ✗ | 🔴 403 Forbidden | 한국 ETF 지원 안함 |
| **pykrx** | ✗ | 🟡 설치됨 | KRX API 변경으로 현재 불안정 |
| **FinanceDataReader** | ✗ | 🟡 설치됨 | 한국 ETF 데이터 제한적 |
| **Naver Finance** | ✗ | 🔴 403 Forbidden | 크롤링 방지 |

### 결론

**한국 ETF 데이터를 무료로 안정적으로 가져오는 공개 API는 현재 제한적입니다.**

---

## 🔧 사용 가능한 옵션

### 옵션 1: 한국투자증권 API 사용 (권장)

**장점:**
- ✅ 공식 API, 안정적
- ✅ 실시간 + 과거 데이터
- ✅ 무료 (일 트랜잭션 제한 있음)

**필요 사항:**
1. 한국투자증권 계좌 개설
2. API 서비스 신청
3. App Key + App Secret 발급

**설정 방법:**

```bash
# 1. secrets.yaml 파일 생성
cd ETFTrading/config
cp secrets.yaml.example secrets.yaml

# 2. 발급받은 키 입력
# secrets.yaml:
kis:
  mock:  # 모의투자 (테스트용)
    app_key: "YOUR_MOCK_APP_KEY"
    app_secret: "YOUR_MOCK_APP_SECRET"
    account_number: "YOUR_ACCOUNT_NUMBER"
    account_code: "01"
  real:  # 실전투자
    app_key: "YOUR_REAL_APP_KEY"
    app_secret: "YOUR_REAL_APP_SECRET"
    account_number: "YOUR_ACCOUNT_NUMBER"
    account_code: "01"
```

**사용 예시:**

```python
from ETFTrading.data.krx_loader import KoreanETFLoader

# 자동으로 KIS API 사용
loader = KoreanETFLoader()
data = loader.load_multiple(
    tickers=["069500", "360750", "152380", "132030"],
    start_date="2020-01-01",
    end_date="2024-10-26",
    source="kis",  # 또는 "auto"
    use_cache=True
)
```

---

### 옵션 2: 시뮬레이션 데이터 사용 (현재 기본값)

**장점:**
- ✅ 즉시 사용 가능
- ✅ 재현 가능한 백테스트
- ✅ 인증 불필요

**단점:**
- ❌ 실제 시장 데이터 아님
- ❌ 현실적 파라미터 기반 생성 데이터

**현재 백테스트 리포트:**

```
BACKTEST_REPORT.md에 명시:
"데이터: 시뮬레이션 데이터 (현실적 파라미터 기반)"
```

백테스트 결과는 **교육/데모 목적**으로만 사용.

---

### 옵션 3: 캐시된 데이터 활용

**사용 방법:**

```bash
# 캐시 디렉토리 확인
ls -la ETFTrading/data/cache/

# 캐시된 데이터 사용
python run_backtest.py  # 자동으로 캐시 우선 사용
```

캐시가 있으면 자동으로 재사용, 없으면 API 호출.

---

### 옵션 4: CSV 파일 업로드

**수동으로 데이터 준비:**

```bash
# 1. 데이터 준비 (예: Excel에서 저장)
# 파일 형식: ticker.csv
# 컬럼: date,open,high,low,close,volume

# 2. 캐시 디렉토리에 저장
cp your_data/069500.csv ETFTrading/data/cache/069500.parquet

# 3. 백테스트 실행
python run_backtest.py
```

---

## 🔄 현재 워크플로우와 통합

### 통합된 데이터 로더 (`krx_loader.py`)

새로 만든 `KoreanETFLoader`는 **자동으로 최적의 데이터 소스 선택**:

```python
from ETFTrading.data.krx_loader import KoreanETFLoader

loader = KoreanETFLoader()

# 자동 선택 (pykrx → fdr → kis → cache 순서)
data = loader.get_etf_ohlcv(
    ticker="069500",
    start_date="2024-01-01",
    end_date="2024-10-26",
    source="auto"  # 자동 선택
)

# 또는 직접 지정
data = loader.get_etf_ohlcv(
    ticker="069500",
    start_date="2024-01-01",
    end_date="2024-10-26",
    source="kis"  # KIS API 강제 사용
)
```

### `run_backtest.py` 수정

기존 `yahoo_loader.py` 대신 `krx_loader.py` 사용:

```python
# 기존 코드 (run_backtest.py 라인 29):
# from ETFTrading.data.yahoo_loader import YahooETFLoader

# 새 코드:
from ETFTrading.data.krx_loader import KoreanETFLoader

# 기존 코드 (라인 71):
# loader = YahooETFLoader()

# 새 코드:
loader = KoreanETFLoader()
data = loader.load_multiple(
    tickers=list(all_tickers),
    start_date=START_DATE,
    end_date=END_DATE,
    source="auto",  # 자동으로 최적의 소스 선택
    use_cache=True
)
```

---

## 🚀 실행 방법

### 방법 1: KIS API 사용 (실제 데이터)

```bash
# 1. API 키 설정
vim ETFTrading/config/secrets.yaml

# 2. 백테스트 실행
cd /home/user/SystemTrading
source setup.sh
python ETFTrading/run_backtest.py
```

### 방법 2: 시뮬레이션 데이터 사용

```bash
# 현재 이미 실행된 백테스트 결과 확인
cat ETFTrading/BACKTEST_REPORT.md

# 결과 시각화 확인
ls -la ETFTrading/backtest_results.png
```

---

## 📝 통합 체크리스트

### 단계 1: 데이터 소스 선택

- [ ] **옵션 A**: KIS API 키 발급 (실제 데이터)
- [ ] **옵션 B**: 시뮬레이션 데이터 사용 (현재 기본값)
- [ ] **옵션 C**: CSV 파일 업로드
- [ ] **옵션 D**: 캐시된 데이터 재사용

### 단계 2: 코드 통합

- [x] `krx_loader.py` 생성 완료
- [ ] `run_backtest.py` 수정 (yahoo_loader → krx_loader)
- [ ] 데이터 소스 설정
- [ ] 테스트 실행

### 단계 3: 백테스트 실행

- [ ] 환경 활성화 (`source setup.sh`)
- [ ] 데이터 확인
- [ ] 백테스트 실행
- [ ] 결과 확인

---

## ⚠️ 주의사항

### pykrx 불안정 이슈

현재 pykrx가 KRX 웹사이트 구조 변경으로 불안정합니다.
공식 이슈: https://github.com/sharebook-kr/pykrx/issues

**해결 방법:**
1. pykrx 최신 버전 업데이트 대기
2. KIS API 사용 (권장)
3. FinanceDataReader 대체 사용

### API 사용 제한

**KIS API 제한:**
- 1초당 20건
- 1일 30,000건

백테스트 시 캐싱 필수!

---

## 🎯 권장 사항

### 교육/테스트 목적
✅ **현재 시뮬레이션 데이터 사용**
- 이미 백테스트 완료
- 결과 확인 가능 (`BACKTEST_REPORT.md`)

### 실전 투자 목적
✅ **KIS API 사용**
- 실제 시장 데이터 필요
- API 키 발급 필수
- 모의투자로 먼저 테스트

---

## 📚 참고 링크

- [한국투자증권 API 신청](https://apiportal.koreainvestment.com/)
- [pykrx 문서](https://github.com/sharebook-kr/pykrx)
- [FinanceDataReader 문서](https://github.com/FinanceData/FinanceDataReader)

---

**작성일**: 2025-10-26
**작성자**: Claude Code
**목적**: ETFTrading 시스템 데이터 소스 통합 가이드
