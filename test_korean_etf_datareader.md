# 한국 ETF 가격 데이터 수집 테스트 결과

## 테스트 일자
2025-10-27

## 테스트 환경
- Python 3.11
- Linux 환경

## 테스트한 라이브러리

### 1. FinanceDataReader
**결과: ❌ 실패**

**문제점:**
- Naver API 접근 실패 (JSONDecodeError)
- ETF 목록 및 가격 데이터 모두 가져오기 실패
- 한국 주식(삼성전자) 데이터도 가져오기 실패

**오류 메시지:**
```
requests.exceptions.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
"069500" invalid symbol or has no data
```

**분석:**
- Naver Finance API의 변경 또는 네트워크 접근 제한
- 테스트 환경에서 Naver 서비스 접근 불가

### 2. pykrx
**결과: ❌ 실패**

**문제점:**
- KRX 웹사이트 데이터 파싱 실패
- 모든 API 호출이 빈 DataFrame 반환
- 티커 목록, 가격 데이터 모두 가져오기 실패

**오류 메시지:**
```
KeyError: '시장'
```

**분석:**
- KRX 웹사이트 구조 변경 가능성
- 테스트 환경에서 KRX 웹사이트 접근 불가 또는 데이터 파싱 실패

### 3. yfinance
**결과: ⚠️ 부분 성공 (의존성 문제)**

**문제점:**
- 핵심 의존성 `multitasking` 설치 실패
- setuptools 호환성 문제로 wheel 빌드 실패

**오류 메시지:**
```
AttributeError: install_layout. Did you mean: 'install_platlib'?
Failed building wheel for multitasking
```

**잠재적 해결 방법:**
1. 다른 환경에서 테스트 (conda 환경 또는 다른 setuptools 버전)
2. multitasking을 GitHub에서 직접 설치
3. yfinance 구버전 사용

**yfinance 사용법 (정상 설치 시):**
```python
import yfinance as yf

# 한국 ETF 티커 형식: 코드.KS (코스피) 또는 코드.KQ (코스닥)
etf = yf.Ticker('069500.KS')  # KODEX 200
data = etf.history(start='2024-01-01', end='2024-12-31')

# 데이터 컬럼: Open, High, Low, Close, Volume, Dividends, Stock Splits
print(data.head())
```

**주요 한국 ETF 티커:**
- KODEX 200: 069500.KS
- KODEX 레버리지: 122630.KS
- TIGER 200: 102110.KS
- KODEX 인버스: 114800.KS
- KODEX 코스닥150: 229200.KS

## 결론 및 권장사항

### 현재 상황
현재 테스트 환경에서는 모든 라이브러리가 제대로 작동하지 않습니다:
1. **FinanceDataReader**: Naver API 접근 불가
2. **pykrx**: KRX 웹사이트 데이터 파싱 실패
3. **yfinance**: 의존성 설치 문제

### 권장사항

#### 옵션 1: yfinance (가장 권장)
**장점:**
- Yahoo Finance는 글로벌 서비스로 안정적
- 한국 ETF 데이터 제공
- 다양한 국가/시장 데이터 통합 가능
- API 키 불필요

**다음 환경에서 재시도:**
```bash
# conda 환경 사용
conda create -n etf python=3.10
conda activate etf
pip install yfinance

# 또는 가상환경 사용
python -m venv venv
source venv/bin/activate
pip install yfinance
```

#### 옵션 2: pykrx (한국 전용 데이터)
**장점:**
- 한국거래소 공식 데이터
- 한국 시장에 특화된 기능
- API 키 불필요

**재시도 방법:**
```python
from pykrx import stock

# 다른 네트워크 환경에서 테스트
# 또는 pykrx 최신 버전 확인
pip install --upgrade pykrx
```

#### 옵션 3: FinanceDataReader
**재시도 방법:**
```bash
# 최신 버전 설치
pip install --upgrade finance-datareader

# 또는 개발 버전
pip install git+https://github.com/FinanceData/FinanceDataReader.git
```

#### 옵션 4: 직접 API 사용
**KRX API 또는 증권사 API 직접 사용:**
- KRX Open API
- 한국투자증권 Open API
- 키움증권 Open API

### 다음 단계

1. **로컬 환경에서 yfinance 테스트**
   ```bash
   pip install yfinance
   python test_yfinance_korean_etf.py
   ```

2. **정상 작동 확인 후 프로젝트 통합**
   - `data/` 디렉토리에 한국 ETF data loader 추가
   - 기존 Binance data loader와 유사한 구조로 구현

3. **캐싱 및 데이터 검증 추가**
   - 다운로드한 데이터를 parquet 형식으로 저장
   - 데이터 품질 검증 로직 추가

## 테스트 코드 예시

### yfinance 테스트 코드
```python
import yfinance as yf
import pandas as pd

def test_korean_etf():
    """한국 ETF 데이터 가져오기 테스트"""

    # KODEX 200
    ticker = '069500.KS'
    etf = yf.Ticker(ticker)

    # 최근 3개월 데이터
    data = etf.history(start='2024-08-01', end='2024-10-31')

    print(f"데이터 shape: {data.shape}")
    print(f"기간: {data.index[0]} ~ {data.index[-1]}")
    print("\n최근 5일:")
    print(data.tail())

    return data

if __name__ == '__main__':
    df = test_korean_etf()
```

### 저장된 테스트 결과
이 문서는 `/home/user/SystemTrading/test_korean_etf_datareader.md`에 저장되었습니다.
