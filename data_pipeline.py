"""
수집 기간 : 2018-01-01 ~ 2025-06-30
분할 기준 : Train 60% / Val 20% / Test 20%  (시계열 순 분할)
지표      : MA(20,60), RSI(14), MACD, 실현변동성(20d, 60d)
타임존    : UTC → KST (UTC+9) 변환
"""

import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import pytz
from datetime import datetime
import os

TICKERS = {
    "005930.KS": "Samsung Electronics (IT)",
    "009540.KS": "HD Korea Shipbuilding (Heavy Industry)",
    "034020.KS": "Doosan Enerbility (Industrials)",
    "105560.KS": "KB Financial (Financials)",
    "005380.KS": "Hyundai Motor (Consumer Disc.)",
    "005490.KS": "POSCO Holdings (Energy/Chem)",
    "207940.KS": "Samsung Biologics (Healthcare)",
    "090430.KS": "AmorePacific (Consumer Staples)",
    "035420.KS": "NAVER (Communication)",
}

START_DATE = "2018-01-01"
END_DATE   = "2025-06-30"
KST        = pytz.timezone("Asia/Seoul")

SPLIT_DATES = {
    "train_end": "2022-06-30",   # 2018-01-01 ~ 2022-06-30  (약 60%)
    "val_end"  : "2023-12-31",   # 2022-07-01 ~ 2023-12-31  (약 20%)
    # test      : 2024-01-01 ~ 2025-06-30            (약 20%)
}

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. 데이터 수집

def fetch_ohlcv(tickers: dict, start: str, end: str) -> pd.DataFrame:
    """yfinance로 OHLCV 수집 후 KST 변환, MultiIndex DataFrame 반환"""
    print("=" * 60)
    print(f"[1] OHLCV 수집  {start} ~ {end}")
    print("=" * 60)

    raw = yf.download(
        tickers=list(tickers.keys()),
        start=start,
        end=end,
        auto_adjust=True,   # 수정주가 사용 (배당·분할 반영)
        progress=False,
    )

    if raw.empty:
        raise ValueError("데이터 수집 실패")

    if raw.index.tzinfo is None:
        # tz-naive인 경우: UTC로 간주 후 KST 변환
        raw.index = raw.index.tz_localize("UTC").tz_convert(KST)
    else:
        raw.index = raw.index.tz_convert(KST)

    # date만 남기기
    raw.index = raw.index.normalize()
    raw.index.name = "Date"

    print(f"  수집 완료 : {len(raw):,}일 × {len(tickers)} 종목")
    print(f"  기간      : {raw.index[0].date()} ~ {raw.index[-1].date()}\n")
    return raw


# 2. 기술 지표 계산

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series,
          fast: int = 12, slow: int = 26, signal: int = 9
          ) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast   = series.ewm(span=fast,   adjust=False).mean()
    ema_slow   = series.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line= macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def compute_indicators(ohlcv: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """단일 종목의 OHLCV로 지표 계산"""
    close = ohlcv[("Close", ticker)]
    high  = ohlcv[("High",  ticker)]
    low   = ohlcv[("Low",   ticker)]
    vol   = ohlcv[("Volume",ticker)]

    df = pd.DataFrame(index=close.index)
    df["Close"]  = close
    df["High"]   = high
    df["Low"]    = low
    df["Volume"] = vol

    # 이동평균
    df["MA20"] = close.rolling(20).mean()
    df["MA60"] = close.rolling(60).mean()

    # 실현변동성 (20일·60일 — 일별 log return의 표준편차)
    log_ret       = (close / close.shift(1)).apply(lambda x: x).pipe(
        lambda s: s.apply(lambda v: __import__("math").log(v) if v > 0 else float("nan"))
    )
    df["Vol20"]   = log_ret.rolling(20).std()
    df["Vol60"]   = log_ret.rolling(60).std()

    # RSI(14)
    df["RSI14"]  = _rsi(close, 14)

    # MACD
    df["MACD"], df["MACD_signal"], df["MACD_hist"] = _macd(close)

    # 일별 수익률 (GA fitness 계산용)
    df["Return"] = close.pct_change()

    return df


def build_indicator_store(ohlcv: pd.DataFrame, tickers: dict) -> dict:
    """전 종목 지표 딕셔너리 반환  {ticker: DataFrame}"""
    print("[2] 기술 지표 계산")
    store = {}
    for ticker in tickers:
        try:
            store[ticker] = compute_indicators(ohlcv, ticker)
            print(f"  ✓ {ticker:12s} {tickers[ticker]}")
        except Exception as e:
            print(f"  ✗ {ticker:12s} 오류: {e}")
    print()
    return store


# 3. Train / Val / Test 분할

def split_data(store: dict, split_dates: dict) -> dict:
    print("[3] Train / Val / Test 분할")
    splits = {ticker: {} for ticker in store}

    for ticker, df in store.items():
        splits[ticker]["train"] = df[df.index <= split_dates["train_end"]]
        splits[ticker]["val"]   = df[(df.index >  split_dates["train_end"]) &
                                     (df.index <= split_dates["val_end"])]
        splits[ticker]["test"]  = df[df.index >  split_dates["val_end"]]

    # 분할 크기 (첫 종목 기준)
    sample = splits[list(store.keys())[0]]
    for name in ("train", "val", "test"):
        d = sample[name]
        if len(d):
            print(f"  {name:5s}: {d.index[0].date()} ~ {d.index[-1].date()}  "
                  f"({len(d):,}일)")
    print()
    return splits


# 4. Greedy Signal 계산
def compute_greedy_signals(df: pd.DataFrame) -> pd.DataFrame:
    # Composite = 0.35×Trend + 0.40×Volatility + 0.25×Momentum
    out = df.copy()

    # Trend signal  : 20MA > 60MA → +1, else -1
    out["sig_trend"] = (out["MA20"] > out["MA60"]).map({True: 1, False: -1})

    # Volatility signal : 20d_vol < 1.3 × 60d_vol → +1, else -1
    out["sig_vol"]   = (out["Vol20"] < 1.3 * out["Vol60"]).map({True: 1, False: -1})

    # Momentum signal : RSI∈[45,70] AND MACD_hist>0 → +1, else -1
    rsi_ok  = out["RSI14"].between(45, 70)
    macd_ok = out["MACD_hist"] > 0
    out["sig_mom"] = ((rsi_ok) & (macd_ok)).map({True: 1, False: -1})

    # 복합 점수
    out["composite"] = (0.35 * out["sig_trend"] +
                        0.40 * out["sig_vol"]   +
                        0.25 * out["sig_mom"])

    # 전략 선택
    def _strategy(row):
        # Volatility override
        if row["sig_vol"] == -1:
            return "Defensive"
        if row["composite"] > 0.20:
            return "Trend"
        elif row["composite"] >= -0.20:
            return "MeanReversion"
        else:
            return "Defensive"

    out["strategy"] = out.apply(_strategy, axis=1)
    return out


# 5. 데이터 저장
def save_data(store: dict, splits: dict, tickers: dict, output_dir: str):
    print("[4] 데이터 저장")

    for ticker, df in store.items():
        # Greedy 시그널 포함 전체 데이터
        df_with_signals = compute_greedy_signals(df)
        fname = f"{output_dir}/{ticker.replace('.', '_')}_full.csv"
        df_with_signals.to_csv(fname)
        print(f"  저장: {fname}")

    # 분할별 Close price 요약 (GA 학습용)
    for split_name in ("train", "val", "test"):
        close_dict = {}
        for ticker in tickers:
            s = splits[ticker][split_name]["Close"]
            if not s.empty:
                close_dict[ticker] = s
        combined = pd.DataFrame(close_dict)
        combined.index = combined.index.date   # tz 제거 후 date만
        fname = f"{output_dir}/close_{split_name}.csv"
        combined.to_csv(fname)
        print(f"  저장: {fname}")

    # 일별 수익률 (train만 — GA fitness용)
    ret_dict = {t: splits[t]["train"]["Return"] for t in tickers
                if not splits[t]["train"].empty}
    ret_df = pd.DataFrame(ret_dict)
    ret_df.index = ret_df.index.date
    ret_df.to_csv(f"{output_dir}/returns_train.csv")
    print(f"  저장: {output_dir}/returns_train.csv\n")


# 6. 요약 출력
def print_summary(store: dict, tickers: dict):
    print("=" * 60)
    print("[요약] 종목별 데이터 품질 체크")
    print("=" * 60)
    print(f"{'Ticker':14s} {'이름':35s} {'총일수':>6s} {'결측(Close)':>10s} {'결측률':>7s}")
    print("-" * 60)
    for ticker, df in store.items():
        total   = len(df)
        missing = df["Close"].isna().sum()
        pct     = missing / total * 100 if total else 0
        flag    = " ⚠" if pct > 5 else ""
        print(f"{ticker:14s} {tickers[ticker]:35s} {total:6,} {missing:10,} {pct:6.1f}%{flag}")
    print()

def run_pipeline():
    ohlcv  = fetch_ohlcv(TICKERS, START_DATE, END_DATE)
    store  = build_indicator_store(ohlcv, TICKERS)
    splits = split_data(store, SPLIT_DATES)
    print_summary(store, TICKERS)
    save_data(store, splits, TICKERS, OUTPUT_DIR)
    print("파이프라인 완료: data/ 폴더를 확인하세요.")
    return store, splits


if __name__ == "__main__":
    store, splits = run_pipeline()
