"""
전략 종류:
  Trend        : 20MA > 60MA 골든크로스 추세 추종
  MeanReversion: 가격이 20MA 대비 -5% 이하일 때 평균 회귀 진입
  Defensive    : 신규 진입 없음, 기존 포지션만 Optimal Stopping에 위임

전략 전환 조건:
  - 새 시그널이 2일 연속 유지될 것
  - 기대 5일 수익이 거래비용(0.20%) 초과할 것
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Literal
import os

Strategy = Literal["Trend", "MeanReversion", "Defensive"]

@dataclass
class GreedyConfig:
    # 시그널 가중치
    w_trend : float = 0.35
    w_vol   : float = 0.40
    w_mom   : float = 0.25

    # 전략 경계값
    threshold_high: float =  0.20   # score >  0.20 → Trend
    threshold_low : float = -0.20   # score < -0.20 → Defensive

    # 변동성 임계 배수 (Vol20 < ratio × Vol60 → 안정)
    vol_ratio: float = 1.30

    # RSI 정상 범위
    rsi_low : float = 45.0
    rsi_high: float = 70.0

    # 전략 전환 조건
    switch_confirm_days: int   = 2      # 연속 유지 일수
    switch_cost_pct    : float = 0.002  # 거래비용 0.20%
    expected_days      : int   = 5      # 기대 수익 계산 기간


@dataclass
class DailySignal:
    date        : pd.Timestamp
    sig_trend   : int          # +1 or -1
    sig_vol     : int          # +1 or -1
    sig_mom     : int          # +1 or -1
    composite   : float
    vol_override: bool         # 변동성 강제 Defensive 여부
    strategy    : Strategy
    ma20        : float
    ma60        : float
    rsi14       : float
    macd_hist   : float
    vol20       : float
    vol60       : float


class GreedyStrategySelector:
    def __init__(self, df: pd.DataFrame, config: GreedyConfig = None):
        """
        Parameters
        ----------
        df : pd.DataFrame
            data_pipeline 에서 생성된 종목 DataFrame
            (Close, MA20, MA60, Vol20, Vol60, RSI14, MACD_hist 컬럼 필요)
        config : GreedyConfig (선택)
        """
        self.df     = df.copy()
        self.cfg    = config or GreedyConfig()
        self._validate()

    def _validate(self):
        required = {"Close", "MA20", "MA60", "Vol20", "Vol60", "RSI14", "MACD_hist"}
        missing  = required - set(self.df.columns)
        if missing:
            raise ValueError(f"누락된 컬럼: {missing}")

    def _trend_signal(self, row) -> int:
        """20MA > 60MA → +1 (상승추세), else -1"""
        if pd.isna(row["MA20"]) or pd.isna(row["MA60"]):
            return -1
        return 1 if row["MA20"] > row["MA60"] else -1

    def _vol_signal(self, row) -> int:
        """20d 변동성 < 1.3 × 60d 변동성 → +1 (안정), else -1 (고변동)"""
        if pd.isna(row["Vol20"]) or pd.isna(row["Vol60"]) or row["Vol60"] == 0:
            return -1
        return 1 if row["Vol20"] < self.cfg.vol_ratio * row["Vol60"] else -1

    def _mom_signal(self, row) -> int:
        """RSI∈[45,70] AND MACD_hist>0 → +1 (모멘텀 양호), else -1"""
        if pd.isna(row["RSI14"]) or pd.isna(row["MACD_hist"]):
            return -1
        rsi_ok  = self.cfg.rsi_low <= row["RSI14"] <= self.cfg.rsi_high
        macd_ok = row["MACD_hist"] > 0
        return 1 if (rsi_ok and macd_ok) else -1

    def _composite(self, s_trend: int, s_vol: int, s_mom: int) -> float:
        cfg = self.cfg
        return cfg.w_trend * s_trend + cfg.w_vol * s_vol + cfg.w_mom * s_mom

    def _raw_strategy(self, composite: float, vol_override: bool) -> Strategy:
        """점수 → 전략 매핑 (전환 안정성 고려 전 단계)"""
        if vol_override:
            return "Defensive"
        if composite > self.cfg.threshold_high:
            return "Trend"
        elif composite >= self.cfg.threshold_low:
            return "MeanReversion"
        else:
            return "Defensive"

    def _apply_switch_filter(
        self,
        raw_strategies : pd.Series,   # 날짜별 raw 전략
        returns        : pd.Series,   # 일별 수익률
    ) -> pd.Series:
        """
        전환 조건:
          1) 새 전략이 switch_confirm_days 일 연속 유지
          2) 새 전략의 과거 expected_days 평균 수익 > 거래비용
        조건 미달 시 이전 전략 유지.
        """
        cfg      = self.cfg
        result   = raw_strategies.copy()
        current  = raw_strategies.iloc[0]
        pending  = None
        pending_count = 0

        # 기존: Brute-Force -> N일 동안 매일 K일치 배열을 잘라내고 더해서 평균 계산 O(N*K)
        # 수정: Sliding Window -> 루프 진입 전 평균 배열 계산. 메인 루프에서는 인덱스 조회 O(N)
        expected_returns = returns.rolling(window=cfg.expected_days).mean()

        for i in range(1, len(raw_strategies)):
            new_strat = raw_strategies.iloc[i]

            if new_strat == current:
                pending       = None
                pending_count = 0
                continue

            # 새 전략 후보
            if new_strat != pending:
                pending       = new_strat
                pending_count = 1
            else:
                pending_count += 1

            # 조건 1: 연속 확인
            if pending_count < cfg.switch_confirm_days:
                result.iloc[i] = current
                continue

            # 조건 2: 기대 수익 > 거래비용
            past_ret = expected_returns.iloc[i]
            
            if pd.isna(past_ret) or past_ret <= cfg.switch_cost_pct:
                result.iloc[i] = current
                continue

            # 전환 승인
            current       = pending
            pending       = None
            pending_count = 0

        return result

    def run(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            원본 컬럼 + 시그널 컬럼 + strategy 컬럼
        """
        df  = self.df
        out = df.copy()

        out["sig_trend"]    = df.apply(self._trend_signal, axis=1)
        out["sig_vol"]      = df.apply(self._vol_signal,   axis=1)
        out["sig_mom"]      = df.apply(self._mom_signal,   axis=1)
        out["composite"]    = out.apply(
            lambda r: self._composite(r["sig_trend"], r["sig_vol"], r["sig_mom"]),
            axis=1
        )
        out["vol_override"] = out["sig_vol"] == -1

        out["strategy_raw"] = out.apply(
            lambda r: self._raw_strategy(r["composite"], r["vol_override"]),
            axis=1
        )

        ret = df["Return"] if "Return" in df.columns else df["Close"].pct_change()
        out["strategy"] = self._apply_switch_filter(out["strategy_raw"], ret)

        return out

    def entry_signal(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        Trend        : 현재 전략이 Trend이고, 전일 strategy_raw가 Trend가 아니었던 첫날
                       (골든크로스 발생 시점)
        MeanReversion: 현재 Close가 MA20 대비 -5% 이하
        Defensive    : 진입 없음 (entry = False)
        """
        df = result.copy()

        # Trend 진입: 골든크로스 발생 여부 (20MA가 60MA를 상향 돌파)
        prev_above = (df["MA20"].shift(1) <= df["MA60"].shift(1))
        curr_above = (df["MA20"] > df["MA60"])
        golden_cross = prev_above & curr_above

        # MeanReversion 진입: Close < MA20 * 0.95
        mean_rev_entry = df["Close"] < df["MA20"] * 0.95

        def _entry(row):
            strat = row["strategy"]
            idx   = row.name
            if strat == "Trend":
                return bool(golden_cross.loc[idx])
            elif strat == "MeanReversion":
                return bool(mean_rev_entry.loc[idx])
            else:
                return False

        df["entry"] = df.apply(_entry, axis=1)
        return df

class PortfolioGreedySelector:
    def __init__(self, store: dict, config: GreedyConfig = None):
        self.store  = store
        self.cfg    = config or GreedyConfig()

    def run(self) -> dict:
        results = {}
        for ticker, df in self.store.items():
            try:
                sel             = GreedyStrategySelector(df, self.cfg)
                res             = sel.run()
                res             = sel.entry_signal(res)
                results[ticker] = res
            except Exception as e:
                print(f"  ✗ {ticker}: {e}")
        return results

    def strategy_summary(self, results: dict) -> pd.DataFrame:
        """
        날짜 × 종목 형태의 전략 요약 테이블.
        각 셀: 'T'(Trend) / 'M'(MeanReversion) / 'D'(Defensive)
        """
        abbrev = {"Trend": "T", "MeanReversion": "M", "Defensive": "D"}
        frames = {}
        for ticker, df in results.items():
            frames[ticker] = df["strategy"].map(abbrev)
        return pd.DataFrame(frames)

    def composite_scores(self, results: dict) -> pd.DataFrame:
        """날짜별 복합 점수 테이블"""
        return pd.DataFrame({t: df["composite"] for t, df in results.items()})

def strategy_distribution(results: dict) -> pd.DataFrame:
    """전략별 선택 비율 요약"""
    rows = []
    for ticker, df in results.items():
        counts = df["strategy"].value_counts()
        total  = len(df)
        rows.append({
            "ticker"       : ticker,
            "Trend_%"      : round(counts.get("Trend",         0) / total * 100, 1),
            "MeanRev_%"    : round(counts.get("MeanReversion", 0) / total * 100, 1),
            "Defensive_%"  : round(counts.get("Defensive",     0) / total * 100, 1),
            "VolOverride_%": round(df["vol_override"].sum()        / total * 100, 1),
            "Entries"      : int(df.get("entry", pd.Series(False)).sum()),
        })
    return pd.DataFrame(rows).set_index("ticker")


def strategy_switch_count(results: dict) -> pd.Series:
    """종목별 전략 전환 횟수"""
    counts = {}
    for ticker, df in results.items():
        s = df["strategy"]
        counts[ticker] = (s != s.shift(1)).sum() - 1   # 첫날 제외
    return pd.Series(counts, name="전략전환횟수")

def save_greedy_results(results: dict, output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    for ticker, df in results.items():
        cols = ["Close", "MA20", "MA60", "RSI14", "MACD_hist",
                "Vol20", "Vol60", "sig_trend", "sig_vol", "sig_mom",
                "composite", "vol_override", "strategy_raw", "strategy", "entry"]
        save_cols = [c for c in cols if c in df.columns]
        fname = f"{output_dir}/{ticker.replace('.', '_')}_greedy.csv"
        df[save_cols].to_csv(fname)
    print(f"  ✓ Greedy 결과 저장 완료 → {output_dir}/")


if __name__ == "__main__":
    from data_pipeline import run_pipeline
    
    print("=" * 60)
    print("Greedy 전략 선택")
    print("=" * 60)

    store, _ = run_pipeline()

    pg = PortfolioGreedySelector(store)
    results = pg.run()

    print("\n[종목별 전략 선택 비율]")
    dist = strategy_distribution(results)
    print(dist.to_string())

    print("\n[종목별 전략 전환 횟수]")
    sw = strategy_switch_count(results)
    print(sw.to_string())

    save_greedy_results(results)
    print("\nGreedy 분석 완료! data/ 폴더를 확인하세요.")