"""
Secretary Problem을 20일 보유 윈도우에 적용.

윈도우 구조:
  Day  1~7  (Observation) : 최고가 P* 기록, 매도 안 함
  Day  8~20 (Selection)   : 첫 번째 Close > P* 발생 시 매도
              Day 20 도달  : 강제 청산

오버라이드 규칙 (전체 보유 기간 내내 유효):
  Stop-Loss   : 수익률 <= -8%  → 즉시 청산
  Take-Profit : 수익률 >= +25% → 즉시 청산
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Literal, Optional
import os

ExitReason = Literal[
    "secretary_trigger",   # P* 초과 → Secretary 조건 충족
    "force_liquidate",     # Day 20 강제 청산
    "stop_loss",           # -8% 손절
    "take_profit",         # +25% 익절
]

@dataclass
class StoppingConfig:
    window       : int   = 20      # 전체 보유 윈도우 (거래일)
    observe_days : int   = 7       # 관찰 구간 (Day 1~7)
    stop_loss    : float = -0.08   # 손절 기준 수익률
    take_profit  : float =  0.25   # 익절 기준 수익률

    @property
    def select_start(self) -> int:
        """선택 구간 시작일 (1-indexed)"""
        return self.observe_days + 1   # Day 8


@dataclass
class TradeResult:
    ticker      : str
    entry_date  : pd.Timestamp
    entry_price : float
    exit_date   : pd.Timestamp
    exit_price  : float
    exit_reason : ExitReason
    hold_days   : int              # 실제 보유 거래일 수
    p_star      : float            # 관찰 구간 최고가
    return_pct  : float            # (exit - entry) / entry

    def __str__(self):
        sign = "+" if self.return_pct >= 0 else ""
        return (
            f"[{self.ticker}] "
            f"{self.entry_date.date()} → {self.exit_date.date()} "
            f"({self.hold_days}일) | "
            f"진입 {self.entry_price:,.0f} → 청산 {self.exit_price:,.0f} | "
            f"수익률 {sign}{self.return_pct*100:.2f}% | "
            f"사유: {self.exit_reason}"
        )


class OptimalStopper:
    def __init__(
        self,
        prices      : pd.Series,        # 종목 전체 Close 시계열
        entry_date  : pd.Timestamp,
        entry_price : float,
        ticker      : str = "UNKNOWN",
        config      : StoppingConfig = None,
    ):
        self.prices       = prices
        self.entry_date   = entry_date
        self.entry_price  = entry_price
        self.ticker       = ticker
        self.cfg          = config or StoppingConfig()

        # 진입일 이후 최대 window 거래일 슬라이스
        future = prices.loc[entry_date:]
        self.window_prices = future.iloc[: self.cfg.window]

    def _return(self, price: float) -> float:
        return (price - self.entry_price) / self.entry_price

    def run(self) -> TradeResult:
        cfg    = self.cfg
        wp     = self.window_prices

        if len(wp) == 0:
            raise ValueError(f"진입일({self.entry_date}) 이후 가격 데이터 없음")

        # 관찰 구간 : Day 1 ~ observe_days 
        observe = wp.iloc[: cfg.observe_days]
        select  = wp.iloc[cfg.observe_days :]      # Day (observe+1) ~ 20

        # 관찰 구간 최고가 P*
        # (관찰 구간보다 데이터가 짧으면 있는 것만 사용)
        p_star = observe.max() if len(observe) > 0 else self.entry_price

        # 관찰 구간 내 Stop-loss / Take-profit 체크
        for day_idx, (date, price) in enumerate(observe.items(), start=1):
            ret = self._return(price)
            if ret <= cfg.stop_loss:
                return self._make_result(date, price, "stop_loss", day_idx, p_star)
            if ret >= cfg.take_profit:
                return self._make_result(date, price, "take_profit", day_idx, p_star)

        # 선택 구간 : Day (observe+1) ~ window 
        for day_idx, (date, price) in enumerate(select.items(),
                                                start=cfg.observe_days + 1):
            ret = self._return(price)

            # 오버라이드 우선 체크
            if ret <= cfg.stop_loss:
                return self._make_result(date, price, "stop_loss", day_idx, p_star)
            if ret >= cfg.take_profit:
                return self._make_result(date, price, "take_profit", day_idx, p_star)

            # Secretary 조건: 현재가 > P*
            if price > p_star:
                return self._make_result(date, price, "secretary_trigger", day_idx, p_star)

        # Day 20 강제 청산
        last_date  = wp.index[-1]
        last_price = wp.iloc[-1]
        return self._make_result(last_date, last_price, "force_liquidate", len(wp), p_star)

    def _make_result(
        self,
        exit_date  : pd.Timestamp,
        exit_price : float,
        reason     : ExitReason,
        hold_days  : int,
        p_star     : float,
    ) -> TradeResult:
        return TradeResult(
            ticker      = self.ticker,
            entry_date  = self.entry_date,
            entry_price = self.entry_price,
            exit_date   = exit_date,
            exit_price  = exit_price,
            exit_reason = reason,
            hold_days   = hold_days,
            p_star      = p_star,
            return_pct  = self._return(exit_price),
        )

class PortfolioStopper:
    def __init__(
        self,
        price_store   : dict,          # {ticker: pd.Series(Close)}
        entry_signals : dict,          # {ticker: pd.DatetimeIndex} 진입일 목록
        config        : StoppingConfig = None,
    ):
        self.price_store   = price_store
        self.entry_signals = entry_signals
        self.cfg           = config or StoppingConfig()

    def run(self) -> list:
        all_trades = []
        for ticker, entry_dates in self.entry_signals.items():
            prices = self.price_store.get(ticker)
            if prices is None:
                print(f"  ✗ {ticker}: 가격 데이터 없음")
                continue
            for entry_date in entry_dates:
                try:
                    entry_price = prices.loc[entry_date]
                    stopper     = OptimalStopper(
                        prices, entry_date, entry_price, ticker, self.cfg
                    )
                    result = stopper.run()
                    all_trades.append(result)
                except Exception as e:
                    print(f"  ✗ {ticker} @ {entry_date}: {e}")
        return all_trades

    @staticmethod
    def summary(trades: list) -> pd.DataFrame:
        """TradeResult 리스트 → 요약 DataFrame"""
        if not trades:
            return pd.DataFrame()
        rows = []
        for t in trades:
            rows.append({
                "ticker"      : t.ticker,
                "entry_date"  : t.entry_date.date(),
                "exit_date"   : t.exit_date.date(),
                "hold_days"   : t.hold_days,
                "entry_price" : round(t.entry_price, 0),
                "exit_price"  : round(t.exit_price,  0),
                "p_star"      : round(t.p_star,       0),
                "return_pct"  : round(t.return_pct * 100, 2),
                "exit_reason" : t.exit_reason,
            })
        return pd.DataFrame(rows)


def trade_statistics(summary_df: pd.DataFrame) -> dict:
    if summary_df.empty:
        return {}

    ret = summary_df["return_pct"]
    reasons = summary_df["exit_reason"].value_counts().to_dict()

    win_mask = ret > 0
    stats = {
        "총 거래 수"        : len(summary_df),
        "승률 (%)"          : round(win_mask.mean() * 100, 1),
        "평균 수익률 (%)"   : round(ret.mean(), 2),
        "평균 보유일"       : round(summary_df["hold_days"].mean(), 1),
        "최대 수익 (%)"     : round(ret.max(), 2),
        "최대 손실 (%)"     : round(ret.min(), 2),
        "수익 거래 평균 (%)": round(ret[win_mask].mean(), 2) if win_mask.any() else 0,
        "손실 거래 평균 (%)": round(ret[~win_mask].mean(), 2) if (~win_mask).any() else 0,
        "청산 사유"         : reasons,
    }
    return stats


def exit_reason_breakdown(summary_df: pd.DataFrame) -> pd.DataFrame:
    """청산 사유별 수익률 분포"""
    if summary_df.empty:
        return pd.DataFrame()
    return (
        summary_df
        .groupby("exit_reason")["return_pct"]
        .agg(["count", "mean", "min", "max"])
        .rename(columns={"count": "건수", "mean": "평균(%)", "min": "최소(%)", "max": "최대(%)"})
        .round(2)
    )


def save_trade_results(summary_df: pd.DataFrame, output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    fname = f"{output_dir}/trade_results.csv"
    summary_df.to_csv(fname, index=False)
    print(f"  ✓ 거래 결과 저장 → {fname}")


if __name__ == "__main__":
    from greedy_strategy  import PortfolioGreedySelector
    from optimal_stopping import PortfolioStopper, trade_statistics

    store, _  = run_pipeline()
    pg        = PortfolioGreedySelector(store)
    results   = pg.run()

    # Greedy entry 시그널 → Optimal Stopping 입력
    price_store   = {t: df["Close"] for t, df in store.items()}
    entry_signals = {
         t: df[df["entry"] == True].index
        for t, df in results.items()
    }

    ps     = PortfolioStopper(price_store, entry_signals)
    trades = ps.run()
    print(trade_statistics(ps.summary(trades)))
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
