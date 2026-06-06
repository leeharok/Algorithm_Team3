"""
Secretary Problem을 60일 보유 윈도우에 적용.

윈도우 구조:
  Day  1~22 (Observation) : 최고가 P* 기록, 매도 안 함
  Day 23~60 (Selection)   : 첫 번째 Close > P* 발생 시 매도
              force_liquidate 없음 (60일 내 미청산 시 secretary_trigger 처리)

오버라이드 규칙 (전체 보유 기간 내내 유효):
  Stop-Loss   : 수익률 <= -12% → 즉시 청산
  Take-Profit : 수익률 >= +20% → 즉시 청산
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
    "stop_loss",           # -8% 손절
    "take_profit",         # +25% 익절
]

# ✅ 추가: 관찰 기간 결정 모드
ObserveMode = Literal[
    "fixed",    # 기존 고정 방식 (observe_days 그대로)
    "dynamic",  # 변동성/추세 기반 동적 조절 (이하록 아이디어 1)
    "ucb1",     # UCB1 알고리즘으로 최적 관찰 기간 탐색 (이하록 아이디어 2)
]


# ✅ 추가: Dynamic Observe 계산기
# 변동성(Vol20/Vol60)과 추세(MA20/MA60)를 보고 관찰 기간을 동적으로 결정
# - 시장 안정적(변동성 낮음 + 상승추세): 관찰 기간 길게 → P* 신뢰도 높임
# - 시장 급변(변동성 높음 + 하락추세): 관찰 기간 짧게 → 빠르게 대응
class DynamicObserveCalculator:
    def __init__(self, min_days: int = 7, max_days: int = 30):
        self.min_days = min_days   # 최소 관찰 기간 (급변장)
        self.max_days = max_days   # 최대 관찰 기간 (안정장)

    def calc(self, prices: pd.Series, entry_date: pd.Timestamp) -> int:
        """
        진입일 기준 과거 60일 데이터로 변동성/추세 계산
        → 관찰 기간(일수) 반환
        """
        hist = prices.loc[:entry_date].iloc[-60:]
        if len(hist) < 20:
            return 22   # 데이터 부족 시 기본값

        returns = hist.pct_change().dropna()
        vol20   = returns.iloc[-20:].std()
        vol60   = returns.std()
        ma20    = hist.iloc[-20:].mean()
        ma60    = hist.mean()

        # 변동성 점수: Vol20이 Vol60보다 클수록 불안정 (0~1)
        vol_ratio   = vol20 / (vol60 + 1e-9)
        vol_score   = min(1.0, max(0.0, (vol_ratio - 0.8) / 1.2))  # 0.8~2.0 → 0~1

        # 추세 점수: MA20 < MA60이면 하락추세 (0~1)
        trend_score = 0.0 if ma20 > ma60 else 1.0

        # 불안정 점수 (높을수록 관찰 기간 짧게)
        instability = 0.6 * vol_score + 0.4 * trend_score  # 0~1

        # 관찰 기간 = max ~ min 사이 선형 보간
        observe = int(self.max_days - instability * (self.max_days - self.min_days))
        return max(self.min_days, min(self.max_days, observe))


# ✅ 추가: UCB1 관찰 기간 선택기
# Arm = 관찰 기간 후보 [5, 10, 15, 22, 30]
# 각 Arm의 과거 수익률 평균 + 탐색 보너스로 최적 관찰 기간 선택
class UCB1ObserveSelector:
    ARMS = [5, 10, 15, 22, 30]   # 관찰 기간 후보 (일)

    def __init__(self):
        self.counts  = {a: 0   for a in self.ARMS}   # 각 Arm 선택 횟수
        self.rewards = {a: 0.0 for a in self.ARMS}   # 각 Arm 누적 수익률 합

    def select(self) -> int:
        """UCB1 공식으로 이번에 쓸 관찰 기간 선택"""
        total = sum(self.counts.values())

        # 한 번도 안 써본 Arm 있으면 먼저 써보기 (탐색)
        for arm in self.ARMS:
            if self.counts[arm] == 0:
                return arm

        # UCB1 점수 = 평균수익 + sqrt(2 * ln(total) / count)
        ucb_scores = {
            arm: (self.rewards[arm] / self.counts[arm])
                 + np.sqrt(2 * np.log(total) / self.counts[arm])
            for arm in self.ARMS
        }
        return max(ucb_scores, key=ucb_scores.get)

    def update(self, arm: int, reward: float):
        """거래 완료 후 해당 Arm의 수익률로 업데이트"""
        self.counts[arm]  += 1
        self.rewards[arm] += reward

@dataclass
class StoppingConfig:
    window       : int         = 60       # 전체 보유 윈도우 (거래일, 약 3달)
    observe_days : int         = 22       # 고정 모드 관찰 구간 (37% — Secretary 최적)
    stop_loss    : float       = -0.12    # 손절 기준 수익률 (-12%)
    take_profit  : float       =  0.20    # 익절 기준 수익률 (+20%)
    # ✅ 추가: 관찰 기간 모드 선택
    # "fixed"   → 기존 방식 (observe_days 고정)
    # "dynamic" → 변동성/추세 기반 자동 조절
    # "ucb1"    → UCB1 알고리즘으로 탐색
    observe_mode : ObserveMode = "fixed"

    @property
    def select_start(self) -> int:
        """선택 구간 시작일 (1-indexed)"""
        return self.observe_days + 1   # 고정 모드 기본값


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

        # 진입일 이후 전체 가격 사용 (window 무제한, 기준값 점감)
        self.window_prices = prices.loc[entry_date:]

        # ✅ 추가: 관찰 기간 결정 (모드에 따라 다르게)
        mode = self.cfg.observe_mode
        if mode == "fixed":
            self._observe_days = self.cfg.observe_days
        elif mode == "dynamic":
            calc = DynamicObserveCalculator(min_days=7, max_days=30)
            self._observe_days = calc.calc(prices, entry_date)
        elif mode == "ucb1":
            # UCB1 selector는 PortfolioStopper에서 주입받음 (없으면 fixed)
            self._observe_days = getattr(self, "_ucb1_observe_days", self.cfg.observe_days)
        else:
            self._observe_days = self.cfg.observe_days

    def _return(self, price: float) -> float:
        return (price - self.entry_price) / self.entry_price

    def run(self) -> TradeResult:
        cfg    = self.cfg
        wp     = self.window_prices

        if len(wp) == 0:
            raise ValueError(f"진입일({self.entry_date}) 이후 가격 데이터 없음")

        # 관찰 구간 : Day 1 ~ observe_days (모드에 따라 동적 결정)
        observe = wp.iloc[: self._observe_days]

        # 관찰 구간 최고가 P* 확정
        p_star = observe.max() if len(observe) > 0 else self.entry_price

        # 관찰 구간 내 Stop-loss / Take-profit 체크
        for day_idx, (date, price) in enumerate(observe.items(), start=1):
            ret = self._return(price)
            if ret <= cfg.stop_loss:
                return self._make_result(date, price, "stop_loss", day_idx, p_star)
            if ret >= cfg.take_profit:
                return self._make_result(date, price, "take_profit", day_idx, p_star)

        # 선택 구간 : Day (observe+1) ~ 끝까지
        # 60일 텀마다 secretary 기준 5%씩 감소 (최소 P* × 1.00)
        #   1텀 (Day 23~82)  : P* × 1.15
        #   2텀 (Day 83~142) : P* × 1.10
        #   3텀 (Day 143~202): P* × 1.05
        #   4텀 (Day 203~  ) : P* × 1.00  (이후 고정)
        select = wp.iloc[self._observe_days :]

        for day_idx, (date, price) in enumerate(select.items(),
                                                start=self._observe_days + 1):
            ret = self._return(price)

            # stop_loss / take_profit 우선 체크
            if ret <= cfg.stop_loss:
                return self._make_result(date, price, "stop_loss", day_idx, p_star)
            if ret >= cfg.take_profit:
                return self._make_result(date, price, "take_profit", day_idx, p_star)

            # 현재 텀 계산 (0-indexed)
            # day_idx는 observe_days+1부터 시작, window=60 단위로 텀 증가
            days_in_select = day_idx - self._observe_days - 1  # 선택구간 내 경과일
            term = days_in_select // cfg.window               # 0, 1, 2, 3, ...

            # Secretary 기준: 1.15에서 텀당 0.05씩 감소, 최소 1.00
            multiplier = max(1.00, 1.15 - term * 0.05)

            # Secretary 조건: 현재가 > P* × multiplier
            if price > p_star * multiplier:
                return self._make_result(date, price, "secretary_trigger", day_idx, p_star)

        # 데이터 끝까지 왔는데 미청산 → 마지막 가격으로 청산
        last_date  = wp.index[-1]
        last_price = wp.iloc[-1]
        return self._make_result(last_date, last_price, "secretary_trigger", len(wp), p_star)

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
        # ✅ 추가: UCB1 모드일 때 전역 selector 공유
        self._ucb1 = UCB1ObserveSelector() if self.cfg.observe_mode == "ucb1" else None

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
                    # ✅ 추가: UCB1 모드면 selector에서 관찰 기간 주입
                    if self._ucb1 is not None:
                        chosen_arm = self._ucb1.select()
                        stopper._ucb1_observe_days = chosen_arm
                        stopper._observe_days      = chosen_arm
                    result = stopper.run()
                    # ✅ 추가: UCB1 결과로 selector 업데이트
                    if self._ucb1 is not None:
                        self._ucb1.update(chosen_arm, result.return_pct)
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
    from data_pipeline    import run_pipeline
    from greedy_strategy  import PortfolioGreedySelector

    store, _  = run_pipeline()
    pg        = PortfolioGreedySelector(store)
    results   = pg.run()

    price_store   = {t: df["Close"] for t, df in store.items()}
    entry_signals = {t: df[df["entry"] == True].index for t, df in results.items()}

    ps     = PortfolioStopper(price_store, entry_signals)
    trades = ps.run()
    stats  = trade_statistics(ps.summary(trades))
    for key, value in stats.items():
        print(f"  {key}: {value}")