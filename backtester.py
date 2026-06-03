"""
세 알고리즘을 날짜 순서대로 통합 실행:
  1. Greedy  → 오늘의 전략 + 진입 시그널 결정
  2. GA      → 진입할 종목에 자본 배분 가중치 적용
  3. Optimal Stopping → 각 포지션의 매도 타이밍 결정

성과 지표:
  - 누적 수익률 (Cumulative Return)
  - 최대 낙폭    (MDD, Maximum Drawdown)
  - 샤프 지수    (Sharpe Ratio)

실행 방법:
  python backtester.py          # 더미 데이터 데모
  python backtester.py --real   # 실제 데이터 (data_pipeline 필요)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import os, sys, time
from greedy_strategy  import (GreedyStrategySelector, GreedyConfig,
                               PortfolioGreedySelector)
from optimal_stopping import OptimalStopper, StoppingConfig, TradeResult
from genetic_algorithm import (GeneticAlgorithm, GAConfig,
                                evaluate_weights, TICKERS)

@dataclass
class BacktestConfig:
    initial_capital : float = 100_000_000   # 초기 자본 1억원
    commission      : float = 0.001         # 거래 수수료 0.10%
    tax             : float = 0.001         # 증권거래세 0.10% (매도 시)
    slippage        : float = 0.001         # 슬리피지 0.10%
    risk_free_rate  : float = 0.03          # 연간 무위험 수익률
    trading_days    : int   = 252

    greedy : GreedyConfig   = field(default_factory=GreedyConfig)
    stop   : StoppingConfig = field(default_factory=StoppingConfig)
    ga     : GAConfig       = field(default_factory=GAConfig)

@dataclass
class Position:
    ticker      : str
    entry_date  : pd.Timestamp
    entry_price : float
    shares      : int
    capital_used: float          # 실제 투입 자본 (수수료 포함)
    strategy    : str            # 진입 당시 전략
    stopper     : OptimalStopper = field(repr=False, default=None)

    @property
    def current_value(self) -> float:
        return self.shares * self.entry_price   # 최신가로 업데이트 필요

@dataclass
class DailySnapshot:
    date            : pd.Timestamp
    portfolio_value : float
    cash            : float
    invested        : float
    n_positions     : int
    daily_return    : float
    strategy        : str          # 당일 지배적 전략

class Backtester:
    def __init__(
        self,
        store      : dict,            # {ticker: df(Close, MA20, ...)}
        ga_weights : dict,            # {ticker: weight}  GA 결과
        cfg        : BacktestConfig = None,
        verbose    : bool = True,
    ):
        self.store      = store
        self.ga_weights = ga_weights
        self.cfg        = cfg or BacktestConfig()
        self.verbose    = verbose

        # 가격 시리즈 캐시
        self.prices : dict = {t: df["Close"] for t, df in store.items()}

        # 상태
        self.cash       : float = cfg.initial_capital if cfg else 100_000_000
        self.positions  : list  = []
        self.closed     : list  = []    # 청산 완료 TradeResult
        self.snapshots  : list  = []

    def _buy_cost(self, price: float, shares: int) -> float:
        """매수 실효 가격 (수수료 + 슬리피지)"""
        return price * shares * (1 + self.cfg.commission + self.cfg.slippage)

    def _sell_proceeds(self, price: float, shares: int) -> float:
        """매도 실효 수익 (수수료 + 세금 + 슬리피지)"""
        return price * shares * (1 - self.cfg.commission - self.cfg.tax - self.cfg.slippage)

    def _enter_position(
        self,
        ticker   : str,
        date     : pd.Timestamp,
        strategy : str,
    ):
        price  = self.prices[ticker].get(date)
        if price is None or np.isnan(price):
            return

        # GA 가중치 → 투입 자본
        weight       = self.ga_weights.get(ticker, 0)
        alloc_capital= self.cfg.initial_capital * weight
        alloc_capital= min(alloc_capital, self.cash * 0.95)   # 현금 5% 여유

        if alloc_capital < price:   # 1주도 못 사면 패스
            return

        shares    = int(alloc_capital / price)
        cost      = self._buy_cost(price, shares)

        if cost > self.cash:
            shares = int(self.cash / (price * (1 + self.cfg.commission + self.cfg.slippage)))
            if shares <= 0:
                return
            cost = self._buy_cost(price, shares)

        self.cash -= cost

        stopper = OptimalStopper(
            self.prices[ticker], date, price, ticker, self.cfg.stop
        )
        pos = Position(
            ticker       = ticker,
            entry_date   = date,
            entry_price  = price,
            shares       = shares,
            capital_used = cost,
            strategy     = strategy,
            stopper      = stopper,
        )
        self.positions.append(pos)

        if self.verbose:
            print(f"    ▲ 매수 [{ticker}] {date.date()}  "
                  f"{price:>8,.0f}원 × {shares}주  "
                  f"(전략: {strategy}, 배분: {weight*100:.1f}%)")

    def _check_and_close(self, date: pd.Timestamp):
        still_open = []
        for pos in self.positions:
            result = pos.stopper.run()

            # 청산일이 오늘 또는 이전이면 처리
            if result.exit_date <= date:
                proceeds      = self._sell_proceeds(result.exit_price, pos.shares)
                self.cash    += proceeds
                self.closed.append(result)

                pnl_pct = result.return_pct * 100
                sign    = "+" if pnl_pct >= 0 else ""
                if self.verbose:
                    print(f"    ▼ 매도 [{pos.ticker}] {result.exit_date.date()}  "
                          f"{sign}{pnl_pct:.2f}%  ({result.exit_reason})")
            else:
                still_open.append(pos)

        self.positions = still_open

    def _portfolio_value(self, date: pd.Timestamp) -> float:
        invested = 0.0
        for pos in self.positions:
            price = self.prices[pos.ticker].get(date, pos.entry_price)
            invested += price * pos.shares
        return self.cash + invested

    def run(
        self,
        start_date : str,
        end_date   : str,
    ) -> dict:
        """
        Returns
        -------
        dict with keys:
          snapshots  : pd.DataFrame  일별 포트폴리오 상태
          trades     : pd.DataFrame  전체 거래 내역
          metrics    : dict          성과 지표
        """
        cfg    = self.cfg
        self.cash = cfg.initial_capital
        self.positions.clear()
        self.closed.clear()
        self.snapshots.clear()

        # 거래일 목록
        sample_prices = next(iter(self.prices.values()))
        trading_dates = sample_prices.loc[start_date:end_date].index

        if self.verbose:
            print("\n" + "=" * 62)
            print(f"  백테스팅 시작  {start_date} ~ {end_date}")
            print(f"  초기 자본: {cfg.initial_capital:,.0f}원")
            print("=" * 62)

        prev_value = cfg.initial_capital

        # Greedy 선택기 초기화
        pg      = PortfolioGreedySelector(self.store, cfg.greedy)
        g_results = pg.run()   # {ticker: df with strategy, entry cols}

        # 이미 포지션이 있는 종목 추적 (중복 진입 방지)
        in_position = set()

        for date in trading_dates:
            self._check_and_close(date)
            in_position = {p.ticker for p in self.positions}

            for ticker, g_df in g_results.items():
                if date not in g_df.index:
                    continue
                row = g_df.loc[date]

                # Defensive → 신규 진입 없음
                if row["strategy"] == "Defensive":
                    continue

                # 이미 해당 종목 포지션 보유 중 → 스킵
                if ticker in in_position:
                    continue

                # entry 시그널 확인
                if row.get("entry", False):
                    self._enter_position(ticker, date, row["strategy"])
                    in_position.add(ticker)

            port_val  = self._portfolio_value(date)
            daily_ret = (port_val / prev_value) - 1 if prev_value > 0 else 0.0

            # 당일 지배적 전략 (포지션 없으면 Greedy 기준)
            if self.positions:
                dom_strategy = max(
                    set(p.strategy for p in self.positions),
                    key=lambda s: sum(1 for p in self.positions if p.strategy == s)
                )
            else:
                strategies = [
                    g_results[t].loc[date, "strategy"]
                    for t in g_results if date in g_results[t].index
                ]
                dom_strategy = max(set(strategies), key=strategies.count) if strategies else "Defensive"

            self.snapshots.append(DailySnapshot(
                date            = date,
                portfolio_value = port_val,
                cash            = self.cash,
                invested        = port_val - self.cash,
                n_positions     = len(self.positions),
                daily_return    = daily_ret,
                strategy        = dom_strategy,
            ))
            prev_value = port_val

        last_date = trading_dates[-1]
        for pos in self.positions:
            last_price = self.prices[pos.ticker].get(last_date, pos.entry_price)
            proceeds   = self._sell_proceeds(last_price, pos.shares)
            self.cash += proceeds
            ret_pct    = (last_price - pos.entry_price) / pos.entry_price
            self.closed.append(TradeResult(
                ticker      = pos.ticker,
                entry_date  = pos.entry_date,
                entry_price = pos.entry_price,
                exit_date   = last_date,
                exit_price  = last_price,
                exit_reason = "force_liquidate",
                hold_days   = (last_date - pos.entry_date).days,
                p_star      = pos.entry_price,
                return_pct  = ret_pct,
            ))
        self.positions.clear()

        snap_df  = self._snapshots_to_df()
        trade_df = self._trades_to_df()
        metrics  = self._compute_metrics(snap_df)

        return {
            "snapshots" : snap_df,
            "trades"    : trade_df,
            "metrics"   : metrics,
        }

    def _snapshots_to_df(self) -> pd.DataFrame:
        rows = [{
            "date"            : s.date,
            "portfolio_value" : s.portfolio_value,
            "cash"            : s.cash,
            "invested"        : s.invested,
            "n_positions"     : s.n_positions,
            "daily_return"    : s.daily_return,
            "strategy"        : s.strategy,
        } for s in self.snapshots]
        df = pd.DataFrame(rows).set_index("date")

        # 누적 수익률
        df["cum_return"] = df["portfolio_value"] / self.cfg.initial_capital - 1
        return df

    def _trades_to_df(self) -> pd.DataFrame:
        if not self.closed:
            return pd.DataFrame()
        rows = [{
            "ticker"      : t.ticker,
            "entry_date"  : t.entry_date.date(),
            "exit_date"   : t.exit_date.date(),
            "hold_days"   : t.hold_days,
            "entry_price" : round(t.entry_price, 0),
            "exit_price"  : round(t.exit_price,  0),
            "return_pct"  : round(t.return_pct * 100, 2),
            "exit_reason" : t.exit_reason,
        } for t in self.closed]
        return pd.DataFrame(rows)

    def _compute_metrics(self, snap_df: pd.DataFrame) -> dict:
        cfg = self.cfg
        pv  = snap_df["portfolio_value"]
        dr  = snap_df["daily_return"]

        # 누적 수익률
        total_ret = pv.iloc[-1] / cfg.initial_capital - 1

        # MDD
        rolling_max = pv.cummax()
        drawdown    = (pv - rolling_max) / rolling_max
        mdd         = drawdown.min()

        # Sharpe
        daily_rf = cfg.risk_free_rate / cfg.trading_days
        excess   = dr - daily_rf
        sharpe   = (excess.mean() / excess.std(ddof=1) * np.sqrt(cfg.trading_days)
                    if excess.std(ddof=1) > 0 else 0.0)

        # 거래 통계
        trades = self._trades_to_df()
        if not trades.empty:
            win_rate   = (trades["return_pct"] > 0).mean() * 100
            avg_ret    = trades["return_pct"].mean()
            n_trades   = len(trades)
            reasons    = trades["exit_reason"].value_counts().to_dict()
        else:
            win_rate = avg_ret = n_trades = 0
            reasons  = {}

        return {
            "누적 수익률 (%)"  : round(total_ret * 100, 2),
            "MDD (%)"         : round(mdd * 100, 2),
            "Sharpe Ratio"    : round(sharpe, 4),
            "최종 자산 (원)"  : round(pv.iloc[-1], 0),
            "총 거래 수"      : n_trades,
            "승률 (%)"        : round(win_rate, 1),
            "평균 거래 수익 (%)": round(avg_ret, 2),
            "청산 사유 분포"  : reasons,
        }

def print_report(result: dict, label: str = "백테스팅"):
    m = result["metrics"]
    print(f"\n{'─'*50}")
    print(f"  {label} 성과 리포트")
    print(f"{'─'*50}")
    print(f"  누적 수익률  : {m['누적 수익률 (%)']:>+8.2f} %")
    print(f"  MDD          : {m['MDD (%)']:>+8.2f} %")
    print(f"  Sharpe Ratio : {m['Sharpe Ratio']:>8.4f}")
    print(f"  최종 자산    : {m['최종 자산 (원)']:>14,.0f} 원")
    print(f"  총 거래 수   : {m['총 거래 수']:>8} 건")
    print(f"  승률         : {m['승률 (%)']:>8.1f} %")
    print(f"  평균 거래 수익: {m['평균 거래 수익 (%)']:>+7.2f} %")
    print(f"  청산 사유    : {m['청산 사유 분포']}")
    print(f"{'─'*50}")

    trades = result["trades"]
    if not trades.empty:
        print(f"\n  [최근 거래 10건]")
        print(trades.tail(10).to_string(index=False))


def save_backtest(result: dict, label: str = "full", output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    result["snapshots"].to_csv(f"{output_dir}/bt_{label}_snapshots.csv")
    if not result["trades"].empty:
        result["trades"].to_csv(f"{output_dir}/bt_{label}_trades.csv", index=False)

    metrics_df = pd.DataFrame([result["metrics"]]).T
    metrics_df.to_csv(f"{output_dir}/bt_{label}_metrics.csv", header=False)
    print(f"  ✓ 결과 저장 → {output_dir}/bt_{label}_*.csv")

if __name__ == "__main__":
    from data_pipeline import run_pipeline
    from backtester import Backtester, BacktestConfig, print_report, save_backtest
    import pandas as pd

    store, splits = run_pipeline()

    # 2단계에서 만든 GA 가중치 로드
    w_df = pd.read_csv("data/ga_best_weights.csv")
    ga_weights = {
        col.replace("_KS", ".KS"): float(w_df[col].iloc[0])
        for col in w_df.columns if col != "cash_reserve"
    }

    cfg = BacktestConfig(initial_capital=100_000_000)
    bt  = Backtester(store, ga_weights, cfg, verbose=True)
    result = bt.run("2018-01-02", "2025-06-30")
    
    print_report(result)
    save_backtest(result, "full")