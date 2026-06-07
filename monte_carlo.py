"""
====================
Monte Carlo 무작위 탐색 포트폴리오 비중 결정

알고리즘:
  n_samples 개의 가중치 벡터를 랜덤으로 생성하고
  각각의 Sharpe Ratio를 계산해 가장 높은 것을 선택

제약 조건 (GAConfig와 동일):
  0.05 <= w_i <= 0.30   (종목별 최소 5% / 최대 30%)
  sum(w_i) <= 0.90      (최최소 10% 현금 보유)

파라미터:
  n_samples : 10,000   (샘플링 횟수)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time

# GA 파일에서 공통 설정/유틸 재사용
from genetic_algorithm import GAConfig, compute_sharpe, TICKERS


class MonteCarloOptimizer:
    """
    mc     = MonteCarloOptimizer(returns_train, tickers, cfg)
    result = mc.run()
    best_w = result["best_weights"]   # dict {ticker: weight}
    """

    def __init__(
        self,
        returns_df : pd.DataFrame,
        tickers    : list  = None,
        config     : GAConfig = None,
        n_samples  : int   = 10_000,
        verbose    : bool  = True,
    ):
        self.returns   = returns_df.copy()
        self.tickers   = tickers or list(returns_df.columns)
        self.cfg       = config or GAConfig()
        self.n_samples = n_samples
        self.verbose   = verbose
        self.n_assets  = len(self.tickers)

    def _sample_weights(self) -> np.ndarray:
        """
        [w_min, w_max] 균등분포로 n_assets개 샘플링 후
        합계가 max_invested 초과하면 비례 스케일다운
        """
        cfg = self.cfg
        w = np.random.uniform(cfg.w_min, cfg.w_max, self.n_assets)
        total = w.sum()
        if total > cfg.max_invested:
            w = w / total * cfg.max_invested
        return w

    def run(self) -> dict:
        cfg = self.cfg
        np.random.seed(42)

        t0 = time.time()

        if self.verbose:
            print("=" * 58)
            print(f"  Monte Carlo Optimizer  |  samples={self.n_samples:,}")
            print(f"  자산 수: {self.n_assets}  |  학습 데이터: {len(self.returns)}일")
            print("=" * 58)

        best_sharpe  = -np.inf
        best_weights = None
        history_rows = []

        log_interval = self.n_samples // 10   # 10% 단위로 진행상황 출력

        for i in range(self.n_samples):
            w      = self._sample_weights()
            sharpe = compute_sharpe(w, self.returns, cfg)

            if sharpe > best_sharpe:
                best_sharpe  = sharpe
                best_weights = w.copy()
                if self.verbose:
                    print(f"  [{i+1:>6}] 새 최고 Sharpe: {best_sharpe:.4f}")

            history_rows.append({
                "sample"      : i + 1,
                "best_sharpe" : best_sharpe,
                "sample_sharpe": sharpe,
            })

            if self.verbose and (i + 1) % log_interval == 0:
                print(f"  진행: {i+1:>6}/{self.n_samples}  현재 최고: {best_sharpe:.4f}")

        elapsed = time.time() - t0

        if self.verbose:
            print("-" * 58)
            print(f"  완료  |  최고 Sharpe: {best_sharpe:.4f}  |  {elapsed:.1f}초")
            print("=" * 58)

        best_weights_dict = {
            t: float(w) for t, w in zip(self.tickers, best_weights)
        }
        return {
            "best_weights"    : best_weights_dict,
            "best_weights_arr": best_weights,
            "best_sharpe"     : best_sharpe,
            "history"         : pd.DataFrame(history_rows),
            "cash_reserve"    : round(1.0 - best_weights.sum(), 4),
        }


if __name__ == "__main__":
    from data_pipeline import run_pipeline

    store, splits = run_pipeline()

    train_ret = pd.DataFrame({
        t: splits[t]["train"]["Return"] for t in splits
    }).dropna()

    mc     = MonteCarloOptimizer(train_ret, list(splits.keys()), n_samples=10_000)
    result = mc.run()

    print("\n  최적 가중치:")
    for ticker, w in result["best_weights"].items():
        print(f"    {ticker}: {w*100:.2f}%")
    print(f"  현금 보유: {result['cash_reserve']*100:.2f}%")
