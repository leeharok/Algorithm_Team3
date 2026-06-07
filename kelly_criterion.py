"""
====================
Kelly Criterion으로 최적 포트폴리오 비중 결정

알고리즘 (O(N), 종목당 한 번만 계산):
  각 종목의 과거 일별 수익률에서
    p  = 수익이 난 날의 비율 (승률)
    b  = 평균 수익 / 평균 손실의 절댓값 (손익비)
  Kelly 공식: f* = p - (1-p)/b
  → f* > 0인 종목만 편입, 음수는 0으로 처리 (투자 안 함)
  → 이후 GAConfig 제약 조건(w_min/w_max/max_invested)에 맞게 클리핑
  승률이 높고 손익비가 클수록 더 많이 배분.

제약 조건 (GAConfig와 동일):
  0.05 <= w_i <= 0.30   (종목별 최소 5% / 최대 30%)
  sum(w_i) <= 0.90      (최소 10% 현금 보유)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import time

from genetic_algorithm import GAConfig, compute_sharpe, TICKERS


def kelly_weights(returns_df: pd.DataFrame, cfg: GAConfig) -> np.ndarray:
    """
    종목별 Kelly 비율 계산 후 제약 조건 적용.

    Parameters
    ----------
    returns_df : (T × n_assets) 일별 수익률
    cfg        : GAConfig (w_min, w_max, max_invested 사용)

    Returns
    -------
    np.ndarray : shape (n_assets,), 제약 적용 완료된 가중치
    """
    ret = returns_df.dropna(how="any")
    n   = ret.shape[1]
    raw_kelly = np.zeros(n)

    for i in range(n):
        r    = ret.iloc[:, i].values
        wins = r[r > 0]
        loss = r[r < 0]

        if len(wins) == 0 or len(loss) == 0:
            raw_kelly[i] = 0.0
            continue

        p = len(wins) / len(r)          # 승률
        b = wins.mean() / abs(loss.mean())  # 손익비

        f = p - (1 - p) / b             # Kelly 공식
        raw_kelly[i] = max(f, 0.0)      # 음수는 투자 안 함 → 0

    # 비중 정규화 → max_invested 맞추기
    total = raw_kelly.sum()
    if total == 0:
        # 모든 종목 Kelly=0인 극단 케이스 → 균등 배분으로 폴백
        w = np.full(n, cfg.max_invested / n)
    else:
        w = raw_kelly / total * cfg.max_invested

    # 제약 클리핑 [w_min, w_max]
    w = np.clip(w, cfg.w_min, cfg.w_max)

    # 클리핑 후 합계 재조정
    total = w.sum()
    if total > cfg.max_invested:
        w = w / total * cfg.max_invested

    return w


class KellyCriterionOptimizer:
    """
    kc     = KellyCriterionOptimizer(returns_train, tickers, cfg)
    result = kc.run()
    best_w = result["best_weights"]   # dict {ticker: weight}

    history: 종목별 Kelly 계산 과정을 담은 DataFrame.
    """

    def __init__(
        self,
        returns_df : pd.DataFrame,
        tickers    : list  = None,
        config     : GAConfig = None,
        verbose    : bool  = True,
    ):
        self.returns  = returns_df.copy()
        self.tickers  = tickers or list(returns_df.columns)
        self.cfg      = config or GAConfig()
        self.verbose  = verbose
        self.n_assets = len(self.tickers)

    def run(self) -> dict:
        cfg = self.cfg
        t0  = time.time()

        if self.verbose:
            print("=" * 58)
            print(f"  Kelly Criterion Optimizer  |  O(N), N={self.n_assets}")
            print(f"  학습 데이터: {len(self.returns)}일")
            print("=" * 58)
            print(f"  {'종목':<16} {'승률':>6} {'손익비':>8} {'Kelly f*':>10} {'최종 비중':>10}")
            print("-" * 58)

        ret  = self.returns.dropna(how="any")
        history_rows = []

        raw_kelly = np.zeros(self.n_assets)
        for i, ticker in enumerate(self.tickers):
            r    = ret.iloc[:, i].values
            wins = r[r > 0]
            loss = r[r < 0]

            if len(wins) == 0 or len(loss) == 0:
                p, b, f = 0.0, 0.0, 0.0
            else:
                p = len(wins) / len(r)
                b = wins.mean() / abs(loss.mean())
                f = max(p - (1 - p) / b, 0.0)

            raw_kelly[i] = f
            history_rows.append({
                "ticker"   : ticker,
                "win_rate" : round(p, 4),
                "win_loss_ratio": round(b, 4),
                "kelly_f"  : round(f, 4),
            })

        # 정규화 & 제약 적용
        total = raw_kelly.sum()
        if total == 0:
            w = np.full(self.n_assets, cfg.max_invested / self.n_assets)
        else:
            w = raw_kelly / total * cfg.max_invested

        w = np.clip(w, cfg.w_min, cfg.w_max)
        s = w.sum()
        if s > cfg.max_invested:
            w = w / s * cfg.max_invested

        # history에 최종 비중 추가
        for i, row in enumerate(history_rows):
            row["final_weight"] = round(float(w[i]), 4)
            if self.verbose:
                print(
                    f"  {row['ticker']:<16} "
                    f"{row['win_rate']:>6.2%} "
                    f"{row['win_loss_ratio']:>8.4f} "
                    f"{row['kelly_f']:>10.4f} "
                    f"{row['final_weight']:>10.4f}"
                )

        sharpe  = compute_sharpe(w, self.returns, cfg)
        elapsed = time.time() - t0

        if self.verbose:
            print("-" * 58)
            print(f"  Sharpe: {sharpe:.4f}  |  현금: {(1-w.sum())*100:.2f}%  |  {elapsed:.3f}초")
            print("=" * 58)

        best_weights_dict = {
            t: float(wi) for t, wi in zip(self.tickers, w)
        }
        return {
            "best_weights"    : best_weights_dict,
            "best_weights_arr": w,
            "best_sharpe"     : sharpe,
            "history"         : pd.DataFrame(history_rows),
            "cash_reserve"    : round(1.0 - w.sum(), 4),
        }


if __name__ == "__main__":
    from data_pipeline import run_pipeline

    store, splits = run_pipeline()

    train_ret = pd.DataFrame({
        t: splits[t]["train"]["Return"] for t in splits
    }).dropna()

    kc     = KellyCriterionOptimizer(train_ret, list(splits.keys()))
    result = kc.run()

    print("\n  최적 가중치:")
    for ticker, w in result["best_weights"].items():
        print(f"    {ticker}: {w*100:.2f}%")
    print(f"  현금 보유: {result['cash_reserve']*100:.2f}%")
