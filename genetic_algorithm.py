"""
====================
각 염색체는 9개 섹터 종목에 대한 자본 배분 가중치 벡터
Sharpe Ratio를 fitness 함수로 사용해 최적 가중치를 진화

제약 조건:
  0.05 <= w_i <= 0.30   (종목별 최소 5% / 최대 30%)
  sum(w_i) <= 0.90      (최소 10% 현금 보유)

파라미터 :
  Population  : 30
  Generations : 50
  Selection   : Tournament (size 3)
  Crossover   : Uniform, p=0.80
  Mutation    : Gaussian, sigma=0.02
  Early stop  : 15세대 개선 없으면 종료
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import os, copy, time

TICKERS = [
    "005930.KS",  # Samsung Electronics (IT)
    "009540.KS",  # HD Korea Shipbuilding (Heavy Industry)
    "034020.KS",  # Doosan Enerbility (Industrials)
    "105560.KS",  # KB Financial (Financials)
    "005380.KS",  # Hyundai Motor (Consumer Disc.)
    "005490.KS",  # POSCO Holdings (Energy/Chem)
    "207940.KS",  # Samsung Biologics (Healthcare)
    "090430.KS",  # AmorePacific (Consumer Staples)
    "035420.KS",  # NAVER (Communication)
]

@dataclass
class GAConfig:
    # 진화 파라미터
    pop_size         : int   = 30
    n_generations    : int   = 50
    tournament_size  : int   = 3
    crossover_prob   : float = 0.80
    mutation_sigma   : float = 0.02
    early_stop_gens  : int   = 15

    # 가중치 제약
    w_min            : float = 0.05    # 종목당 최소 배분
    w_max            : float = 0.30    # 종목당 최대 배분
    max_invested     : float = 0.90    # 최대 투자 비율 (현금 최소 10%)

    # Sharpe 계산
    trading_days     : int   = 252
    risk_free_rate   : float = 0.03    # 연간 무위험 수익률

class Chromosome:
    def __init__(self, weights: np.ndarray, cfg: GAConfig):
        self.cfg     = cfg
        self.weights = self._repair(weights.copy())
        self.fitness : float = -np.inf

    def _repair(self, w: np.ndarray) -> np.ndarray:
        """
        1) 클리핑 [w_min, w_max]
        2) 합계가 max_invested 초과 시 비례 스케일다운
        3) 합계가 n*w_min 미만이면 비례 스케일업
        """
        cfg = self.cfg
        n   = len(w)

        # Step 1: 개별 클리핑
        w = np.clip(w, cfg.w_min, cfg.w_max)

        # Step 2: 합계 제약
        total = w.sum()
        if total > cfg.max_invested:
            w = w / total * cfg.max_invested

        # Step 3: 최소 합계 보장
        total = w.sum()
        min_total = n * cfg.w_min
        if total < min_total:
            w = w / total * min_total

        # Step 4: 재클리핑 
        w = np.clip(w, cfg.w_min, cfg.w_max)
        return w

    def copy(self) -> "Chromosome":
        c = Chromosome.__new__(Chromosome)
        c.cfg     = self.cfg
        c.weights = self.weights.copy()
        c.fitness = self.fitness
        return c

    def __repr__(self):
        w_str = ", ".join(f"{v:.3f}" for v in self.weights)
        return f"Chromosome(fitness={self.fitness:.4f}, weights=[{w_str}])"

def compute_sharpe(weights: np.ndarray, returns_df: pd.DataFrame, cfg: GAConfig) -> float:
    """
    Parameters
    ----------
    weights    : shape (n_assets,)
    returns_df : shape (T, n_assets)  일별 수익률, NaN 허용
    """
    # NaN 행 제거
    ret = returns_df.dropna(how="any")
    if len(ret) < 20:
        return -np.inf

    # 포트폴리오 일별 수익률
    port_ret = ret.values @ weights          # (T,)

    mean_ret = port_ret.mean()
    std_ret  = port_ret.std(ddof=1)

    if std_ret == 0 or np.isnan(std_ret):
        return -np.inf

    # 연간화 Sharpe
    daily_rf = cfg.risk_free_rate / cfg.trading_days
    sharpe   = (mean_ret - daily_rf) / std_ret * np.sqrt(cfg.trading_days)
    return float(sharpe)

def tournament_selection(population: list, cfg: GAConfig) -> Chromosome:
    """토너먼트 선택: 랜덤 k명 중 fitness 최고 개체 반환"""
    candidates = np.random.choice(len(population), size=cfg.tournament_size, replace=False)
    best       = max(candidates, key=lambda i: population[i].fitness)
    return population[best].copy()


def uniform_crossover(p1: Chromosome, p2: Chromosome, cfg: GAConfig
                      ) -> tuple["Chromosome", "Chromosome"]:
    """Uniform Crossover: 각 유전자를 50% 확률로 부모에서 선택"""
    if np.random.rand() > cfg.crossover_prob:
        return p1.copy(), p2.copy()

    mask = np.random.rand(len(p1.weights)) < 0.5
    w1   = np.where(mask, p1.weights, p2.weights)
    w2   = np.where(mask, p2.weights, p1.weights)
    return Chromosome(w1, cfg), Chromosome(w2, cfg)


def gaussian_mutation(chrom: Chromosome, cfg: GAConfig) -> Chromosome:
    """각 유전자에 N(0, sigma) 노이즈 추가 후 repair"""
    noise      = np.random.normal(0, cfg.mutation_sigma, len(chrom.weights))
    new_weights = chrom.weights + noise
    return Chromosome(new_weights, cfg)

class GeneticAlgorithm:
    """
    ga     = GeneticAlgorithm(returns_train, tickers, cfg)
    result = ga.run()
    best_w = result["best_weights"]   # dict {ticker: weight}
    """

    def __init__(
        self,
        returns_df : pd.DataFrame,   # 학습용 일별 수익률 (T × n_assets)
        tickers    : list  = None,
        config     : GAConfig = None,
        verbose    : bool  = True,
    ):
        self.returns   = returns_df.copy()
        self.tickers   = tickers or list(returns_df.columns)
        self.cfg       = config or GAConfig()
        self.verbose   = verbose
        self.n_assets  = len(self.tickers)

        # 진화 기록
        self.history : list[dict] = []

    def _init_population(self) -> list:
        pop = []
        for _ in range(self.cfg.pop_size):
            w = np.random.uniform(
                self.cfg.w_min,
                self.cfg.w_max,
                self.n_assets
            )
            pop.append(Chromosome(w, self.cfg))
        return pop

    def _evaluate(self, population: list):
        for chrom in population:
            chrom.fitness = compute_sharpe(chrom.weights, self.returns, self.cfg)

    def run(self) -> dict:
        cfg = self.cfg
        np.random.seed(42)

        t0  = time.time()
        pop = self._init_population()
        self._evaluate(pop)

        best_chrom     = max(pop, key=lambda c: c.fitness).copy()
        no_improve_cnt = 0

        if self.verbose:
            print("=" * 58)
            print(f"  Genetic Algorithm 시작  |  pop={cfg.pop_size}, gen={cfg.n_generations}")
            print(f"  자산 수: {self.n_assets}  |  학습 데이터: {len(self.returns)}일")
            print("=" * 58)
            print(f"  {'세대':>4}  {'최고 Sharpe':>12}  {'평균 Sharpe':>12}  {'개선':>6}")
            print("-" * 58)

        for gen in range(1, cfg.n_generations + 1):

            new_pop = []

            # 엘리트 보존: 상위 2개 그대로 유지
            elite = sorted(pop, key=lambda c: c.fitness, reverse=True)[:2]
            new_pop.extend([e.copy() for e in elite])

            while len(new_pop) < cfg.pop_size:
                p1 = tournament_selection(pop, cfg)
                p2 = tournament_selection(pop, cfg)
                c1, c2 = uniform_crossover(p1, p2, cfg)
                c1 = gaussian_mutation(c1, cfg)
                c2 = gaussian_mutation(c2, cfg)
                new_pop.extend([c1, c2])

            pop = new_pop[: cfg.pop_size]
            self._evaluate(pop)

            gen_best = max(pop, key=lambda c: c.fitness)
            improved = gen_best.fitness > best_chrom.fitness

            if improved:
                best_chrom     = gen_best.copy()
                no_improve_cnt = 0
                flag           = " ✓"
            else:
                no_improve_cnt += 1
                flag           = ""

            avg_fitness = np.mean([c.fitness for c in pop])
            self.history.append({
                "generation"  : gen,
                "best_sharpe" : best_chrom.fitness,
                "avg_sharpe"  : avg_fitness,
                "improved"    : improved,
            })

            if self.verbose and (gen % 5 == 0 or gen == 1 or improved):
                print(f"  {gen:>4}  {best_chrom.fitness:>12.4f}  {avg_fitness:>12.4f}  {flag}")

            if no_improve_cnt >= cfg.early_stop_gens:
                if self.verbose:
                    print(f"\n  ⏹  Early stop (세대 {gen}: {cfg.early_stop_gens}세대 개선 없음)")
                break

        elapsed = time.time() - t0
        if self.verbose:
            print("-" * 58)
            print(f"  완료  |  최고 Sharpe: {best_chrom.fitness:.4f}  |  {elapsed:.1f}초")
            print("=" * 58)

        best_weights_dict = {
            t: float(w) for t, w in zip(self.tickers, best_chrom.weights)
        }
        return {
            "best_weights"  : best_weights_dict,
            "best_weights_arr": best_chrom.weights,
            "best_sharpe"   : best_chrom.fitness,
            "history"       : pd.DataFrame(self.history),
            "cash_reserve"  : round(1.0 - best_chrom.weights.sum(), 4),
        }

    def convergence_summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

def evaluate_weights(
    weights_arr : np.ndarray,
    returns_df  : pd.DataFrame,
    cfg         : GAConfig,
    label       : str = "Set",
) -> dict:

    ret  = returns_df.dropna(how="any")
    if len(ret) == 0:
        return {}

    port_ret  = ret.values @ weights_arr
    cum_ret   = pd.Series((1 + port_ret).cumprod(), index=ret.index)

    # Cumulative Return
    total_ret = cum_ret.iloc[-1] - 1

    # MDD
    rolling_max = cum_ret.cummax()
    drawdown    = (cum_ret - rolling_max) / rolling_max
    mdd         = drawdown.min()

    # Sharpe
    sharpe = compute_sharpe(weights_arr, returns_df, cfg)

    if cfg.verbose if hasattr(cfg, "verbose") else True:
        pass

    return {
        "label"           : label,
        "기간"             : f"{ret.index[0].date()} ~ {ret.index[-1].date()}",
        "거래일 수"        : len(ret),
        "누적 수익률 (%)"  : round(total_ret * 100, 2),
        "MDD (%)"         : round(mdd * 100, 2),
        "Sharpe Ratio"    : round(sharpe, 4),
    }


def weight_summary(weights_dict: dict, tickers_meta: dict = None) -> pd.DataFrame:
    """가중치 요약 테이블"""
    rows = []
    for ticker, w in weights_dict.items():
        name = (tickers_meta or {}).get(ticker, ticker)
        rows.append({"ticker": ticker, "name": name, "weight": round(w, 4),
                     "weight_%": round(w * 100, 2)})
    df = pd.DataFrame(rows).sort_values("weight", ascending=False)
    df["rank"] = range(1, len(df) + 1)
    return df.set_index("rank")


def save_ga_result(result: dict, output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)

    # 가중치 CSV
    w_df = pd.DataFrame([result["best_weights"]])
    w_df["cash_reserve"] = result["cash_reserve"]
    w_df.to_csv(f"{output_dir}/ga_best_weights.csv", index=False)

    # 수렴 히스토리 CSV
    result["history"].to_csv(f"{output_dir}/ga_history.csv", index=False)
    print(f"  ✓ GA 결과 저장 → {output_dir}/ga_best_weights.csv")
    print(f"  ✓ GA 히스토리  → {output_dir}/ga_history.csv")

if __name__ == "__main__":
    from data_pipeline import run_pipeline
    store, splits = run_pipeline()

    def get_returns(splits, split_name):
        return pd.DataFrame({
            t: splits[t][split_name]["Return"]
            for t in splits
        }).dropna()

    train_ret = get_returns(splits, "train")
    
    ga     = GeneticAlgorithm(train_ret, list(splits.keys()))
    result = ga.run()
    
    save_ga_result(result)