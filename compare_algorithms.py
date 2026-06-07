"""
세 가지 포트폴리오 최적화 알고리즘 비교
  - Monte Carlo (무작위 탐색)
  - Genetic Algorithm (진화 알고리즘)
  - Kelly Criterion (수학 공식)
"""

import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from data_pipeline          import run_pipeline
from genetic_algorithm      import GeneticAlgorithm, GAConfig, evaluate_weights
from monte_carlo    import MonteCarloOptimizer
from kelly_criterion import KellyCriterionOptimizer

# ── 데이터 준비 ──────────────────────────────────────────
store, splits = run_pipeline()

def get_returns(split_name):
    return pd.DataFrame({
        t: splits[t][split_name]["Return"] for t in splits
    }).dropna()

train_ret = get_returns("train")
test_ret  = get_returns("test")
tickers   = list(splits.keys())
cfg       = GAConfig()

# ── 세 알고리즘 실행 ─────────────────────────────────────
print("\n▶ 1/3  Monte Carlo")
mc_result = MonteCarloOptimizer(train_ret, tickers, cfg, n_samples=10_000).run()

print("\n▶ 2/3  Genetic Algorithm")
ga_result = GeneticAlgorithm(train_ret, tickers, cfg).run()

print("\n▶ 3/3  Kelly Criterion")
kc_result = KellyCriterionOptimizer(train_ret, tickers, cfg).run()

# ── 결과 비교 ────────────────────────────────────────────
algorithms = {
    "Monte Carlo"      : mc_result["best_weights_arr"],
    "Genetic Algorithm": ga_result["best_weights_arr"],
    "Kelly Criterion"  : kc_result["best_weights_arr"],
}

rows = []
for name, w_arr in algorithms.items():
    train_eval = evaluate_weights(w_arr, train_ret, cfg, label="train")
    test_eval  = evaluate_weights(w_arr, test_ret,  cfg, label="test")
    rows.append({
        "알고리즘"          : name,
        "Train Sharpe"      : train_eval["Sharpe Ratio"],
        "Test  Sharpe"      : test_eval["Sharpe Ratio"],
        "Train 누적수익(%)": train_eval["누적 수익률 (%)"],
        "Test  누적수익(%)": test_eval["누적 수익률 (%)"],
        "Train MDD(%)"      : train_eval["MDD (%)"],
        "Test  MDD(%)"      : test_eval["MDD (%)"],
    })

result_df = pd.DataFrame(rows).set_index("알고리즘")

print("\n" + "=" * 70)
print("  알고리즘 비교 결과")
print("=" * 70)
print(result_df.to_string())
print("=" * 70)

# ── 종목별 비중 비교 ─────────────────────────────────────
weight_df = pd.DataFrame({
    name: {t: round(w * 100, 2) for t, w in res["best_weights"].items()}
    for name, res in [
        ("Monte Carlo",       mc_result),
        ("Genetic Algorithm", ga_result),
        ("Kelly Criterion",   kc_result),
    ]
})
weight_df.index.name = "ticker"
weight_df["비중 차이(MC↔GA)"] = (weight_df["Monte Carlo"] - weight_df["Genetic Algorithm"]).round(2)
weight_df["비중 차이(GA↔KC)"] = (weight_df["Genetic Algorithm"] - weight_df["Kelly Criterion"]).round(2)

print("\n  종목별 비중 (%) 비교")
print("-" * 70)
print(weight_df.to_string())
print("-" * 70)
