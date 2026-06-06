"""
Rolling GA 가중치 변화 비교 그래프
실행: python plot_rolling_ga.py
출력: ga_rolling_weights.png

필요 파일: data/ga_rolling/ga_weights_*.csv
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager
import glob, warnings
warnings.filterwarnings("ignore")

# ── 한글 폰트 ─────────────────────────────────────────────────
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
font_manager.fontManager.addfont(FONT_PATH)
prop = font_manager.FontProperties(fname=FONT_PATH)
matplotlib.rc("font", family=prop.get_name())
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 색상 테마 ─────────────────────────────────────────────────
BG      = "#F8F7F4"
PANEL   = "#FFFFFF"
BORDER  = "#E8E6E0"
TEXT    = "#2C2C2A"
SUBTEXT = "#888780"

# 종목별 고유 색상 (세련된 팔레트)
PALETTE = [
    "#534AB7",  # purple
    "#1D9E75",  # teal
    "#D85A30",  # coral
    "#D4537E",  # pink
    "#378ADD",  # blue
    "#639922",  # green
    "#BA7517",  # amber
    "#E24B4A",  # red
    "#5F5E5A",  # gray
]

# ── 데이터 로드 ───────────────────────────────────────────────
files = sorted(glob.glob("data/ga_rolling/ga_weights_*.csv"))
if not files:
    raise FileNotFoundError(
        "data/ga_rolling/ 폴더에 CSV 파일이 없습니다.\n"
        "backtester.py 먼저 실행하세요."
    )

all_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
all_df["retraining_date"] = pd.to_datetime(all_df["retraining_date"])
all_df = all_df.sort_values("retraining_date").reset_index(drop=True)

meta_cols   = ["retraining_date", "sharpe", "train_days"]
ticker_cols = [c for c in all_df.columns if c not in meta_cols]
ticker_labels = {c: c.replace("_KS", ".KS") for c in ticker_cols}
date_labels   = [d.strftime("%Y-%m-%d") for d in all_df["retraining_date"]]
n_dates   = len(all_df)
n_tickers = len(ticker_cols)

# ── 레이아웃 ─────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 18), facecolor=BG)
fig.patch.set_facecolor(BG)

gs = fig.add_gridspec(3, 1, hspace=0.55,
                      height_ratios=[2.2, 1.4, 1.0],
                      left=0.07, right=0.96,
                      top=0.93, bottom=0.05)

axes = [fig.add_subplot(gs[i]) for i in range(3)]
for ax in axes:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=SUBTEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(0.8)

# ── 1. 그룹 막대 차트 ─────────────────────────────────────────
ax = axes[0]
x       = np.arange(n_tickers)
width   = 0.75 / max(n_dates, 1)
offsets = np.linspace(-0.375 + width/2, 0.375 - width/2, n_dates)

for i, (_, row) in enumerate(all_df.iterrows()):
    weights = [row[c] * 100 for c in ticker_cols]
    color   = PALETTE[i % len(PALETTE)]
    ax.bar(x + offsets[i], weights,
           width=width * 0.88,
           color=color, alpha=0.90,
           label=date_labels[i],
           edgecolor="white", linewidth=0.4,
           zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(
    [ticker_labels[c] for c in ticker_cols],
    rotation=18, ha="right",
    color=TEXT, fontsize=9.5
)
ax.set_ylabel("가중치 (%)", color=SUBTEXT, fontsize=10)
ax.set_title("재학습 날짜별 종목 가중치 비교",
             color=TEXT, fontsize=13, fontweight="bold", pad=12, loc="left")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.grid(True, color=BORDER, lw=0.6, axis="y", zorder=0)
ax.set_axisbelow(True)
legend = ax.legend(
    title="재학습 날짜", title_fontsize=8.5,
    fontsize=8.5, frameon=True,
    facecolor=PANEL, edgecolor=BORDER,
    labelcolor=TEXT, loc="upper right"
)
legend.get_title().set_color(SUBTEXT)

# ── 2. 라인 차트 — 종목별 가중치 시계열 ──────────────────────
ax = axes[1]
for j, col in enumerate(ticker_cols):
    weights = all_df[col].values * 100
    color   = PALETTE[j % len(PALETTE)]
    ax.plot(range(n_dates), weights,
            marker="o", ms=7, lw=2.2,
            color=color, markerfacecolor="white",
            markeredgewidth=2.0,
            label=ticker_labels[col],
            zorder=4)
    for k, w in enumerate(weights):
        ax.annotate(
            f"{w:.1f}%",
            (k, w),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center", fontsize=7.5,
            color=color, fontweight="bold"
        )

ax.set_xticks(range(n_dates))
ax.set_xticklabels(date_labels, rotation=18, ha="right",
                   color=TEXT, fontsize=9)
ax.set_ylabel("가중치 (%)", color=SUBTEXT, fontsize=10)
ax.set_title("종목별 가중치 변화 추이",
             color=TEXT, fontsize=13, fontweight="bold", pad=12, loc="left")
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.grid(True, color=BORDER, lw=0.6, zorder=0)
ax.set_axisbelow(True)
legend2 = ax.legend(
    fontsize=8, frameon=True,
    facecolor=PANEL, edgecolor=BORDER,
    labelcolor=TEXT, loc="upper right", ncol=3
)

# ── 3. Sharpe Ratio 변화 ──────────────────────────────────────
ax = axes[2]
sharpes = all_df["sharpe"].values
bar_colors = ["#E24B4A" if s < 0 else "#1D9E75" for s in sharpes]

ax.bar(range(n_dates), sharpes,
       color=bar_colors, alpha=0.75,
       edgecolor="white", linewidth=0.5,
       zorder=3)
ax.plot(range(n_dates), sharpes,
        color="#534AB7", marker="o", ms=7,
        lw=2.0, markerfacecolor="white",
        markeredgewidth=2.0, zorder=5)
ax.axhline(0, color=BORDER, lw=1.0, ls="--")

for k, s in enumerate(sharpes):
    ax.annotate(
        f"{s:.3f}",
        (k, s),
        textcoords="offset points",
        xytext=(0, 9 if s >= 0 else -14),
        ha="center", fontsize=9,
        color="#534AB7", fontweight="bold"
    )

ax.set_xticks(range(n_dates))
ax.set_xticklabels(date_labels, rotation=18, ha="right",
                   color=TEXT, fontsize=9)
ax.set_ylabel("Sharpe Ratio", color=SUBTEXT, fontsize=10)
ax.set_title("재학습 시점별 Sharpe Ratio",
             color=TEXT, fontsize=13, fontweight="bold", pad=12, loc="left")
ax.grid(True, color=BORDER, lw=0.6, axis="y", zorder=0)
ax.set_axisbelow(True)

# ── 제목 ─────────────────────────────────────────────────────
fig.suptitle(
    f"Rolling GA 가중치 변화  |  총 {n_dates}회 재학습",
    color=TEXT, fontsize=15, fontweight="bold", y=0.97
)

OUTPUT = "ga_rolling_weights.png"
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"✓ 저장 완료: {OUTPUT}")
plt.show()