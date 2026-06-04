"""
백테스팅 결과 시각화
실행: python plot_backtest.py
출력: backtest_report.png  (data/ 폴더와 같은 위치)

필요 파일:
  data/bt_full_snapshots.csv
  data/bt_full_trades.csv     (있으면 추가 차트)
  data/bt_full_metrics.csv    (있으면 성과 표)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os, warnings
warnings.filterwarnings("ignore")

# ── 한글 폰트 설정 ────────────────────────────────────────────
import matplotlib
from matplotlib import font_manager

FONT_PATH = "C:\Windows\Fonts\malgunsl.ttf"
font_manager.fontManager.addfont(FONT_PATH)
prop = font_manager.FontProperties(fname=FONT_PATH)
matplotlib.rc("font", family=prop.get_name())
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_DIR   = "data"
OUTPUT     = "backtest_report.png"

COLORS = {
    "portfolio" : "#4FC3F7",
    "pos_area"  : "#1976D2",
    "cash"      : "#90A4AE",
    "profit"    : "#66BB6A",
    "loss"      : "#EF5350",
    "mdd"       : "#FF7043",
    "bg"        : "#0D1117",
    "panel"     : "#161B22",
    "grid"      : "#21262D",
    "text"      : "#E6EDF3",
    "subtext"   : "#8B949E",
    "Trend"         : "#4FC3F7",
    "MeanReversion" : "#FFD54F",
    "Defensive"     : "#90A4AE",
}

SPLIT_LINES = {
    "Train 끝\n(2022-06-30)" : "2022-06-30",
    "Val 끝\n(2023-12-31)"   : "2023-12-31",
}

# ── 데이터 로드 ───────────────────────────────────────────────
snap_path = os.path.join(DATA_DIR, "bt_full_snapshots.csv")
if not os.path.exists(snap_path):
    raise FileNotFoundError(f"{snap_path} 파일이 없습니다. backtester.py 먼저 실행하세요.")

snap = pd.read_csv(snap_path, index_col=0, parse_dates=True)
snap.index = pd.to_datetime(snap.index).tz_localize(None)

# trades
trade_path = os.path.join(DATA_DIR, "bt_full_trades.csv")
trades = pd.read_csv(trade_path) if os.path.exists(trade_path) else pd.DataFrame()

# metrics
met_path = os.path.join(DATA_DIR, "bt_full_metrics.csv")
if os.path.exists(met_path):
    met_df  = pd.read_csv(met_path, header=None, index_col=0)
    metrics = met_df[1].to_dict()
else:
    metrics = {}

# ── 파생 계산 ─────────────────────────────────────────────────
initial = snap["portfolio_value"].iloc[0]
snap["cum_pct"]    = (snap["portfolio_value"] / initial - 1) * 100
rolling_max        = snap["portfolio_value"].cummax()
snap["drawdown"]   = (snap["portfolio_value"] - rolling_max) / rolling_max * 100
snap["cash_pct"]   = snap["cash"] / snap["portfolio_value"] * 100

# 30일 롤링 샤프
dr = snap["daily_return"]
rf_daily = 0.03 / 252
roll_sharpe = (dr - rf_daily).rolling(30).mean() / (dr - rf_daily).rolling(30).std() * np.sqrt(252)
snap["roll_sharpe"] = roll_sharpe

# ── 레이아웃 ──────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 20), facecolor=COLORS["bg"])
gs  = GridSpec(5, 2, figure=fig,
               hspace=0.45, wspace=0.30,
               left=0.07, right=0.96, top=0.93, bottom=0.05)

ax_cum    = fig.add_subplot(gs[0, :])    # 누적 수익률 (전체 너비)
ax_cash   = fig.add_subplot(gs[1, 0])   # 현금 비율
ax_pos    = fig.add_subplot(gs[1, 1])   # 포지션 수
ax_mdd    = fig.add_subplot(gs[2, :])   # MDD (전체 너비)
ax_sharpe = fig.add_subplot(gs[3, 0])   # 롤링 샤프
ax_strat  = fig.add_subplot(gs[3, 1])   # 전략 분포 파이
ax_ret    = fig.add_subplot(gs[4, 0])   # 거래 수익률 분포
ax_metric = fig.add_subplot(gs[4, 1])   # 핵심 지표 표

all_axes = [ax_cum, ax_cash, ax_pos, ax_mdd, ax_sharpe, ax_strat, ax_ret, ax_metric]
for ax in all_axes:
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(colors=COLORS["subtext"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["grid"])

def add_split_lines(ax):
    for label, date_str in SPLIT_LINES.items():
        d = pd.Timestamp(date_str)
        if snap.index[0] <= d <= snap.index[-1]:
            ax.axvline(d, color="#555", lw=1, ls="--", alpha=0.7)
            ax.text(d, ax.get_ylim()[1]*0.95, label,
                    color=COLORS["subtext"], fontsize=7, ha="center",
                    va="top", backgroundcolor=COLORS["panel"])

def style_xaxis(ax):
    ax.xaxis.set_major_locator(mticker.MaxNLocator(8))
    ax.tick_params(axis="x", rotation=30)

# ── 1. 누적 수익률 ────────────────────────────────────────────
ax = ax_cum
pos_mask = snap["cum_pct"] >= 0
ax.fill_between(snap.index, snap["cum_pct"], 0,
                where=pos_mask,  color=COLORS["profit"], alpha=0.25)
ax.fill_between(snap.index, snap["cum_pct"], 0,
                where=~pos_mask, color=COLORS["loss"],   alpha=0.25)
ax.plot(snap.index, snap["cum_pct"],
        color=COLORS["portfolio"], lw=1.5, label="누적 수익률")
ax.axhline(0, color=COLORS["subtext"], lw=0.8, ls="--")
ax.set_title("누적 수익률 (%)", color=COLORS["text"], fontsize=12, pad=8)
ax.set_ylabel("%", color=COLORS["subtext"], fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax.grid(True, color=COLORS["grid"], lw=0.5)
style_xaxis(ax)
add_split_lines(ax)

final_ret = snap["cum_pct"].iloc[-1]
color_ret = COLORS["profit"] if final_ret >= 0 else COLORS["loss"]
ax.annotate(f"최종: {final_ret:+.2f}%",
            xy=(snap.index[-1], final_ret),
            xytext=(-60, 10), textcoords="offset points",
            color=color_ret, fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=color_ret, lw=1.2))

# ── 2. 현금 비율 ──────────────────────────────────────────────
ax = ax_cash
ax.fill_between(snap.index, snap["cash_pct"],
                color=COLORS["cash"], alpha=0.5)
ax.plot(snap.index, snap["cash_pct"],
        color=COLORS["cash"], lw=1.2)
ax.axhline(10, color="#FFD54F", lw=0.8, ls=":", alpha=0.8)
ax.text(snap.index[5], 11.5, "최소 현금 10%",
        color="#FFD54F", fontsize=7)
ax.set_title("현금 비율 (%)", color=COLORS["text"], fontsize=11, pad=8)
ax.set_ylabel("%", color=COLORS["subtext"], fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
ax.grid(True, color=COLORS["grid"], lw=0.5)
style_xaxis(ax)

# ── 3. 보유 포지션 수 ─────────────────────────────────────────
ax = ax_pos
ax.fill_between(snap.index, snap["n_positions"],
                color=COLORS["pos_area"], alpha=0.6, step="mid")
ax.step(snap.index, snap["n_positions"],
        color=COLORS["pos_area"], lw=1.2, where="mid")
ax.set_title("보유 포지션 수", color=COLORS["text"], fontsize=11, pad=8)
ax.set_ylabel("종목 수", color=COLORS["subtext"], fontsize=9)
ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
ax.grid(True, color=COLORS["grid"], lw=0.5)
style_xaxis(ax)

# ── 4. MDD ────────────────────────────────────────────────────
ax = ax_mdd
ax.fill_between(snap.index, snap["drawdown"], 0,
                color=COLORS["mdd"], alpha=0.4)
ax.plot(snap.index, snap["drawdown"],
        color=COLORS["mdd"], lw=1.3)
ax.set_title("낙폭 / MDD (%)", color=COLORS["text"], fontsize=12, pad=8)
ax.set_ylabel("%", color=COLORS["subtext"], fontsize=9)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
ax.grid(True, color=COLORS["grid"], lw=0.5)
style_xaxis(ax)
add_split_lines(ax)
mdd_val = snap["drawdown"].min()
mdd_idx = snap["drawdown"].idxmin()
ax.annotate(f"MDD: {mdd_val:.2f}%",
            xy=(mdd_idx, mdd_val),
            xytext=(30, -20), textcoords="offset points",
            color=COLORS["mdd"], fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COLORS["mdd"], lw=1.1))

# ── 5. 롤링 샤프 (30일) ───────────────────────────────────────
ax = ax_sharpe
rs = snap["roll_sharpe"].dropna()
pos_s = rs >= 0
ax.fill_between(rs.index, rs, 0, where=pos_s,
                color=COLORS["profit"], alpha=0.3)
ax.fill_between(rs.index, rs, 0, where=~pos_s,
                color=COLORS["loss"],   alpha=0.3)
ax.plot(rs.index, rs, color=COLORS["portfolio"], lw=1.1)
ax.axhline(0, color=COLORS["subtext"], lw=0.7, ls="--")
ax.set_title("롤링 샤프 지수 (30일)", color=COLORS["text"], fontsize=11, pad=8)
ax.set_ylabel("Sharpe", color=COLORS["subtext"], fontsize=9)
ax.grid(True, color=COLORS["grid"], lw=0.5)
style_xaxis(ax)

# ── 6. 전략 분포 파이 ─────────────────────────────────────────
ax = ax_strat
strat_counts = snap["strategy"].value_counts()
strat_colors = [COLORS.get(s, "#aaa") for s in strat_counts.index]
wedges, texts, autotexts = ax.pie(
    strat_counts.values,
    labels=strat_counts.index,
    colors=strat_colors,
    autopct="%1.1f%%",
    startangle=140,
    textprops={"color": COLORS["text"], "fontsize": 9},
    wedgeprops={"edgecolor": COLORS["bg"], "linewidth": 1.5},
)
for at in autotexts:
    at.set_color(COLORS["bg"])
    at.set_fontsize(8)
ax.set_title("전략 분포 (일수 기준)", color=COLORS["text"], fontsize=11, pad=8)

# ── 7. 거래 수익률 분포 ───────────────────────────────────────
ax = ax_ret
if not trades.empty and "return_pct" in trades.columns:
    ret_vals = trades["return_pct"].dropna()
    bins = np.linspace(ret_vals.min() - 0.5, ret_vals.max() + 0.5, 35)
    profit_mask = ret_vals >= 0
    ax.hist(ret_vals[profit_mask],  bins=bins, color=COLORS["profit"],
            alpha=0.7, label=f"수익 ({profit_mask.sum()}건)")
    ax.hist(ret_vals[~profit_mask], bins=bins, color=COLORS["loss"],
            alpha=0.7, label=f"손실 ({(~profit_mask).sum()}건)")
    ax.axvline(ret_vals.mean(), color="#FFD54F", lw=1.2, ls="--",
               label=f"평균 {ret_vals.mean():+.2f}%")
    ax.set_xlabel("%", color=COLORS["subtext"], fontsize=9)
    ax.set_title("거래 수익률 분포", color=COLORS["text"], fontsize=11, pad=8)
    ax.legend(fontsize=8, facecolor=COLORS["panel"],
              labelcolor=COLORS["text"], edgecolor=COLORS["grid"])
    ax.grid(True, color=COLORS["grid"], lw=0.5, axis="y")

    # 청산 사유 표시
    if "exit_reason" in trades.columns:
        reason_txt = trades["exit_reason"].value_counts().to_string()
        ax.text(0.98, 0.97, "청산 사유\n" + reason_txt,
                transform=ax.transAxes, fontsize=7,
                color=COLORS["subtext"], va="top", ha="right",
                family="monospace")
else:
    ax.text(0.5, 0.5, "거래 데이터 없음",
            ha="center", va="center",
            color=COLORS["subtext"], fontsize=11,
            transform=ax.transAxes)
    ax.set_title("거래 수익률 분포", color=COLORS["text"], fontsize=11, pad=8)

# ── 8. 핵심 지표 표 ───────────────────────────────────────────
ax = ax_metric
ax.axis("off")

# metrics dict 파싱
def fmt(k, v):
    try:
        f = float(v)
        if "수익률" in k or "MDD" in k or "승률" in k or "수익" in k:
            return f"{f:+.2f}%"
        if "Sharpe" in k:
            return f"{f:.4f}"
        if "자산" in k:
            return f"{f:,.0f}원"
        if "거래 수" in k:
            return f"{int(f):,}건"
        return str(v)
    except:
        return str(v)

display_keys = [
    "누적 수익률 (%)", "MDD (%)", "Sharpe Ratio",
    "최종 자산 (원)", "총 거래 수", "승률 (%)", "평균 거래 수익 (%)"
]

rows = []
for k in display_keys:
    if k in metrics:
        v = metrics[k]
        rows.append([k, fmt(k, v)])

# 직접 계산한 값으로 보완
if not rows:
    rows = [
        ["누적 수익률", f"{snap['cum_pct'].iloc[-1]:+.2f}%"],
        ["MDD",         f"{snap['drawdown'].min():.2f}%"],
        ["총 거래 수",  f"{len(trades):,}건" if not trades.empty else "N/A"],
    ]

tbl = ax.table(
    cellText=rows,
    colLabels=["지표", "값"],
    cellLoc="center",
    loc="center",
    bbox=[0.0, 0.05, 1.0, 0.90],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)

for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor(COLORS["panel"] if r > 0 else "#21262D")
    cell.set_edgecolor(COLORS["grid"])
    cell.set_text_props(color=COLORS["text"])
    if r > 0 and c == 1:
        val_str = cell.get_text().get_text()
        if val_str.startswith("+"):
            cell.set_text_props(color=COLORS["profit"])
        elif val_str.startswith("-"):
            cell.set_text_props(color=COLORS["loss"])

ax.set_title("핵심 성과 지표", color=COLORS["text"], fontsize=11, pad=8)

# ── 제목 ─────────────────────────────────────────────────────
fig.suptitle(
    "백테스팅 종합 리포트  |  Greedy + GA + Optimal Stopping",
    color=COLORS["text"], fontsize=15, fontweight="bold", y=0.97
)

plt.savefig(OUTPUT, dpi=150, bbox_inches="tight",
            facecolor=COLORS["bg"])
print(f"✓ 저장 완료: {OUTPUT}")
plt.show()