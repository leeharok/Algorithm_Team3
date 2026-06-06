1. 6/3/26 HD현대 빼고 시도
누적 수익률 (%),-1.02
MDD (%),-18.77
Sharpe Ratio,-0.3978
최종 자산 (원),98978951.0
총 거래 수,217
승률 (%),57.6
평균 거래 수익 (%),0.49
청산 사유 분포,"{'secretary_trigger': 111, 'stop_loss': 65, 'force_liquidate': 40, 'take_profit': 1}"

지표 설명
누적 수익률 -1.02% — 1억 넣었다가 최종적으로 약 102만원 손해
MDD -18.77% — 운용 기간 중 최악의 시점에 고점 대비 1877만원까지 빠진 적 있음. 수익률이 -1%여도 중간에 이렇게 크게 흔들렸다는 뜻
Sharpe Ratio -0.3978 — 위험 대비 수익이 마이너스. 그냥 은행에 넣어두는 게 나았다는 의미
승률 57.6% — 217번 거래 중 125번은 수익으로 끝냄
평균 거래 수익 +0.49% — 거래 한 번당 평균은 플러스인데 누적이 마이너스인 건, 크게 잃은 몇 번이 전체를 갉아먹은 것

청산 사유 설명
secretary_trigger 111건 — P*(관찰 구간 최고가)를 넘는 순간 매도. "이 정도면 충분히 올랐다" 판단. 가장 많이 발생
stop_loss 65건 — 수익률 -8% 찍으면 강제 손절. 손실 제한용인데 65번이나 터진 게 문제
force_liquidate 40건 — 20일 다 됐는데 secretary_trigger도 stop_loss도 안 걸려서 그냥 강제 청산. 애매한 상태로 끝난 거래들
take_profit 1건 — +25% 익절 조건인데 딱 1번만 터짐. secretary_trigger가 훨씬 먼저 팔아버려서 큰 수익을 못 먹는 구조

2. 6/5/26 코드 수정 
누적 수익률 (%),21.95
MDD (%),-10.6
Sharpe Ratio,0.5647
최종 자산 (원),121948497.0
총 거래 수,41
승률 (%),56.1
평균 거래 수익 (%),4.61
청산 사유 분포,"{'stop_loss': 17, 'take_profit': 14, 'secretary_trigger': 10}"

누적 수익률 +21.95%
1억이 1억 2195만으로 늘었어요. 2023~2025 기간 동안 22%면 연 환산 약 10~11% 수준이에요. 코스피 같은 기간 수익률이랑 비교해볼 필요 있어요.
MDD -10.60%
고점 대비 최대 1060만원까지 빠진 적 있는데, 전보다 절반 가까이 줄었어요. 보유 기간 늘리고 stop_loss를 -12%로 여유 준 게 효과 있었어요.
Sharpe Ratio 0.5647
0 이상으로 올라왔어요. 은행 이자보다 나은 수익을 위험 대비로 냈다는 뜻이에요. 1.0 이상이면 아주 좋고, 0.5~1.0이면 준수한 수준이에요.
총 거래 수 41건
217건 → 41건으로 크게 줄었어요. 백테스팅 기간이 2023~2025(약 2.5년)으로 줄어든 영향도 있지만, window 늘리고 force_liquidate 없앤 덕에 불필요한 매매가 줄었어요

take_profit이 14건으로 크게 늘었어요. 수정 전엔 217건 중 1건이었는데 41건 중 14건으로 완전히 달라진 거예요. secretary 기준을 P*×1.15로 올려서 충분히 오를 때까지 들고 있다가 +20%에서 익절하는 구조가 제대로 작동한 거예요.
stop_loss가 여전히 제일 많긴 해요 (41%).
17건 × -12% = 상당한 손실이 쌓이고 있어요. 근데 평균 수익이 +4.61%라서 take_profit이 이걸 상쇄하고도 남는 구조예요.
secretary_trigger 10건은 적당해요.
P*×1.15 못 넘고 기준이 낮아지면서 팔린 경우인데, 이 중 수익인 게 얼마나 되는지 bt_full_trades.csv 열어보면 확인할 수 있어요.

1️.GA 학습/백테스팅 기간 분리 (data_pipeline.py, backtester.py)
- 문제: GA가 학습한 데이터로 매매까지 해서 의미없는 결과
- 변경: Train 2018~2021 / Val 2022 / 백테스팅 2023~2025

2️.GA 배분 기준 변경 (backtester.py)
- 문제: 손실 나도 초기자본 기준으로 같은 금액 매수
- 변경: 현재 자산 기준으로 배분 (잃으면 줄고 벌면 늘어남)

3️.보유 윈도우 및 파라미터 변경 (optimal_stopping.py)
- 문제: 20일 너무 짧아 추세 못 탐
- 변경: window 20→60일 / observe 7→22일 / stop-loss -8→-12% / take-profit +25→+20%

4️.Secretary 기준 변경 (optimal_stopping.py)
- 문제: P* 1원만 넘어도 즉시 매도 → take_profit 217건 중 1건밖에 안 터짐
- 변경: P*×1.15 넘을 때 매도 / 60일마다 기준 5%씩 완화 (최소 P*×1.00)

5️. force_liquidate 제거 (optimal_stopping.py)
- 문제: 상승 중에도 20일 되면 강제 청산 (전체의 18%)
- 변경: stop_loss / take_profit / secretary 조건 걸릴 때까지 보유

3. optimal_stopping/GA 수정

optimal_stopping.py

ObserveMode, DynamicObserveCalculator, UCB1ObserveSelector 클래스 추가
StoppingConfig에 observe_mode 파라미터 추가 (기본값 "fixed")
window=60, observe=22, stop=-12%, take=+20%
force_liquidate 제거
secretary 기준 P*×1.15 → 60일 텀마다 5%씩 완화
main 블록 import 버그 수정

backtester.py

argparse 추가 (--observe-mode, --start, --end, --capital)
Rolling GA:6개월마다 자동으로 재학습
data/ga_rolling/ 폴더에 weight저장