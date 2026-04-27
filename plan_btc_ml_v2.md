# BTC ML v2: 알고리즘 고도화 + 자율 거래 봇 로드맵

## Context

v1 (이미 구현됨)에서 확인된 사실:
- 파이프라인은 동작: data → feature → train → backtest → report
- smoke test (3개월 학습): **rank-IC = 0.0101**, sign accuracy 50.1%, return -29%
- 핵심 병목 식별: 모델 edge < round-trip cost(0.20%) → fee로 손실

**v2 목표**: rank-IC ≥ 0.04, daily-Sharpe ≥ 1.5, max DD ≤ 25% on 6+ year walk-forward, 그리고 OKX paper trading에서 동일 분포 검증 후 실거래 자율 봇 운영.

**현실 점검**: 1분봉 BTC 단일 종목으로 무한히 끌어올리는 것은 어렵다. 학계/실무 벤치마크상 "잘 만든 1m crypto 모델"의 Sharpe(daily)는 1.5–2.5 범위. 그 이상은 마이크로구조(L2 orderbook, tick) 또는 cross-asset 데이터를 추가해야 가능. 이번 로드맵은 1.5–2.5 범위 도달이 현실적 천장이라는 가정에서 설계.

---

## 핵심 전략 (영향력 큰 순서)

각 항목 옆 **[+est IC]**는 v1 baseline rank-IC 대비 추가 기대치(BTC 1m 회귀 기준 통상치). 합산이 아닌 누적 효과.

1. **Triple Barrier Labeling** [+0.005~0.010] — 단순 forward return 대신 "stop/target 중 어디 먼저 도달하는지"로 학습. 라벨 노이즈 대폭 감소.
2. **Meta-Labeling (이중 모델)** [+0.010~0.020] — 1차 모델이 방향, 2차 모델이 "이 신호로 정말 거래할지"를 결정. 거짓 양성 제거 효과 가장 큼.
3. **Volatility-Normalized Target** [+0.005~0.010] — `ret / realized_vol`로 학습. 시장 regime 변화에 robust.
4. **Multi-horizon 앙상블** [+0.005~0.010] — 5m/15m/60m 각각 학습 → consensus 시에만 거래.
5. **Funding rate + OI feature** [+0.003~0.008] — OKX의 funding/OI history는 1m보다는 8h/4h 단위지만 강력한 alpha source.
6. **Cost-aware thresholding** — 예측치에서 `expected_cost`를 차감 후 임계값 비교. 거래수 감소 + 평균 trade edge 증가.
7. **Optuna 하이퍼파라미터 + feature selection** [+0.003~0.008] — 노이즈 피처 제거.
8. **CPCV / Deflated Sharpe** — IC 부풀림 방지. 진짜 edge만 통과.

목표 누적: v1 0.0101 → v2 0.04+ (4배 향상이지만 위 1+2+3 만으로도 거의 달성 가능).

---

## Phase 1: Edge 강화 (3–4주, 백테스트 확률 향상의 90%)

### 1.1 — 풀 데이터 백테스트 (baseline 확립)
- `scripts/fetch_btc_history.py --start 2020-01-01` 실행 (~2시간, 백그라운드)
- v1 그대로 walk-forward (50+ folds) 학습 + 백테스트
- **검증 게이트**: 평균 rank-IC, Sharpe(daily), max DD를 baseline 메트릭으로 기록
- 산출: `backtest/results/btc_ml_v1_full/report.html` — v2가 이를 능가해야 함

### 1.2 — Triple Barrier Labeling
**신규 파일**: [strategy/ml_labels.py](strategy/ml_labels.py) — Lopez de Prado triple barrier 구현
- 각 bar에서 ATR 기반 동적 상/하 barrier 설정 (예: ±1.5×ATR)
- 시간 barrier: max 60-bar (1시간)
- 라벨: `{+1: target hit, -1: stop hit, 0: time expired}` 또는 회귀 변형 (실현 ret을 barrier 시점에서 측정)
- 회귀 변형 추천 (분류 시 임계값 양상이 더 noisy)

**수정**: [strategy/ml_features.py](strategy/ml_features.py) — `target_column()`에 `y_tb` 추가, 기존 `y_15`와 병행 학습 가능하게.

**검증 게이트**: 동일 모델로 학습했을 때 rank-IC가 baseline보다 높아야 함.

### 1.3 — Volatility-Normalized Target
- `y_norm_15 = log(close[t+15]/close[t]) / realized_vol_60[t]`
- 기존 `y_15`와 병렬 학습 → out-of-sample IC 비교
- **승자 채택 기준**: rank-IC가 5% 이상 우월할 때

### 1.4 — Multi-horizon 앙상블
- 5분/15분/60분 각각 모델 학습 (`y_5`, `y_15`, `y_60`)
- 거래 결정 로직:
  - 3개 중 2개 이상이 같은 방향 + 평균 |ŷ_norm| > θ → 거래
  - 그 외 flat
- **신규 파일**: [strategy/ml_ensemble.py](strategy/ml_ensemble.py)
- **검증 게이트**: 단일 모델 대비 trade count 30~50% 감소 + 평균 trade edge 증가.

### 1.5 — Meta-Labeling (이중 모델 — 가장 영향 큰 단계)
**구조**:
- **1차 모델**: 기존 LGB 회귀 → 방향 시그널
- **2차 모델**: feature = [1차 모델 예측치, 모델 confidence, 시장 regime feature, 시간대] → label = 해당 거래의 실현 PnL이 양수였는지 binary classification
- 거래 결정: 1차가 신호 + 2차가 "거래해도 좋다"고 동의할 때만

**신규 파일**: [strategy/ml_meta_label.py](strategy/ml_meta_label.py)

**검증 게이트**: 1차 단독 대비 거래수 감소 (~40%) + Sharpe 증가 (~30%+).

### 1.6 — Cost-Aware Thresholding
**수정**: [backtest/ml_backtester.py](backtest/ml_backtester.py)
- 진입 조건: `ŷ - sign(ŷ) × (2 × fee + 2 × slippage + funding_carry_estimate) > θ`
- `funding_carry_estimate` = 보유 예정 시간 × 평균 funding rate 부호
- 그리고 별도로 `θ`도 fold별 학습 분포 기반으로 동적 조정 (75 percentile → 80 percentile 시도)

**검증 게이트**: 동일 모델에서 거래수 감소하면서도 win rate 증가.

### 1.7 — Optuna + Feature Selection
**신규 파일**: [strategy/ml_optuna.py](strategy/ml_optuna.py)
- Optuna 30~50 trials walk-forward CV (`min_data_in_leaf`, `learning_rate`, `num_leaves`, `lambda_l2`, `feature_fraction`)
- SHAP 기반 feature 중요도 분석 → 하위 30% 피처 제거
- LightGBM의 `feature_importance(importance_type="gain")`로 빠른 1차 컷, 그 다음 SHAP

**검증 게이트**: hold-out 마지막 1개월에서 rank-IC 5% 이상 향상.

### 1.8 — Funding Rate / Open Interest 피처
**신규 파일**: [data/okx_funding_loader.py](data/okx_funding_loader.py)
- OKX `fetch_funding_rate_history` 사용 (8시간 단위 → forward-fill로 1m alignment)
- `fetch_open_interest_history` (4시간 단위) — OI 변화율 feature
- 추가 피처: `funding_rate_current`, `funding_rate_chg_24h`, `oi_chg_24h`, `oi_zscore_7d`

**수정**: [strategy/ml_features.py](strategy/ml_features.py) — 이 피처들 흡수 (옵션 플래그로 ON/OFF)

**검증 게이트**: feature importance plot에 funding/OI 피처가 상위 30개에 진입하면 효과 인정.

---

## Phase 2: 검증 강건성 (1주)

### 2.1 — Combinatorial Purged Cross-Validation
**신규 파일**: [strategy/ml_cpcv.py](strategy/ml_cpcv.py)
- 6개월 단위 group, k=6, 2개씩 hold out → 15개 path
- 각 path별 metric 분포 → IC 분산 확인
- 단일 walk-forward의 점추정보다 신뢰도 높음

### 2.2 — Deflated Sharpe Ratio
- 우리가 시도한 모델 변형 수 N (Triple Barrier, Vol-norm, Multi-horizon, Optuna trials, ...) 기록
- DSR = `Sharpe / sqrt(1 + (skew/2 - 1) × var(Sharpe))` × Bonferroni 정정
- 실제 알파 vs 다중검정 우연 구분

### 2.3 — Stress Test 매트릭스
**신규 파일**: [backtest/stress_test.py](backtest/stress_test.py)
- Fee +50%, +100% 시 PnL
- Slippage 2x, 5x 시 PnL
- 1초 / 5초 / 30초 latency 추가 시 PnL
- 2022 베어 / 2024 불 분리 backtest
- "어떤 시장 환경에서 무너지는가"를 보고서에 자동 포함

### 2.4 — Regime Breakdown
**수정**: [backtest/ml_report.py](backtest/ml_report.py)
- 기존 ADX bucket별 표 외에:
  - 일별 변동성 quartile별 Sharpe
  - 주말 vs 평일
  - US/Asia 세션별
  - Funding 양/음수 구간별

---

## Phase 3: 비용 최적화 (1주)

### 3.1 — 실제 Funding Rate History
- 1.8에서 다운로드한 history를 `_funding_charge`에 주입
- v1의 0.01% 상수 가정 제거. 부호 (long↔short) 정확히 처리.

### 3.2 — Maker Fee 시뮬레이션 (선택)
- v1은 taker(0.05%) 가정. OKX maker는 0.02% (또는 음수 rebate).
- 진입을 limit order at bar.close 기준 ± 1bp로 가정 → 일정 비율 미체결 시뮬레이션
- **현실 경고**: 1m 봉 단위에서 limit fill을 정확히 모사하긴 어렵다. 보수적으로 80% 체결율로 시뮬.
- 효과 크면 실거래에서도 maker 우선 시도

### 3.3 — Volatility-Aware Position Sizing
**신규 파일**: [risk/vol_target_sizer.py](risk/vol_target_sizer.py)
- Target portfolio vol = 일일 1.5% (annualized ~24%)
- 현재 시장 realized_vol 기반으로 position_pct 조정
- v1의 고정 10% → 동적 5~15%
- 변동성 폭증 시 자동 축소 → max DD 감소 효과

---

## Phase 4: 자율 거래 봇 구축 (3–4주)

전제: Phase 1–3에서 검증된 모델만 라이브로 진입. baseline rank-IC < 0.025면 라이브 보류.

### 4.1 — Live Inference Loop
**신규 파일**: [core/ml_trader.py](core/ml_trader.py)
- 1분 주기 (cron 또는 asyncio loop):
  1. 최신 1m 봉 1개 fetch (`OKXRestClient.fetch_ohlcv`)
  2. 최근 240+15분 봉을 메모리 캐시 + 신규 봉 append
  3. `build_features()` 호출 (마지막 row만 추출)
  4. 활성 fold 모델로 inference (`booster.predict`)
  5. 의사결정 → 1차 + 2차 모델 + cost-aware 임계 → 진입/청산
- 기존 [core/bot_engine.py](core/bot_engine.py)와 별도 모드(env var `BOT_MODE=ML`)로 분기.

### 4.2 — OKX Paper Trading 검증 (필수 게이트)
- OKX demo (`x-simulated-trading=1` header — 이미 [data/okx_rest_client.py](data/okx_rest_client.py)에 구현됨) 환경에서 최소 **2주 운영**
- 매일 백테스트 vs 라이브 분포 비교:
  - 진입 빈도가 ±20% 안인지
  - 평균 ŷ 분포가 ±15% 안인지
  - 일별 PnL이 백테스트 분포의 95% CI 안인지
- 통과 못하면 모델/환경 진단

### 4.3 — Position Reconciliation
**신규 파일**: [execution/reconciliation.py](execution/reconciliation.py)
- 매 분 (또는 trade event 직후): 봇 내부 포지션 vs `OKXRestClient.fetch_positions` 비교
- 불일치 발견 시:
  - 사이즈 차이 < 1% → 봇 상태를 거래소 기준으로 동기화
  - 사이즈 차이 ≥ 1% → 안전 모드 진입 (거래 중지 + 알림)
- 외부 수동 개입, 거래소 부분 청산, 네트워크 실패 등 대응

### 4.4 — Kill Switch + Circuit Breaker
**수정**: [risk/risk_manager.py](risk/risk_manager.py) → ML 컨텍스트로 확장
- 자동 정지 트리거:
  - 일별 PnL < -3%
  - 7일 누적 PnL < -8%
  - 연속 손실 5회
  - 라이브 sign accuracy(최근 200 trades) < 47%
  - 라이브 ŷ 분포가 학습 분포 KL-div > 임계
  - 거래소 API 에러율 > 5% (1시간 윈도우)
- 정지 시: 모든 포지션 청산 + Telegram/Discord 알림 + 봇 서비스 중지
- 재가동: 사람이 명시적으로 `--resume` 플래그로 재시작

### 4.5 — Online Learning (주간 재학습)
**신규 파일**: [scripts/weekly_retrain.py](scripts/weekly_retrain.py)
- cron: 매주 일요일 03:00 UTC
- 직전 30일 데이터 추가 fetch → feature 재빌드 → 마지막 fold 모델을 `init_model`로 warm-start retrain
- Validation: 직전 7일 hold-out에서 rank-IC > 0.015이면 새 모델 promotion, 아니면 유지
- 모델 버전: `models/btc_lgb_v1/fold_YYYY-MM_v{N}.lgb`로 저장 + `current.lgb` symlink

### 4.6 — 모니터링 + 알림
- 기존 [notifications/telegram.py](notifications/telegram.py), [notifications/discord.py](notifications/discord.py) 재사용
- 알림 종류:
  - 진입/청산 (시그널 + ŷ + 수익률)
  - 일별 PnL 요약 (UTC 00:00에)
  - kill switch 발동
  - 주간 재학습 완료
  - reconciliation 실패
- 대시보드 (선택): 기존 [web/](web/) Next.js 앱에 ML 패널 추가 (live equity curve, 최근 trades, kill switch 상태)

---

## 검증 게이트 요약

| Phase | 진입 게이트 | 통과 게이트 |
|---|---|---|
| 1.1 baseline | — | rank-IC, Sharpe, DD를 v1 메트릭으로 고정 |
| 1.2 Triple Barrier | 1.1 통과 | rank-IC ≥ baseline + 0.005 |
| 1.3 Vol-norm | 1.2 통과 | rank-IC ≥ baseline + 0.010 |
| 1.4 Multi-horizon | 1.3 통과 | trade count -30~50%, Sharpe ≥ baseline + 0.3 |
| 1.5 Meta-label | 1.4 통과 | trade count -40%, Sharpe ≥ baseline + 0.5 |
| 1.6 Cost-aware | 1.5 통과 | win rate +5%p |
| 1.7 Optuna | 1.6 통과 | rank-IC ≥ phase1.5 + 0.005 |
| 1.8 Funding/OI | 1.7 통과 | feature importance 상위 30위 진입 + IC 향상 |
| Phase 2 | Phase 1 누적 통과 | DSR > 0, CPCV 분산 < 50% mean |
| Phase 3 | Phase 2 통과 | stress test에서 fee 2x 시 Sharpe > 0.7 |
| **Live promotion** | rank-IC ≥ 0.025, Sharpe(daily) ≥ 1.5 | 2주 paper trading 분포 일치 |
| 실거래 promotion | paper 통과 | 30일 paper Sharpe ≥ 1.0, 사용자 명시 승인 |

---

## 신규/수정 파일 (전체)

**신규 (Phase 1)**: ml_labels.py, ml_ensemble.py, ml_meta_label.py, ml_optuna.py, okx_funding_loader.py
**신규 (Phase 2)**: ml_cpcv.py, stress_test.py
**신규 (Phase 3)**: vol_target_sizer.py
**신규 (Phase 4)**: ml_trader.py, reconciliation.py, weekly_retrain.py

**수정**: ml_features.py(target 추가), ml_model.py(triple barrier target 옵션), ml_backtester.py(cost-aware), ml_report.py(regime breakdown 확장), risk_manager.py(ML kill switch), bot_engine.py(BOT_MODE=ML 분기)

**재사용 (수정 없음)**: indicators.py, performance.py, db_manager.py, telegram.py, discord.py, okx_rest_client.py

---

## 시간/비용 추정

- Phase 1: 3–4주 (가장 중요, edge 90% 여기서)
- Phase 2: 1주
- Phase 3: 1주
- Phase 4: 3–4주 (paper trading 검증 2주 + 인프라 1–2주)
- **총 8–10주**, 단 본인 작업 외 데이터 다운로드/학습은 백그라운드.

학습 비용: M-series Mac에서 단일 fold ~3–5분, 60 fold × 매주 retrain × 26주 → 비싸지 않음. 단 Optuna는 50 trial × 60 fold = 수일 걸릴 수 있어 한 번에만 실행.

---

## 주의사항 (반복적으로 회귀하기 쉬운 함정)

1. **Lookahead leakage 재발 방지**: 새 피처 추가할 때마다 `tests/test_ml_features.py` 테스트 재실행 의무화.
2. **Embargo 누락**: triple barrier는 학습 끝에서 max barrier 시간 만큼 더 drop해야 함. 60분 → 사실상 120분 embargo로 확장.
3. **Survivorship 없음**: BTC 단일 종목이라 OK. cross-symbol로 확장 시 상장폐지 종목 포함 필수.
4. **Backtest fees 보수성**: maker rebate 가정은 라이브에서 못 받을 수 있다 (마이크로구조 불일치). 보수적 시뮬레이션 권장.
5. **Funding rate 부호**: long은 funding rate 양수일 때 지불, 음수일 때 수령. v1의 "항상 비용" 가정은 보수적이지만 한쪽 방향으로 편향. 실제 history로 교체 필수.
6. **Online learning catastrophic forgetting**: warm-start retrain이 직전 30일에만 과적합되면 장기 패턴 잃을 수 있다. monthly full retrain + weekly incremental 병행 권장.
7. **Live ↔ Backtest gap**: paper trading 2주는 협상 불가. 한 번이라도 분포 mismatch 발견하면 모델 promote 보류.
