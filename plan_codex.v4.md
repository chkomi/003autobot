# 003 Autobot Codex Plan (v4.1) — BTC 2020+ 검증 기반 ML 자율거래 로드맵

> **v3 → v4 변경 요약**
> - 메인 거래 엔진: 룰베이스 multi-symbol → **ML(LightGBM 회귀) BTC 전용**으로 전환
> - v1 ML 백테스트 파이프라인 이미 구현됨 ([data/okx_history_loader.py](data/okx_history_loader.py), [strategy/ml_features.py](strategy/ml_features.py), [strategy/ml_model.py](strategy/ml_model.py), [backtest/ml_backtester.py](backtest/ml_backtester.py))
> - **Phase 0 추가**: ML Edge 강화 (Triple Barrier · Meta-labeling · Vol-Norm · Multi-horizon)
> - v3의 Trade Journal / Reflection Loop / Sandbox 골격은 ML 컨텍스트로 재해석해 유지
> - Bandit 대상 재정의: 룰베이스 전략 묶음 → **ML 모델 variants + 룰베이스 fallback** 의 동적 가중
> - 가드레일 강화: ML 분포 drift 감지, position reconciliation, kill switch 이중화
> - **v4.1 반영**: 데이터 무결성 게이트, 2020+ OOS 분할, 기대값(EV) 기반 진입, Paper/Small-cap 승격 기준 강화
> 작성일: 2026-04-27

---

## 1. 배경 및 문제 정의

### v3까지 진행 상황

v3에서 그려진 "자율학습 거래 루프"의 골격(Trade Journal → Reflection → Bandit → Sandbox) 자체는 유효하다. 단, v3는 **이미 존재하는 룰베이스 시그널**(`trend_filter`/`momentum_trigger`/`micro_confirmation` triple confirmation)을 가정하고 그 위에 학습 루프를 얹는 설계였다.

### v1 ML 구현으로 드러난 사실

v1 BTC ML 파이프라인 (1m bar + LightGBM 회귀, 15분 forward log-return) 구현 후 smoke test (3개월 학습) 결과:

| 메트릭 | 값 | 의미 |
|---|---|---|
| rank-IC | **0.0101** | 통계적으론 양수지만 임계값 0.01 직전 |
| sign accuracy | 50.1% | 사실상 random (학습 데이터 부족) |
| 반환률 | -29.14% | round-trip cost 0.20% > 모델 edge |
| 평균 보유 | 49분 | 60-bar timeout 근접 |

**핵심 결론**: 룰베이스 멀티전략을 이대로 자율화해봐야 천장이 낮다. **ML 엔진을 메인으로 격상**하고, v3의 자율학습 인프라(저널/반성/밴딧/샌드박스)를 ML 모델 진화에 맞게 재구성하는 것이 v4의 출발점.

### 현재 로컬 상태 (2026-04-27 기준)

| 항목 | 현재 | 해석 |
|---|---|---|
| raw 1m 데이터 | `2026-02`, `2026-03`, `2026-04` | 2020+ 학습에는 절대 부족 |
| feature 데이터 | `features/btc_1m_v1/2026-02~04.parquet` | smoke test 전용 |
| 모델 fold | `fold_2026-04.lgb` 1개 | walk-forward라고 부르기에는 표본 부족 |
| v1 백테스트 | 315 trades, -29.14%, win rate 30.16%, PF 0.28, MDD 29.54% | 실거래 후보 아님 |

**즉시 보완점**: 현재 `fetch_btc_history.py`는 저장된 최신 timestamp 이후부터 resume한다. 이미 2026년 데이터가 있는 상태에서 `--start 2020-01-01`만 실행하면 2020~2026 과거 구간을 backfill하지 못할 수 있다. 따라서 Phase -1에서 `--mode backfill` 또는 `--ignore-resume`/`--gap-fill` 옵션을 먼저 추가한 뒤 풀 데이터를 수집한다.

### 왜 BTC 단일 종목인가
- 1m bar × 6.4년(2020~2026) = 3.36M 행 — 단일 종목으로도 학습 데이터 풍부
- 마이크로구조 / funding / OI 데이터가 가장 풍부한 자산
- 멀티심볼 일반화는 규모의 경제 효과 작고 cross-asset bias 위험 큼 (v5 이후 검토)

### 의사결정 원칙

이 계획의 목표는 단순 승률을 높이는 것이 아니다. 낮은 승률이어도 평균 이익이 평균 손실보다 충분히 크면 좋은 전략이고, 높은 승률이어도 비용·펀딩·꼬리손실에 먹히면 실패 전략이다.

최종 진입 조건은 아래 기대값이 양수일 때만 통과한다.

```
EV = p_win * avg_win - (1 - p_win) * avg_loss - expected_fee - expected_slippage - expected_funding
```

운영 판단 우선순위는 `OOS 기대값 > Max DD > Profit Factor > Sharpe/Calmar > 승률` 순서로 둔다.

### 핵심 알고리즘 구조

v4.1의 최종 형태는 **Regime-Gated Hybrid + ML Meta-Labeling**이다.

1. 룰 기반 엔진이 후보 진입만 만든다. 예: 4H 추세, 1H 모멘텀, 15M/1M 미시 구조.
2. Regime gate가 `trend`, `range`, `high-vol`, `event-risk` 상태를 판별하고 불리한 구간은 flat 처리한다.
3. 1차 ML 모델은 방향과 기대 움직임을 예측한다.
4. 2차 meta-label 모델은 "이 후보를 실제로 거래할지"를 판단한다.
5. Cost-aware EV gate가 수수료·슬리피지·펀딩을 뺀 기대값이 양수일 때만 주문을 허용한다.
6. Bandit은 regime별로 `ML primary`, `ensemble`, `meta-label`, `rule fallback`, `flat` 중 어느 arm을 쓸지 조정한다.

---

## 2. 2026년 트렌드: 자율거래 + ML edge 강화

### 2.1 v3 트렌드 (유지)

1. Trade Journal + Reflection Loop
2. Regime Detector → 전략 스위칭
3. Thompson Sampling 밴딧
4. LLM 멀티에이전트 협업 (옵션)
5. 샌드박스 승격 파이프라인
6. 설명가능성(XAI)
7. 가드레일 우선

### 2.2 ML edge 강화 패턴 (v4 추가)

quant HFT 펀드(XTX Markets, HRT, Two Sigma 등)와 학계(Lopez de Prado 2018+) 합의:

1. **Triple Barrier Labeling** — 단순 forward return 대신 "stop/target 중 어디 먼저 도달하는지"로 라벨링. 라벨 노이즈 대폭 감소.
2. **Meta-Labeling** — 1차 모델은 방향, 2차 모델은 "이 신호로 정말 거래할지" 판단. 거짓 양성 제거 효과 가장 큼 (단일 항목 최대 영향).
3. **Volatility-Normalized Target** — `ret / realized_vol` 학습. regime 변화에 robust.
4. **Multi-horizon Ensemble** — 5분/15분/60분 합의 시에만 거래.
5. **Cost-Aware Thresholding** — 예상 cost 차감 후 임계값 비교.
6. **Combinatorial Purged CV + Deflated Sharpe** — 다중검정 부풀림 방지.
7. **Microstructure / Funding / OI feature** — 가격만으로 부족한 alpha 보충.
8. **BTC 온체인 피처** — 주식의 재무제표에 해당하는 체인 고유 신호. MVRV, NVT, Realized Cap, Exchange Netflow, SOPR, Active Addresses 등. 가격·파생상품 데이터와 독립적인 alpha 레이어.

### 2.3 멀티코인 펀더멘탈 스크리닝 (v4.2 추가)

BTC를 메인 거래 엔진으로 유지하면서, **알트코인 전체를 주식 재무제표처럼 스크리닝**하는 별도 모듈을 병행한다. 주식 PER/PBR에 대응하는 지표로 종목을 정량 평가하고, 상위 종목은 symbol_ranker에 입력된다.

| 주식 지표 | 코인 대응 지표 | 데이터 소스 |
|---|---|---|
| PER (시총/순이익) | P/F ratio (시총 ÷ 연간 수수료 수익) | Token Terminal |
| PBR (시총/자산) | P/S ratio (시총 ÷ 거래량/TVL) | Token Terminal, DeFiLlama |
| 매출/영업이익 | Protocol Revenue (실제 수수료 수익) | Token Terminal |
| 자산 규모 | TVL (Total Value Locked) | DeFiLlama |
| 사용자 수 | DAU, 활성 지갑 수 | Glassnode, Nansen |
| 주식 희석 | 토큰 인플레이션율 | Messari, CoinGecko |
| 재무 건전성 | Treasury 자산 규모 | Messari |
| 개발 활성도 | GitHub Commits / 주 | Token Terminal, Santiment |

**핵심 원칙**:
- 수익(Revenue)이 좋아도 **토큰 인플레이션이 높으면 감점** — 희석 조정 수익률로 평가
- TVL 성장률이 P/S보다 빠르면 저평가 시그널 (DeFi 종목 한정)
- BTC는 온체인 피처로 ML 입력에 직접 반영, 알트코인은 종목 선택 필터로 활용

### 2.4 참고 오픈소스 / 문헌

**v3 유지**: TradingAgents, AgenticTrading, FinMem, EliasAbouKhater/trading-bot

**v4 추가**:
- López de Prado, *Advances in Financial Machine Learning* (2018) — Triple Barrier · Meta-Labeling · CPCV · Deflated Sharpe 원전
- AlphaPy / mlfinlab — 위 기법 오픈소스 구현 레퍼런스
- LightGBM walk-forward best practices (Kaggle Numerai 우승팀 글)

**v4.2 추가 (온체인/펀더멘탈)**:
- Glassnode — BTC/ETH 온체인 심층 지표 (MVRV, SOPR, Realized Cap, Exchange Netflow)
- Token Terminal — 프로토콜 수익·P/F·P/S 등 재무제표형 코인 데이터
- DeFiLlama — TVL, 체인별·프로토콜별 수익 특화
- Nansen — 지갑 행동·스마트 머니 분석
- Messari — 토큰 인플레이션, Treasury, 프로젝트 리서치

---

## 3. 003 Autobot 자산 매핑 (v4 업데이트)

| 트렌드 패턴 | 기존 모듈 | 신규(v1 구현) | 상태 |
|---|---|---|---|
| Trade Journal | `database/`, `execution/` | — | 부분 존재, 스키마 부족 |
| Reflection Loop | `strategy/feedback_loop.py` | — | 스텁, 되먹임 미연결 |
| Regime 분류 | `strategy/regime_detector.py` | feature `adx_bucket` | 존재 |
| 룰베이스 신호 | `strategy/signal_aggregator.py`, `trend_filter.py`, `momentum_trigger.py`, `micro_confirmation.py` | — | 기존 (백업/fallback) |
| **ML 신호** | `strategy/ml_filter.py` (분류) | **`strategy/ml_features.py`, `strategy/ml_model.py`** (회귀, 메인) | v1 신규, **메인 엔진 후보** |
| 백테스트 | `backtest/backtester.py` | **`backtest/ml_backtester.py`** | 룰 + ML 둘 다 |
| 리포트 | `backtest/performance.py` | **`backtest/ml_report.py`** | 둘 다 |
| 데이터 인프라 | `data/okx_rest_client.py` (live) | **`data/okx_history_loader.py`** (deep history) | 분리됨 |
| 실행 | `execution/order_manager.py`, `trade_lifecycle.py` | — | 존재, ML hook 필요 |
| 리스크 | `risk/risk_manager.py`, `position_sizer.py` | — | 존재, ML kill switch 추가 필요 |
| 알림 | `notification/telegram_notifier.py`, `notification/discord_notifier.py` | — | 존재 |
| 데이터 품질 | `data/okx_history_loader.py`, `scripts/fetch_btc_history.py` | `scripts/validate_btc_history.py` (신규) | backfill/gap-fill 필요 |
| 비용 모델 | `backtest/ml_backtester.py` | `data/okx_funding_loader.py` (신규) | fee/slippage/funding 현실화 필요 |
| **BTC 온체인 피처** | — | `data/onchain_loader.py` (신규) | MVRV, NVT, Realized Cap, Netflow, SOPR, Active Addr → ML feature 입력 |
| **멀티코인 펀더멘탈** | `strategy/symbol_ranker.py` | `data/fundamental_loader.py` (신규) | P/F, P/S, TVL, Revenue, 인플레이션율 → 종목 선택 스코어 |

→ Phase -1~1은 **데이터 무결성 + 신규 ML 모듈 추가 + 학습 루프 연결**이 70%, 인프라 재활용이 30%.

---

## 4. 6단계 자율화 로드맵

v3는 4단계였다. v4는 **Phase 0 (ML edge 강화)**를 앞에 추가했다. v4.1은 그 앞에 **Phase -1 (데이터 무결성)**을 추가한다. 데이터가 비어 있거나 누락되면 어떤 ML 개선도 의미가 없기 때문이다.

### 검증 분할 원칙 (고정)

| 구간 | 용도 | 규칙 |
|---|---|---|
| 2020-01-01 ~ 2023-12-31 | Discovery / feature 연구 | 피처·라벨·모델 구조 탐색 가능 |
| 2024-01-01 ~ 2024-12-31 | Validation / 파라미터 압축 | 임계값·하이퍼파라미터 최종 선택 |
| 2025-01-01 ~ 2026-04-27 | Final OOS | 최종 성능 확인만. 여기서 맞추기 금지 |

월별 walk-forward는 별도로 유지한다. 학습은 과거 18~36개월, 테스트는 다음 1개월, embargo는 최소 `max(target_horizon, barrier_horizon)` 이상으로 둔다.

### Phase -1. 데이터 무결성 Foundation (2~4일)

**목표**: 2020-01-01 이후 BTC 1분봉 원천 데이터와 feature store를 신뢰 가능한 상태로 고정한다.

| # | 작업 | 신규/수정 파일 | 게이트 |
|---|---|---|---|
| -1.1 | 과거 backfill 모드 추가 (`resume` 우회) | `scripts/fetch_btc_history.py`, `data/okx_history_loader.py` | 2020-01부터 현재까지 월별 parquet 생성 |
| -1.2 | gap-fill 모드 추가 | `scripts/fetch_btc_history.py` | 누락 구간만 재요청 가능 |
| -1.3 | 데이터 검증 리포트 | `scripts/validate_btc_history.py` (신규) | 중복 0, 1분 gap 누락률 ≤ 0.1%, 월별 gap 표 출력 |
| -1.4 | feature 재빌드 | `strategy/ml_features.py` 실행 | 2020~현재 feature parquet 생성 |
| -1.5 | v1 full baseline 봉인 | `models/btc_lgb_v1_full/`, `backtest/results/btc_ml_v1_full/` | 이후 모든 개선은 이 baseline과 비교 |

**완료 기준**: raw/features/model/report가 2020+ 전체 범위에서 생성되고, `baseline_manifest.json`에 데이터 범위·row 수·gap 수·git sha·설정값이 저장된다.

### Phase 0. ML Edge 강화 (3~4주, 자율학습 가치의 90% 좌우)

전제: 학습 가능한 모델이 없으면 자율학습 루프는 텅 빈 상자다. 먼저 백테스트 메트릭이 라이브 진입 임계 (rank-IC ≥ 0.025, daily-Sharpe ≥ 1.5)를 통과해야 한다.

**목표 메트릭**: rank-IC 0.01 → 0.04+, daily Sharpe 0 → 1.5+, max DD ≤ 25%, 6+년 walk-forward.

**작업 항목** (영향력 순):

| # | 작업 | 신규/수정 파일 | 게이트 |
|---|---|---|---|
| 0.1 | Phase -1 baseline 기준선 고정 + 실험 추적 | `backtest/results/btc_ml_v1_full/baseline_manifest.json` | rank-IC, Sharpe, PF, DD, monthly hit rate 기록 |
| 0.2 | Triple Barrier Labeling | `strategy/ml_labels.py` (신규), `ml_features.py` 수정 | rank-IC ≥ baseline + 0.005 |
| 0.3 | Vol-Normalized Target | `ml_features.py` 수정 | rank-IC ≥ baseline + 0.010 |
| 0.4 | Multi-horizon 앙상블 | `strategy/ml_ensemble.py` (신규) | trade count -30~50%, Sharpe ≥ baseline + 0.3 |
| 0.5 | **Meta-Labeling** (단일 최대 효과) | `strategy/ml_meta_label.py` (신규) | trade count -40%, Sharpe ≥ baseline + 0.5 |
| 0.6 | Cost-aware EV thresholding | `backtest/ml_backtester.py` 수정 | 평균 trade EV > 0, win rate +5%p 또는 PF +0.2 |
| 0.7 | Funding / OI 피처 | `data/okx_funding_loader.py` (신규), `ml_features.py` 수정 | feature importance 상위 30위 진입 |
| 0.7b | **BTC 온체인 피처** | `data/onchain_loader.py` (신규), `ml_features.py` 수정 | MVRV·NVT·Realized Cap·Exchange Netflow·SOPR·Active Addr 6종 이상 feature importance 상위 20위 진입 |
| 0.8 | Optuna + SHAP feature selection | `strategy/ml_optuna.py` (신규) | rank-IC +0.005 |
| 0.9 | CPCV + Deflated Sharpe + Stress test | `strategy/ml_cpcv.py`, `backtest/stress_test.py` (신규) | DSR > 0, fee 2x 시 Sharpe > 0.7 |
| 0.10 | **멀티코인 펀더멘탈 스크리너** | `data/fundamental_loader.py` (신규), `strategy/symbol_ranker.py` 수정 | P/F·P/S·TVL·Revenue·인플레이션 조정 스코어 → symbol_ranker 통합, 주 1회 업데이트 |

**완료 기준**: 0.1~0.6 누적 통과 + Phase 0 종료 시 검증 메트릭이 라이브 후보 임계 도달.

**라이브 후보 임계**:
- Final OOS profit factor > 1.25
- Final OOS daily Sharpe > 1.0, 전체 walk-forward daily Sharpe > 1.5
- Max drawdown ≤ 20~25%
- 월별 수익 구간 ≥ 60%
- Monte Carlo 수익 확률 ≥ 70%, fee/slippage 2x stress에서도 손익분기 이상

### Phase 1. Trade Journal 표준화 (1~2일)

**목표**: ML 학습 + 룰베이스 reflection 양쪽의 "교재" 단일화.

- `database/`에 `trade_journal` 테이블 추가 (v3와 동일 + ML 필드)
  - 기본: `trade_id`, `opened_at`, `closed_at`, `symbol`, `side`, `pnl`, `pnl_pct`, `mfe`, `mae`, `entry_reason`, `exit_reason`
  - **ML 필드**: `model_version`, `fold_id`, `y_hat` (예측치), `y_realized` (실현치), `meta_label_score` (2차 모델 점수), `decision_path` (예: "ml_primary+meta_pass+cost_passed")
  - 기존 `regime`, `indicators_snapshot (JSON)` 유지
- `execution/order_manager.py` 진입·청산 훅에서 기록 호출
- 기존 거래 데이터도 가능한 범위 백필

**완료 기준**: 최근 거래 10건이 ML 메타데이터 포함 완전 구조로 DB 적재.

### Phase 2. Reflection Loop + ML 주간 재학습 (3~4일)

**목표**: 저널이 다음 의사결정과 다음 모델을 동시에 바꾸게 만들기.

#### 2.1 야간 배치 (rule-based, 매일)
- 최근 50~200건 저널 요약
  - regime × decision_path × win rate / avg PnL / Sharpe
  - 상습 손절 패턴 ("US session × ADX>25 × meta_pass: -65% win rate" 등)
- `strategy/feedback_loop.py`가 리포트를 읽어:
  - cost-aware threshold 동적 보정 (예: 직전 7일 win rate < 45%면 임계 +10%)
  - meta_label 모델의 confidence cutoff 조정
- 저널에 `entry_reason` 자연어 요약 자동 생성 (옵션 LLM, 캐시)

#### 2.2 주간 ML 재학습 (Sun 03:00 UTC)
**신규 파일**: `scripts/weekly_retrain.py`

- 직전 7일치 데이터 fetch → feature 재빌드 → 마지막 fold를 `init_model`로 warm-start retrain
- Validation: 직전 3일 hold-out에서 rank-IC > 0.015이면 새 모델 promotion
- 모델 버전: `models/btc_lgb_v4/fold_YYYY-MM_v{N}.lgb` + `current.lgb` symlink
- 매월 마지막 일요일은 full retrain (catastrophic forgetting 방지)

**완료 기준**:
- 주 1회 재학습 자동 실행
- 야간 리포트가 파일로 생성되고, 다음 거래 로그에 `feedback_adjusted=true` 표식 남음

### Phase 3. Thompson Sampling 밴딧 (3~5일)

**목표**: 사람이 모델 ON/OFF 하지 않아도 자동 적응.

#### v3와의 차이
- v3 밴딧: regime × **rule-based 전략** (trend/momentum/micro)
- v4 밴딧: regime × **모델 variant** + 룰베이스 fallback

#### 밴딧 arms (선택지)
1. ML primary (single 15m horizon)
2. ML multi-horizon ensemble
3. ML + meta-label (보수)
4. 룰베이스 triple confirmation (fallback)
5. **Flat** (거래 안 함) — 모든 arm의 기대 reward 음수일 때 선택

#### 메커니즘
- regime별로 각 arm의 `alpha` (성공), `beta` (실패) 베타분포 유지
- 매 시그널 평가 시 현 regime에서 샘플링된 점수로 arm 선택
- 거래 종료 후 자동 업데이트 (PnL > expected_cost → alpha++; 그 외 beta++)
- **탐색 보장**: 각 arm × regime 조합이 최소 30 거래 누적 전엔 균등 가중

**신규 파일**: `strategy/bandit.py`

**대시보드**: regime × arm 현재 가중 + alpha/beta 시각화

**완료 기준**: 동일 조건 3일 연속 손실 arm이 가중 0.1 미만으로 자동 수렴.

### Phase 4. Sandbox 승격 + Live 운영 (2주+)

**목표**: 자가학습이 실계좌를 망가뜨리지 않도록 안전 게이트.

#### 4.1 승격 단계 (모든 변경 — 모델 / 룰베이스 파라미터 / 밴딧 — 공통 적용)

```
[Backtest Gate] → [Paper Trading Gate] → [Small-cap Live Gate] → [Full Live]
```

| 단계 | 기간 | 통과 기준 |
|---|---|---|
| Backtest | 2020+ walk-forward + Final OOS | rank-IC ≥ 0.025, Sharpe(daily) ≥ 1.5, PF > 1.25, max DD ≤ 25%, fee 2x stress Sharpe > 0.7 |
| Paper Trading (OKX demo) | **2주 필수** | 진입 빈도 ±20%, 평균 ŷ 분포 ±15%, 일별 PnL이 백테스트 95% CI 안 |
| Small-cap Live | 14~30일 | 자본 5%만 운용. Paper 대비 Sharpe ≥ 80%, 일 손실 한도 2% 미발동 |
| Full Live | — | 사용자 명시 승인 후 |

#### 4.2 Live Inference Loop
**신규 파일**: `core/ml_trader.py`
- 1분 주기 (asyncio loop):
  1. 최신 1m 봉 1개 fetch
  2. 240+15분 캐시 + 신규 봉 append
  3. `build_features()` 마지막 row만 추출
  4. 활성 fold 모델로 inference
  5. (Phase 3 활성 시) 밴딧 arm 선택 → 의사결정
- 기존 `core/bot_engine.py`와 별도 모드 (`BOT_MODE=ML`)

#### 4.3 Position Reconciliation (필수)
**신규 파일**: `execution/reconciliation.py`
- 매 분 봇 내부 포지션 vs `OKXRestClient.fetch_positions` 비교
- 사이즈 차이 < 1% → 거래소 기준 동기화
- 사이즈 차이 ≥ 1% → 안전 모드 + 알림 (외부 수동 개입 / 부분 청산 / 네트워크 실패 대응)

**완료 기준**: 신규 ML 모델 1종을 sandbox 4단계 통과시켜 small-cap live까지 도달.

---

## 5. 가드레일 (필수, Phase 1과 병행 도입)

### 5.1 손실 통제 (v3 유지 + ML 분포 drift 추가)
- **일일 손실 한도**: 자본 -2% → 당일 신규 진입 금지
- **연속 손절 쿨다운**: 3회 → 1시간 정지
- **MDD 브레이커**: 주간 MDD -8% → 전 모드 페이퍼 강등
- **ML 분포 drift (신규)**: 라이브 ŷ 분포가 학습 분포 KL-div > 임계 → 안전 모드
- **Live sign accuracy 모니터링 (신규)**: 직전 200 trades sign accuracy < 47% → 안전 모드
- **데이터 stale 차단 (신규)**: 최신 확정 1m 봉이 2분 이상 지연되면 신규 진입 금지
- **미확정 candle 제외 (신규)**: 현재 진행 중인 봉은 feature/inference 입력에서 제외

### 5.2 모델/버전 추적 (v3 유지 + ML 메타 추가)
- 모든 거래에 `model_version`, `bandit_arm`, `bandit_snapshot` 기록
- ML 모델 버전: `fold_YYYY-MM_v{N}.lgb`, metadata.json 불변 보관
- 가중치 히스토리 테이블로 롤백 가능
- 데이터 버전: `data_manifest.json`에 raw 범위, feature 범위, gap count, funding/OI 범위 기록

### 5.3 Kill Switch (이중화 강화)
- 텔레그램 명령:
  - 전체 포지션 시장가 청산
  - 밴딧 학습 동결
  - 신규 진입 중단
- **이중화**: 텔레그램 명령 + 로컬 파일 플래그 (`/tmp/autobot_kill`) 둘 다 인식
- 운영자 수동 승인 전엔 재가동 불가

### 5.4 자동 정지 트리거 (강화)
| 트리거 | 조치 |
|---|---|
| 일별 PnL < -3% | 당일 진입 금지 |
| 7일 누적 PnL < -8% | 페이퍼 강등 |
| 연속 손실 5회 | 1시간 쿨다운 |
| Live sign accuracy < 47% (200건) | 안전 모드 |
| ŷ 분포 KL-div > 임계 | 안전 모드 |
| 1m 데이터 지연 > 2분 | 신규 진입 금지 |
| 미확정 candle 감지 | 해당 loop skip |
| 거래소 API 에러율 > 5% (1h) | 안전 모드 |
| Position reconciliation 실패 | 안전 모드 + 즉시 알림 |

---

## 6. 관측성·운영

### 6.1 대시보드 추가 패널 (기존 [web/](web/) Next.js 앱)
- **현재 regime** (실시간)
- **regime × bandit arm 가중치** 히트맵
- **최근 100건 거래** entry_reason / exit_reason 카드뷰 (ML 모드면 ŷ + meta_score 노출)
- **ML 메트릭 대시보드**: 라이브 sign accuracy 7일 이동평균, ŷ 분포 히스토그램, drift 알람
- **Sandbox 승격 상태판**: 각 모델 버전이 어느 단계에 있는지 (backtest → paper → small-cap → full)
- **가드레일 트리거 로그**

### 6.2 알림 (기존 [notification/](notification/) 재사용)
- 진입/청산 (ML 모드: ŷ + meta_score + bandit_arm 포함)
- 일별 PnL 요약 (UTC 00:00)
- 주간 ML 재학습 결과 (rank-IC 변화, promote 여부)
- 가드레일 발동
- Reconciliation 실패
- 텔레그램 일일 리포트에 "오늘 학습으로 바뀐 가중치 Top 3" + "이번 주 ML 모델 업데이트"

---

## 7. 일정 (총 9~11주)

| 주차 | 작업 | 산출물 |
|---|---|---|
| W1 | Phase -1 데이터 backfill/gap-fill + full baseline | 2020+ Parquet, feature store, baseline manifest |
| W2 | Phase 0.2~0.4 (Triple Barrier, Vol-norm, Multi-horizon) | rank-IC ≥ 0.02 |
| W3 | Phase 0.5~0.7 (Meta-label, Cost-aware EV, Funding/OI) | PF > 1.0, Sharpe ≥ 1.0 |
| W4 | Phase 0.8~0.9 (Optuna, CPCV, Stress) | DSR > 0, fee 2x stress 통과 |
| W5 | Phase 1 Trade Journal + 가드레일 | Journal 운영, kill switch 초안 |
| W6 | Phase 2 Reflection Loop + 주간 재학습 | 야간 리포트, retrain 자동화 |
| W7 | Phase 3 Thompson Sampling 밴딧 | 밴딧 paper 모드 |
| W8 | Phase 4 Live Inference + Reconciliation | `ml_trader.py` paper 운영 시작 |
| W9 | Paper Trading 검증 1주차 | 분포 일치 리포트 |
| W10 | Paper Trading 검증 2주차 + Small-cap 준비 | promote/no-promote 결정 |
| W11 | Small-cap 검증 + 운영 문서 + 모니터링 보강 | Full live 후보 |

---

## 8. 리스크 및 완화

| 리스크 | 완화책 |
|---|---|
| `fetch_btc_history.py` resume 구조로 2020~2026 과거 구간 누락 | Phase -1에서 backfill/ignore-resume/gap-fill 모드 추가 후 수집 |
| 데이터 gap이 모델 edge처럼 보임 | `validate_btc_history.py`를 학습 전 필수 게이트로 실행 |
| Phase 0에서 rank-IC < 0.025 못 넘김 | Phase 4 진입 보류. 룰베이스 fallback만 운영. cross-asset feature(ETH/BTC ratio) 옵션 도입 검토 (v5) |
| Triple Barrier embargo 누락 → 누설 | embargo를 정적 60분 → barrier max horizon으로 동적화. `tests/test_ml_features.py`에 triple barrier 누설 테스트 추가 강제 |
| 밴딧이 한 arm만 선택 (탐험 부족) | 각 arm × regime 최소 30 거래까지 균등 가중, 그 후에도 epsilon=0.05 탐험 유지 |
| Online learning catastrophic forgetting | weekly warm-start + monthly full retrain 병행 |
| Live ↔ Backtest 분포 mismatch | Paper 2주는 협상 불가. mismatch 발견 시 즉시 promote 보류 |
| Funding rate 부호 오류 | constant 0.01% 가정은 v1 한정. Phase 0.7에서 실제 history로 교체. unit test로 long/short 부호 검증 |
| Kill Switch 오발 / 장애 시 미작동 | 이중화 (Telegram + local file flag) |
| 모델 버전 꼬임 | 불변 metadata.json + symlink 패턴, Alembic 마이그레이션 강제 |
| Position reconciliation 오탐 | 1% 미만 차이는 자동 동기화, 그 이상만 알림 + 안전 모드 |
| 학습 비용 폭증 (Optuna) | Optuna는 1회만 (Phase 0.8). 이후엔 고정 하이퍼파라미터로 weekly retrain만 |

---

## 9. 즉시 시작 체크리스트

**Phase -1 시작 전 (지금 가능)**:
- [x] v1 ML 파이프라인 구현 (data/feature/train/backtest/report)
- [ ] `scripts/fetch_btc_history.py`에 `--mode backfill` 또는 `--ignore-resume` 추가
- [ ] `scripts/validate_btc_history.py` 생성 (월별 row/gap/duplicate 리포트)
- [ ] `python3 scripts/fetch_btc_history.py --start 2020-01-01 --mode backfill` 실행
- [ ] 풀 데이터 feature 재빌드 + v1 baseline 학습/백테스트 → `baseline_manifest.json` 기록

**Phase 0 진입 후**:
- [ ] `strategy/ml_labels.py` Triple Barrier 구현 + 단위 테스트
- [ ] `strategy/ml_meta_label.py` 메타라벨 모델
- [ ] `data/okx_funding_loader.py` funding/OI history 수집
- [ ] `backtest/ml_backtester.py`에 EV/cost-aware threshold 반영
- [ ] `data/onchain_loader.py` BTC 온체인 피처 수집 (Glassnode API 또는 무료 대체재: CryptoQuant, Checkonchain)
  - MVRV Ratio, NVT Ratio, Realized Cap, Exchange Netflow, SOPR, Active Addresses
  - 일별 granularity → 1m bar에 forward-fill로 병합
- [ ] `data/fundamental_loader.py` 멀티코인 펀더멘탈 수집 (Token Terminal API, DeFiLlama API)
  - P/F ratio, P/S ratio, Protocol Revenue, TVL, 토큰 인플레이션율
  - 주 1회 갱신, `strategy/symbol_ranker.py` 점수에 20~30% 가중
- [ ] `strategy/symbol_ranker.py`에 펀더멘탈 스코어 레이어 추가
  - 기존: 7d return(40%) + trend/ADX(30%) + volume(20%) + volatility(10%)
  - 개선: 7d return(30%) + trend/ADX(25%) + volume(15%) + volatility(10%) + **펀더멘탈(20%)**

**Phase 1 진입 게이트** (Phase 0 통과 시):
- [ ] `database/`에 `trade_journal` 테이블 + ML 필드 스키마
- [ ] `execution/order_manager.py` 진입/청산 훅
- [ ] 일일 손실 한도 + 연속 손절 쿨다운
- [ ] 텔레그램 Kill Switch 명령 + 로컬 플래그 이중화

---

## 10. 참고 자료

### v3 유지
- Agentic Trading — https://wundertrading.com/journal/en/learn/article/agentic-trading
- TradingAgents — https://github.com/TauricResearch/TradingAgents
- AgenticTrading — https://github.com/Open-Finance-Lab/AgenticTrading
- FinMem — https://github.com/pipiku915/FinMem-LLM-StockTrading
- TradeReflector + Thompson Sampling — https://dev.to/ai-agent-economy/our-trading-bot-rewrites-its-own-rules-heres-how-and-what-went-wrong-5dg9
- Deep Q-Network Bitcoin Trading — https://www.tandfonline.com/doi/full/10.1080/23322039.2025.2594873

### v4 추가
- López de Prado, *Advances in Financial Machine Learning* (Wiley, 2018) — Chapters 3 (Triple Barrier), 4 (Sample Weights), 5 (Meta-Labeling), 7 (CPCV), 14 (Backtest Statistics)
- mlfinlab (López de Prado 기법 오픈소스 구현) — https://github.com/hudson-and-thames/mlfinlab
- LightGBM 시계열 walk-forward 패턴 — Numerai forum, Kaggle Optiver 우승자 글
- OKX API funding-rate-history docs — https://www.okx.com/docs-v5/en/#public-data-rest-api-get-funding-rate-history

### v4.2 추가 (온체인/펀더멘탈)

**BTC 온체인 (ML 피처용)**:
- Glassnode API — MVRV, SOPR, Realized Cap, Exchange Netflow, Active Addresses (유료, 무료 tier 일부 제공) — https://glassnode.com
- CryptoQuant — Exchange Inflow/Outflow, Miner Flow, Funding Rate (무료 tier 있음) — https://cryptoquant.com
- Checkonchain — BTC 온체인 지표 시각화 + 무료 CSV — https://checkonchain.com

**멀티코인 펀더멘탈 스크리닝**:
- Token Terminal API — P/F, P/S, Revenue, Fees 등 재무제표형 데이터 (무료 tier 있음) — https://tokenterminal.com/api
- DeFiLlama API — TVL, 체인별 프로토콜 수익 (완전 무료) — https://defillama.com/docs/api
- Messari API — 토큰 인플레이션, Treasury, 리서치 (무료 tier 있음) — https://messari.io/api
- CoinGecko API — 시총, 발행량, 가격 (무료 tier 있음) — https://www.coingecko.com/api/documentation
