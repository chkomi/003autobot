# 003 Autobot Codex Plan (v3) — 자율학습 거래 로드맵

> 변경 요약 (v2 → v3)
> - 주제 전환: 서비스 아키텍처 중심(v2) → "자율거래 학습 루프" 중심(v3)
> - 2026년 자율거래 봇 트렌드 조사 결과 반영
> - 4단계 자율화 로드맵 추가 (Trade Journal → Reflection → Bandit → Sandbox)
> - 가드레일(일일 손실 한도, 모델 버전 태깅, Kill switch) 명시
> - 기존 모듈(`regime_detector`, `feedback_loop`, `ml_filter`, `score_engine`, `signal_aggregator`, `backtest`) 재활용 매핑
> 작성일: 2026-04-24

---

## 1. 배경 및 문제 정의

현재 003 Autobot은 신호 생성 · 실행 · 백테스트 · 대시보드까지 골격이 갖춰져 있으나, 운영자 체감상 "봇이 스스로 학습해서 거래 방식을 바꾸는" 느낌이 약하다. 원인은 기능 부재가 아니라 **모듈 간 학습 루프 연결이 느슨**하다는 것이다.

- 저널(거래 로그)에 지표/장세/손익이 구조화되어 저장되지 않음
- `feedback_loop.py`가 존재하지만 다음 의사결정의 가중치 조정으로 되먹임되지 않음
- `regime_detector` 결과가 전략 가중에 자동 반영되지 않음
- 신규 전략을 실거래에 얹기 전 자동 검증·승격 파이프라인 없음

---

## 2. 2026년 자율거래 봇 트렌드 (조사 결과)

### 2.1 공통 패턴 7가지

1. **Trade Journal + Reflection Loop**
   - 매 거래를 regime / strategy_id / 지표 스냅샷 / 진입근거 / 청산근거 / PnL / MFE·MAE로 구조화
   - 종료 후 "TradeReflector"가 분석 → 다음 판단의 컨텍스트로 재주입

2. **Regime Detector → 전략 스위칭**
   - 추세/횡보/고변동 자동 분류 → 장세별 전략 가중 자동 조정
   - 횡보장에서 모멘텀 가짜신호를 막는 가장 큰 효과

3. **Thompson Sampling 강화학습 밴딧**
   - regime × strategy 조합별 베타분포 유지
   - 잘 맞히는 조합에 자금 더 배분 (수동 ON/OFF 불요)

4. **LLM 멀티에이전트 협업** (TradingAgents, AgenticTrading, FinMem)
   - 리서처 / 트레이더 / 리스크매니저 / 반성자 역할 분담
   - 계층형 메모리로 성공·실패 사례 축적

5. **샌드박스 승격 파이프라인**
   - 신규 전략 = 백테스트 + 페이퍼 일정 기간 → 샤프·MDD 기준 통과 시 소액 → 정상 배분

6. **설명가능성(XAI)**
   - 왜 진입/청산했는지 자연어 근거를 저널에 함께 저장 → 운영자가 신뢰·개입 가능

7. **가드레일 우선**
   - 일일 손실 한도, 연속 손절 쿨다운, 모델 버전 태깅, Kill switch 표준 탑재

### 2.2 참고 오픈소스

- `TauricResearch/TradingAgents` — 멀티에이전트 LLM 프레임워크
- `Open-Finance-Lab/AgenticTrading` — memory-augmented 백테스트 + 지속학습
- `pipiku915/FinMem-LLM-StockTrading` — 계층형 메모리 LLM 에이전트
- `EliasAbouKhater/trading-bot` — regime-aware 포트폴리오 리밸런싱 구현 레퍼런스

---

## 3. 003 Autobot 자산 매핑

| 트렌드 패턴             | 기존 모듈                                   | 상태        |
|-------------------------|---------------------------------------------|-------------|
| Trade Journal           | `database/` (SQLite), `execution/`          | 부분 존재, 스키마 부족 |
| Reflection Loop         | `strategy/feedback_loop.py`                 | 스텁 수준, 되먹임 미연결 |
| Regime 분류             | `strategy/regime_detector.py`               | 존재        |
| 전략 가중 (밴딧)        | `strategy/signal_aggregator.py`, `score_engine.py` | 수동 가중치 |
| 신호 필터               | `strategy/ml_filter.py`, `trend_filter.py`  | 존재        |
| 백테스트                | `backtest/`, `run_full_backtest.py`         | 존재        |
| 실행                    | `execution/`                                | 존재        |
| 리스크                  | `risk/`                                     | 존재        |

→ 신규 개발보다 **연결·확장** 비중이 70% 이상.

---

## 4. 자율화 4단계 로드맵

### Phase 1. Trade Journal 표준화 (1~2일)

**목표**: 학습의 "교재" 만들기.

- `database/`에 `trade_journal` 테이블 추가
- 컬럼: `trade_id`, `opened_at`, `closed_at`, `symbol`, `side`, `strategy_id`, `regime`, `indicators_snapshot (JSON)`, `entry_reason`, `exit_reason`, `pnl`, `pnl_pct`, `mfe`, `mae`, `model_version`
- `execution/` 진입·청산 훅에서 기록 호출
- 기존 거래 데이터도 가능한 범위에서 백필

**완료 기준**: 최근 거래 10건이 완전한 구조로 DB에 들어감.

### Phase 2. Reflection Loop 연결 (2~3일)

**목표**: 저널을 읽어 다음 판단을 바꾸게 만들기.

- 야간 배치 Job: 최근 50~200건 저널 요약 리포트 생성
  - regime × strategy × 승률 / 평균 PnL / 샤프
  - 상습 손절 패턴 (예: "횡보장 모멘텀 롱은 -65% 승률")
- `strategy/feedback_loop.py`가 리포트를 읽어 `score_engine` 기본 가중치 업데이트
- (옵션) LLM 요약을 프롬프트 컨텍스트에 주입, 결과 캐시로 토큰 절약
- 저널에 `entry_reason` 자연어 근거 기록 (XAI 기반)

**완료 기준**: 리포트가 파일로 생성되고, 다음 거래의 가중치 로그에 "feedback_loop 적용됨" 표식이 남음.

### Phase 3. Thompson Sampling 밴딧 (3~5일)

**목표**: 사람이 전략 ON/OFF 하지 않아도 시장 변화에 자동 적응.

- `strategy/signal_aggregator.py`에 밴딧 레이어 추가
- regime별로 각 전략의 `alpha` (성공), `beta` (실패) 파라미터 유지 → 베타분포 샘플링
- 매 신호 생성 시 현재 regime에서 샘플링된 점수로 전략 가중 결정
- 거래 종료 후 밴딧 파라미터 자동 업데이트 (PnL > 0 → alpha++, 아니면 beta++)
- 대시보드에 regime × strategy 현재 가중치 시각화

**완료 기준**: 동일 조건에서 3일 연속 손실인 전략이 자동으로 가중 0.1 미만으로 수렴.

### Phase 4. 샌드박스 승격 파이프라인 (1주)

**목표**: 자가학습이 실계좌를 망가뜨리지 않도록 안전하게 새 전략을 도입.

- 신규/수정 전략은 자동으로 다음 단계 통과 요구
  1. `backtest/` 과거 2년 데이터, 샤프 > 1.0, MDD < 15%
  2. 페이퍼 모드 14일, 실시간 성과가 백테스트의 80% 이상
  3. 소액 실거래 (전체 자본의 5%) 14일
  4. 기준 통과 시 정상 배분 편입
- 단계별 성과 저조 시 자동 중단 + 알림
- 모든 전략 버전에 `model_version` 태그 부여

**완료 기준**: 신규 전략 1개를 이 파이프라인으로 통과시켜 소액 실거래까지 도달.

---

## 5. 가드레일 (필수 · Phase 1과 병행 도입)

### 5.1 손실 통제
- **일일 손실 한도**: 자본의 -2% 도달 시 당일 신규 진입 금지
- **연속 손절 쿨다운**: 3회 연속 손절 시 1시간 거래 정지
- **MDD 브레이커**: 주간 MDD -8% 초과 시 전 전략 페이퍼 모드로 강등

### 5.2 모델/버전 추적
- 모든 의사결정에 `model_version`, `strategy_id`, `bandit_snapshot` 기록
- 롤백 가능하도록 가중치 히스토리 테이블 유지

### 5.3 Kill Switch
- 텔레그램 명령 한 줄로:
  - 전체 포지션 시장가 청산
  - 밴딧 학습 동결
  - 신규 진입 중단
- 운영자 수동 승인 전까지 재가동 불가

---

## 6. 관측성·운영

- 대시보드 추가 패널
  - 현재 regime (실시간)
  - regime × strategy 밴딧 가중치 히트맵
  - 최근 20건 거래의 entry_reason / exit_reason 카드뷰
  - 가드레일 트리거 로그
- 텔레그램 일일 리포트에 "오늘 학습으로 바뀐 가중치 Top 3" 추가

---

## 7. 일정 (총 2~3주)

| 주차   | 작업                                       | 산출물                              |
|--------|--------------------------------------------|-------------------------------------|
| W1 전반 | Phase 1 Trade Journal + 가드레일 v1        | `trade_journal` 테이블, 한도/쿨다운 |
| W1 후반 | Phase 2 Reflection Loop                    | 야간 리포트, feedback 되먹임        |
| W2 전반 | Phase 3 Thompson Sampling 밴딧             | `signal_aggregator` 밴딧 버전       |
| W2 후반 | Phase 4 샌드박스 파이프라인                | 승격 자동화, 대시보드 패널          |
| W3 전반 | 관측성·텔레그램 리포트·Kill Switch 보강    | 운영 완료                           |
| W3 후반 | 버퍼 (페이퍼 모드 실전 관찰, 튜닝)         | 운영 문서                           |

---

## 8. 리스크 및 완화

| 리스크                              | 완화책                                                 |
|-------------------------------------|--------------------------------------------------------|
| 밴딧이 과적합해 한 전략만 선택      | regime별 최소 샘플수 도달 전엔 균등 가중, 탐험 계수 유지 |
| 저널 데이터 부족으로 학습 불안정    | 초기 2주는 페이퍼 모드 병행, 최소 샘플 확보            |
| LLM 요약 비용 증가                  | 주 1회 배치만 LLM 사용, 일 배치는 규칙 기반 통계       |
| Kill Switch 오발 / 장애 시 미작동   | 이중화 (로컬 파일 플래그 + 텔레그램 명령)              |
| 모델 버전 꼬임                      | 불변 이벤트 로그, DB 마이그레이션은 Alembic 강제       |

---

## 9. 즉시 시작 체크리스트

- [ ] `database/` 에 `trade_journal` 테이블 스키마 정의
- [ ] `execution/` 진입/청산 지점에 저널 기록 훅 삽입
- [ ] 일일 손실 한도 + 연속 손절 쿨다운 도입
- [ ] 텔레그램 Kill Switch 명령 등록
- [ ] Phase 2 야간 배치 크론 작성

---

## 10. 참고 자료

- Agentic Trading — https://wundertrading.com/journal/en/learn/article/agentic-trading
- TradingAgents — https://github.com/TauricResearch/TradingAgents
- AgenticTrading — https://github.com/Open-Finance-Lab/AgenticTrading
- FinMem — https://github.com/pipiku915/FinMem-LLM-StockTrading
- TradeReflector + Thompson Sampling — https://dev.to/ai-agent-economy/our-trading-bot-rewrites-its-own-rules-heres-how-and-what-went-wrong-5dg9
- Deep Q-Network Bitcoin Trading — https://www.tandfonline.com/doi/full/10.1080/23322039.2025.2594873
