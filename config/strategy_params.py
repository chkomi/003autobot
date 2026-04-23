"""
strategy_params.yaml을 로딩하는 dataclass 정의.
전략 모듈은 이 파일의 dataclass를 사용해 파라미터를 주입받는다.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _load_yaml() -> dict:
    yaml_path = Path(__file__).parent / "strategy_params.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class TrendFilterParams:
    ema_fast: int = 21
    ema_mid: int = 55
    ema_slow: int = 200
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    timeframe: str = "4h"

    @classmethod
    def from_yaml(cls) -> "TrendFilterParams":
        cfg = _load_yaml().get("trend_filter", {})
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class MomentumParams:
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi_period: int = 14
    rsi_long_min: float = 40
    rsi_long_max: float = 65
    rsi_short_min: float = 35
    rsi_short_max: float = 60
    timeframe: str = "1h"

    @classmethod
    def from_yaml(cls) -> "MomentumParams":
        cfg = _load_yaml().get("momentum", {})
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class MicroParams:
    bb_period: int = 20
    bb_std: float = 2.0
    volume_multiplier: float = 1.5
    timeframe: str = "15m"

    @classmethod
    def from_yaml(cls) -> "MicroParams":
        cfg = _load_yaml().get("micro", {})
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class RiskParams:
    atr_period: int = 14
    atr_sl_multiplier: float = 1.0
    atr_tp1_multiplier: float = 2.0
    atr_tp2_multiplier: float = 3.5
    tp1_exit_pct: float = 0.5
    tp2_exit_pct: float = 0.3
    partial_exit_pct: float = 0.5  # legacy
    trailing_atr_multiplier: float = 2.0
    kelly_fraction: float = 0.25
    max_position_pct: float = 0.10
    # 본절 이동 시 수수료+슬리피지 버퍼 (진입가의 %). LONG은 +, SHORT은 -로 적용됨.
    breakeven_buffer_pct: float = 0.0006
    # 트레일링 스탑 최소 변경폭: |new - old| >= ATR × ratio 여야 재배치
    min_trail_step_ratio: float = 0.15
    # 타임 리밋 도달 시 이익률이 이 값 이상이면 청산 대신 트레일링 유지
    time_exit_profit_lock_pct: float = 0.003
    # 적응형 SL: TP1 이전까지 ATR 변화에 따라 SL 거리 동적 조정
    adaptive_sl_enabled: bool = True

    @classmethod
    def from_yaml(cls) -> "RiskParams":
        cfg = _load_yaml().get("risk", {})
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class FiltersParams:
    funding_rate_abs_threshold: float = 0.0005
    funding_filter_enabled: bool = True
    event_blackout_minutes: int = 30
    event_blackout_enabled: bool = True
    event_times_utc: list = None  # list[str] "YYYY-MM-DD HH:MM"

    def __post_init__(self):
        if self.event_times_utc is None:
            self.event_times_utc = []

    @classmethod
    def from_yaml(cls) -> "FiltersParams":
        cfg = _load_yaml().get("filters", {}) or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class MacroParams:
    """거시경제 매크로 파라미터"""
    fg_extreme_fear: int = 20       # Fear & Greed 극단 공포 기준
    fg_fear: int = 40               # 공포 기준
    fg_greed: int = 60              # 탐욕 기준
    fg_extreme_greed: int = 80      # 극단 탐욕 기준
    event_danger_hours: float = 2.0   # 고영향 이벤트 위험 구간 (시간)
    event_caution_hours: float = 6.0  # 고영향 이벤트 주의 구간 (시간)
    btc_dom_high: float = 55.0      # BTC 도미넌스 높음 기준
    btc_dom_low: float = 42.0       # BTC 도미넌스 낮음 기준
    macro_enabled: bool = True      # 매크로 스코어링 활성화

    @classmethod
    def from_yaml(cls) -> "MacroParams":
        cfg = _load_yaml().get("macro", {}) or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class ScoringParams:
    # 가중치 (합계 = 1.0)
    weight_trend: float = 0.25
    weight_momentum: float = 0.22
    weight_volume: float = 0.18
    weight_volatility: float = 0.12
    weight_sentiment: float = 0.10
    weight_macro: float = 0.13       # 거시경제 매크로 가중치

    # 진입 기준 점수
    strong_signal_threshold: float = 75.0
    weak_signal_threshold: float = 60.0
    exit_threshold: float = 40.0

    # ADX
    adx_strong: float = 25.0
    adx_very_strong: float = 40.0

    # 스토캐스틱
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0

    # OBV
    obv_lookback: int = 5

    # 볼린저 밴드 폭
    bb_bandwidth_low: float = 0.02
    bb_bandwidth_high: float = 0.08

    # 라이브 시그널 결합 모드: "AND" | "OR" | "SCORE_ONLY"
    signal_combine_mode: str = "AND"

    @classmethod
    def from_yaml(cls) -> "ScoringParams":
        cfg = _load_yaml().get("scoring", {}) or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class SentimentParams:
    oi_change_lookback_hours: int = 4
    oi_surge_threshold: float = 0.05
    long_short_extreme_long: float = 2.0
    long_short_extreme_short: float = 0.5
    funding_neutral_range: float = 0.0001

    @classmethod
    def from_yaml(cls) -> "SentimentParams":
        cfg = _load_yaml().get("sentiment", {}) or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class BacktestParams:
    """백테스팅 전용 파라미터 (라이브와 분리)"""
    signal_mode: str = "OR"              # "AND" / "OR" / "SCORE_ONLY"
    min_score_threshold: float = 50.0
    allow_partial_ema: bool = True
    rsi_long_min: float = 35.0
    rsi_long_max: float = 70.0
    rsi_short_min: float = 30.0
    rsi_short_max: float = 65.0
    macd_cross_lookback: int = 5
    volume_multiplier: float = 1.2
    bb_long_threshold: float = 0.40
    bb_short_threshold: float = 0.60

    @classmethod
    def from_yaml(cls) -> "BacktestParams":
        cfg = _load_yaml().get("backtest", {}) or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class SensitivityParams:
    """파라미터 민감도 분석 설정"""
    enabled: bool = True
    perturbation_pct: float = 0.15
    max_sharpe_drop_pct: float = 0.50

    @classmethod
    def from_yaml(cls) -> "SensitivityParams":
        cfg = _load_yaml().get("sensitivity", {}) or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class RegimeParams:
    """시장 레짐 감지 파라미터"""
    adx_trending_threshold: float = 25.0
    adx_ranging_threshold: float = 20.0
    bb_bandwidth_low: float = 0.03
    bb_bandwidth_high: float = 0.06
    atr_pct_volatile: float = 0.02
    block_ranging: bool = True

    @classmethod
    def from_yaml(cls) -> "RegimeParams":
        cfg = _load_yaml().get("regime", {}) or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class TradingHoursParams:
    """거래 시간 필터 파라미터"""
    enabled: bool = True
    start_hour_utc: int = 4
    end_hour_utc: int = 22

    @classmethod
    def from_yaml(cls) -> "TradingHoursParams":
        cfg = _load_yaml().get("trading_hours", {}) or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class WalkForwardParams:
    """워크포워드 검증 설정"""
    enabled: bool = True
    train_days: int = 45
    test_days: int = 15
    min_trades_per_window: int = 3
    step_days: int = 15

    @classmethod
    def from_yaml(cls) -> "WalkForwardParams":
        cfg = _load_yaml().get("walk_forward", {}) or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class NewsWatchParams:
    """뉴스 수집/위험도 평가 설정 (news_collector.py 사용)."""
    enabled: bool = True
    poll_interval_sec: int = 300           # 5분마다 뉴스 폴링
    lookback_minutes: int = 120            # 최근 2시간 내 헤드라인만 평가
    elevated_threshold: int = 20           # aggregate 점수 임계 — ELEVATED
    high_threshold: int = 40               # HIGH
    critical_threshold: int = 70           # CRITICAL
    # 액션
    reduce_on_high_ratio: float = 0.5      # HIGH 감지 시 모든 포지션을 이 비율만큼 축소
    close_all_on_critical: bool = True     # CRITICAL 감지 시 전량 청산
    pause_entries_on_high: bool = True     # HIGH 이상 시 신규 진입 일시 정지

    @classmethod
    def from_yaml(cls) -> "NewsWatchParams":
        cfg = _load_yaml().get("news_watch", {}) or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class VolSpikeParams:
    """변동성 급증 감지 설정 (volatility_spike.py 사용)."""
    enabled: bool = True
    timeframe: str = "15m"                 # 감지에 쓸 캔들 타임프레임
    recent_bars: int = 3
    baseline_bars: int = 72                # ≈ 18시간(15m 기준)
    vol_ratio_threshold: float = 3.0
    z_threshold: float = 3.5
    range_ratio_threshold: float = 3.0
    close_all_on_spike: bool = True        # 스파이크 감지 시 전량 청산
    pause_minutes_after_spike: int = 60    # 이후 N분간 신규 진입 차단

    @classmethod
    def from_yaml(cls) -> "VolSpikeParams":
        cfg = _load_yaml().get("vol_spike", {}) or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class EventActionParams:
    """예정된 고영향 경제 이벤트에 대한 선제 대응 설정."""
    enabled: bool = True
    pre_event_reduce_hours: float = 2.0    # 이벤트 2시간 전부터 축소 대상
    pre_event_reduce_ratio: float = 0.5    # 이 비율만큼 축소
    pre_event_close_hours: float = 0.5     # 이벤트 30분 전부터 전량 청산
    resume_minutes_after: int = 45         # 이벤트 종료 후 N분 뒤 재개 허용

    @classmethod
    def from_yaml(cls) -> "EventActionParams":
        cfg = _load_yaml().get("event_action", {}) or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class AutoResumeParams:
    """드로다운 정지 후 자동 복구 설정."""
    enabled: bool = True
    min_halt_minutes: int = 120            # 정지 후 최소 이 시간은 대기
    recovery_drawdown_pct: float = 0.08    # 드로다운이 이 값 이하로 복구되어야 재개
    max_auto_resumes_per_day: int = 1      # 하루 최대 자동 복구 횟수

    @classmethod
    def from_yaml(cls) -> "AutoResumeParams":
        cfg = _load_yaml().get("auto_resume", {}) or {}
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


@dataclass
class AllStrategyParams:
    trend: TrendFilterParams
    momentum: MomentumParams
    micro: MicroParams
    risk: RiskParams
    filters: FiltersParams
    scoring: ScoringParams
    sentiment: SentimentParams
    macro: MacroParams
    backtest: BacktestParams = None
    sensitivity: SensitivityParams = None
    walk_forward: WalkForwardParams = None
    regime: RegimeParams = None
    trading_hours: TradingHoursParams = None
    news_watch: NewsWatchParams = None
    vol_spike: VolSpikeParams = None
    event_action: EventActionParams = None
    auto_resume: AutoResumeParams = None

    def __post_init__(self):
        if self.backtest is None:
            self.backtest = BacktestParams()
        if self.sensitivity is None:
            self.sensitivity = SensitivityParams()
        if self.walk_forward is None:
            self.walk_forward = WalkForwardParams()
        if self.regime is None:
            self.regime = RegimeParams()
        if self.trading_hours is None:
            self.trading_hours = TradingHoursParams()
        if self.news_watch is None:
            self.news_watch = NewsWatchParams()
        if self.vol_spike is None:
            self.vol_spike = VolSpikeParams()
        if self.event_action is None:
            self.event_action = EventActionParams()
        if self.auto_resume is None:
            self.auto_resume = AutoResumeParams()

    @classmethod
    def from_yaml(cls) -> "AllStrategyParams":
        return cls(
            trend=TrendFilterParams.from_yaml(),
            momentum=MomentumParams.from_yaml(),
            micro=MicroParams.from_yaml(),
            risk=RiskParams.from_yaml(),
            filters=FiltersParams.from_yaml(),
            scoring=ScoringParams.from_yaml(),
            sentiment=SentimentParams.from_yaml(),
            macro=MacroParams.from_yaml(),
            backtest=BacktestParams.from_yaml(),
            sensitivity=SensitivityParams.from_yaml(),
            walk_forward=WalkForwardParams.from_yaml(),
            regime=RegimeParams.from_yaml(),
            trading_hours=TradingHoursParams.from_yaml(),
            news_watch=NewsWatchParams.from_yaml(),
            vol_spike=VolSpikeParams.from_yaml(),
            event_action=EventActionParams.from_yaml(),
            auto_resume=AutoResumeParams.from_yaml(),
        )
