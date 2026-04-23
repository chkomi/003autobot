"""
시그널 통합기 (핵심 인텔리전스).
Layer 1~3 이진 확인 + 종합 스코어링 시스템을 결합해
최종 거래 시그널을 생성한다.
진입가, SL, TP1, ATR 등 주문에 필요한 모든 값을 계산한다.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from loguru import logger

from config.strategy_params import AllStrategyParams
from data.market_data import MarketData
from strategy.indicators import calc_atr
from strategy.micro_confirmation import MicroConfirmation, MicroSignal
from strategy.momentum_trigger import MomentumTrigger, MomentumSignal
from strategy.score_engine import MacroData, ScoreBreakdown, ScoreEngine, SentimentData
from strategy.regime_detector import RegimeDetector, RegimeResult
from strategy.trend_filter import TrendFilter, TrendSignal


@dataclass
class SignalResult:
    """거래 시그널 결과"""
    direction: str          # "LONG" | "SHORT" | "FLAT"
    confidence: float       # 0.0 ~ 1.0 (레이어별 점수 합산)

    symbol: str = ""        # ccxt 심볼 (예: BTC/USDT:USDT)
    entry_price: float = 0.0
    stop_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    atr_value: float = 0.0

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # 디버깅용 레이어 결과 보관
    trend: Optional[TrendSignal] = None
    momentum: Optional[MomentumSignal] = None
    micro: Optional[MicroSignal] = None
    score: Optional[ScoreBreakdown] = None  # 종합 스코어
    regime: Optional[RegimeResult] = None   # 시장 레짐

    @property
    def is_actionable(self) -> bool:
        return self.direction in ("LONG", "SHORT")

    def summary(self) -> str:
        if not self.is_actionable:
            return f"FLAT (신호 없음)"
        score_str = f" | 스코어: {self.score.total:.1f}" if self.score else ""
        return (
            f"{self.direction} | 진입: ${self.entry_price:,.2f} | "
            f"SL: ${self.stop_price:,.2f} | TP1: ${self.tp1_price:,.2f} | "
            f"TP2: ${self.tp2_price:,.2f} | "
            f"ATR: ${self.atr_value:.2f} | 신뢰도: {self.confidence:.0%}"
            f"{score_str}"
        )


class SignalAggregator:
    """Triple Confirmation + Scoring System 시그널 통합기"""

    def __init__(self, params: AllStrategyParams):
        self._params = params
        self._trend_filter = TrendFilter(params.trend)
        self._momentum = MomentumTrigger(params.momentum)
        self._micro = MicroConfirmation(params.micro)
        self._score_engine = ScoreEngine(params.scoring, params.sentiment, params.macro)
        self._regime_detector = RegimeDetector(
            adx_trending_threshold=params.regime.adx_trending_threshold,
            adx_ranging_threshold=params.regime.adx_ranging_threshold,
            bb_bandwidth_low=params.regime.bb_bandwidth_low,
            bb_bandwidth_high=params.regime.bb_bandwidth_high,
            atr_pct_volatile=params.regime.atr_pct_volatile,
        )

    async def evaluate(
        self,
        market: MarketData,
        funding_rate: Optional[float] = None,
        open_interest: Optional[float] = None,
        prev_open_interest: Optional[float] = None,
        long_short_ratio: Optional[float] = None,
        symbol: Optional[str] = None,
        macro: Optional[MacroData] = None,
    ) -> SignalResult:
        """시장 데이터를 분석해 현재 시그널을 반환한다.

        Args:
            market: 캔들/시세
            funding_rate: 현재 펀딩비 (없으면 필터 스킵)
            open_interest: 현재 미결제약정
            prev_open_interest: 이전 미결제약정 (변화율 계산용)
            long_short_ratio: 현재 롱숏비율
            symbol: 평가할 심볼 (없으면 market의 첫 번째 심볼)
            macro: 거시경제 매크로 데이터
        """
        flat = SignalResult(direction="FLAT", confidence=0.0, symbol=symbol or "")

        # 이벤트 블랙아웃: 지정 시각 ±N분 진입 차단
        f = self._params.filters
        if f.event_blackout_enabled and self._in_event_blackout():
            logger.info("이벤트 블랙아웃 구간 — 신규 진입 차단")
            return flat

        # 거래 시간 필터
        th = self._params.trading_hours
        if th.enabled:
            now_hour = datetime.now(timezone.utc).hour
            if not (th.start_hour_utc <= now_hour < th.end_hour_utc):
                logger.info(f"거래 시간 외 (UTC {now_hour}시) — 진입 차단")
                return flat

        try:
            # Layer 1: 추세 필터 (4H)
            df_4h = market.get_candles(self._params.trend.timeframe, symbol=symbol)
            trend_signal = self._trend_filter.analyze(df_4h)

            sym_tag = symbol.split("/")[0] if symbol else ""
            if trend_signal is None or trend_signal.regime == "NEUTRAL":
                logger.info(f"🔍 [{sym_tag}] 시그널 체크 — Layer1=NEUTRAL → FLAT")
                return flat

            # 시장 레짐 감지
            regime_result = self._regime_detector.detect(df_4h)
            if regime_result and regime_result.is_ranging and self._params.regime.block_ranging:
                logger.info(f"🔍 [{sym_tag}] 횡보 시장 감지 (ADX={regime_result.adx_value:.1f}) — 진입 차단")
                return flat

            # Layer 2: 모멘텀 트리거 (1H)
            df_1h = market.get_candles(self._params.momentum.timeframe, symbol=symbol)
            mom_signal = self._momentum.analyze(df_1h)

            # Layer 3: 미시 확인 (15M)
            df_15m = market.get_candles(self._params.micro.timeframe, symbol=symbol)
            micro_signal = self._micro.analyze(df_15m)

            if mom_signal is None and micro_signal is None:
                logger.info(f"🔍 [{sym_tag}] 시그널 체크 — Layer1={trend_signal.regime} / Layer2,3=데이터부족 → FLAT")
                return flat

            # Layer2 / Layer3 각각의 방향 일치 여부 (OR 조건)
            layer2_pass = False
            layer3_pass = False
            mom_brief = "N/A"
            micro_brief = "N/A"

            if mom_signal is not None:
                mom_brief = f"MACD={mom_signal.macd_cross},RSI={mom_signal.rsi_value:.1f}"
                if trend_signal.is_bullish and mom_signal.long_trigger:
                    layer2_pass = True
                elif trend_signal.is_bearish and mom_signal.short_trigger:
                    layer2_pass = True

            if micro_signal is not None:
                micro_brief = f"BB={micro_signal.bb_pct:.2f},Vol={'OK' if micro_signal.volume_confirmed else 'X'}"
                if trend_signal.is_bullish and micro_signal.long_confirm:
                    layer3_pass = True
                elif trend_signal.is_bearish and micro_signal.short_confirm:
                    layer3_pass = True

            # 시그널 결합 모드: AND (엄격) / OR (완화: 하나만 통과해도 스코어로 검증)
            combine_mode = (self._params.scoring.signal_combine_mode or "AND").upper()
            if combine_mode == "OR":
                layers_ok = layer2_pass or layer3_pass
            elif combine_mode == "SCORE_ONLY":
                layers_ok = True  # 스코어 게이트만으로 진입 (아래에서 검증)
            else:  # AND
                layers_ok = layer2_pass and layer3_pass

            if not layers_ok:
                l2_tag = "OK" if layer2_pass else f"FAIL({mom_brief})"
                l3_tag = "OK" if layer3_pass else f"FAIL({micro_brief})"
                logger.info(f"🔍 [{sym_tag}] 시그널 체크 — Layer1={trend_signal.regime} / Layer2={l2_tag} / Layer3={l3_tag} / mode={combine_mode} → FLAT")
                return flat

            # 진입 확정 — 어느 레이어가 통과했는지 로그
            passed_layers = []
            if layer2_pass:
                passed_layers.append(f"Layer2({mom_brief})")
            if layer3_pass:
                passed_layers.append(f"Layer3({micro_brief})")
            logger.info(f"✅ [{sym_tag}] 시그널 진입 조건 충족 — Layer1={trend_signal.regime} / {' + '.join(passed_layers)}")

            # micro_signal이 None이면 현재가를 진입가로 사용
            if micro_signal is None:
                # 미시 데이터 없을 때는 최신 1h 종가를 진입가로
                entry_price_fallback = float(df_1h["close"].iloc[-1])
                # 더미 micro_signal 생성용 플래그
                micro_signal = None  # SignalResult에서 None 허용

            # ── 시그널 생성 ───────────────────────────────────────
            direction = "LONG" if trend_signal.is_bullish else "SHORT"
            # 진입가: micro_signal이 있으면 15m 종가, 없으면 1h 종가
            entry_price = micro_signal.price if micro_signal is not None else float(df_1h["close"].iloc[-1])

            # ATR(1H) 기반 SL/TP 계산
            atr_val = self._calc_atr_1h(df_1h)
            if atr_val is None or atr_val == 0:
                logger.warning("ATR 계산 실패 — 시그널 무시")
                return flat

            # 적응형 ATR 멀티플라이어 (변동성 기반)
            atr_pct = atr_val / entry_price if entry_price > 0 else 0
            rp = self._params.risk
            if atr_pct < 0.005:
                # 저변동: SL 좁게, TP 넓게
                sl_mult = rp.atr_sl_multiplier * 0.8
                tp1_mult = rp.atr_tp1_multiplier * 1.25
                tp2_mult = rp.atr_tp2_multiplier * 1.15
            elif atr_pct > 0.015:
                # 고변동 (BTC 등): SL 넓게, TP 타이트
                sl_mult = rp.atr_sl_multiplier * 1.2
                tp1_mult = rp.atr_tp1_multiplier
                tp2_mult = rp.atr_tp2_multiplier * 0.85
            else:
                sl_mult = rp.atr_sl_multiplier
                tp1_mult = rp.atr_tp1_multiplier
                tp2_mult = rp.atr_tp2_multiplier

            if direction == "LONG":
                stop_price = entry_price - atr_val * sl_mult
                tp1_price = entry_price + atr_val * tp1_mult
                tp2_price = entry_price + atr_val * tp2_mult
            else:
                stop_price = entry_price + atr_val * sl_mult
                tp1_price = entry_price - atr_val * tp1_mult
                tp2_price = entry_price - atr_val * tp2_mult

            # 펀딩비 필터: 과열 방향의 포지션 진입 차단
            # (양수 펀딩 = 롱이 숏에게 지급 → 롱 과열 → 롱 진입 차단)
            if (
                f.funding_filter_enabled
                and funding_rate is not None
                and abs(funding_rate) >= f.funding_rate_abs_threshold
            ):
                if funding_rate > 0 and direction == "LONG":
                    logger.info(f"펀딩비 과열({funding_rate:.4%}) — 롱 진입 차단")
                    return flat
                if funding_rate < 0 and direction == "SHORT":
                    logger.info(f"펀딩비 과열({funding_rate:.4%}) — 숏 진입 차단")
                    return flat

            # ── 종합 스코어링 평가 ────────────────────────────────
            sentiment_data = SentimentData(
                funding_rate=funding_rate,
                open_interest=open_interest,
                prev_open_interest=prev_open_interest,
                long_short_ratio=long_short_ratio,
            )
            score_result = self._score_engine.evaluate(
                df_4h, df_1h, df_15m, sentiment=sentiment_data, macro=macro
            )

            # 스코어 기준 미달 시 약한 시그널로 강등 또는 거부
            sp = self._params.scoring
            if score_result.total < sp.weak_signal_threshold:
                logger.info(
                    f"스코어 미달 ({score_result.total:.1f} < {sp.weak_signal_threshold}) — 진입 거부"
                )
                return flat

            # 신뢰도: 스코어 기반으로 산출 (0.6~1.0)
            if score_result.total >= sp.strong_signal_threshold:
                confidence = 1.0
            else:
                # 60~75 → 0.6~1.0 선형 보간
                confidence = 0.6 + 0.4 * (
                    (score_result.total - sp.weak_signal_threshold)
                    / (sp.strong_signal_threshold - sp.weak_signal_threshold)
                )

            # EMA 완전 정렬 보너스
            if trend_signal.ema_aligned:
                confidence = min(confidence + 0.05, 1.0)

            signal = SignalResult(
                direction=direction,
                confidence=confidence,
                symbol=symbol or "",
                entry_price=entry_price,
                stop_price=stop_price,
                tp1_price=tp1_price,
                tp2_price=tp2_price,
                atr_value=atr_val,
                trend=trend_signal,
                momentum=mom_signal,
                micro=micro_signal,
                score=score_result,
                regime=regime_result,
            )

            logger.info(f"[시그널] {signal.summary()}")
            return signal

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"시그널 평가 중 데이터 오류: {e}")
            return flat

    def _in_event_blackout(self) -> bool:
        """이벤트 블랙아웃(지정 UTC 시각 ±N분)에 현재가 포함되는지."""
        f = self._params.filters
        if not f.event_times_utc:
            return False
        now = datetime.now(timezone.utc)
        window = timedelta(minutes=f.event_blackout_minutes)
        for ts in f.event_times_utc:
            try:
                event = datetime.strptime(ts, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            except ValueError:
                logger.debug(f"이벤트 블랙아웃 시각 파싱 실패: {ts}")
                continue
            if event - window <= now <= event + window:
                return True
        return False

    def _calc_atr_1h(self, df_1h: pd.DataFrame) -> Optional[float]:
        """1H 타임프레임의 최신 ATR 값을 반환한다."""
        try:
            atr = calc_atr(
                df_1h["high"],
                df_1h["low"],
                df_1h["close"],
                self._params.risk.atr_period,
            )
            val = float(atr.iloc[-1])
            return val if not pd.isna(val) else None
        except (KeyError, ZeroDivisionError) as e:
            logger.debug(f"ATR 계산 실패: {e}")
            return None
