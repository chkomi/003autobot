"""
봇 메인 오케스트레이션 엔진 (고도화 버전).

비동기 루프:
  1. 캔들 데이터 갱신 (멀티 심볼)
  2. 시그널 평가 (멀티 심볼)
  3. 포지션 모니터
  4. 리스크 모니터 (드로다운 감시 + 자동 복구)
  5. 피드백 루프 (성과 분석 + 자동 튜닝)
  6. 매크로 데이터 갱신 (Fear&Greed, BTC.D, 경제 이벤트 캘린더)
  7. [NEW] 뉴스 감시 (암호화폐/지정학 키워드 실시간 위험도 평가)
  8. [NEW] 변동성 스파이크 감시 (15m 실현변동성 급증 감지)
  9. [NEW] 이벤트 선제 대응 (고영향 이벤트 N시간 전 자동 축소/청산)
"""
import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

# KST = UTC+9 (한국 표준시)
_KST = timezone(timedelta(hours=9))

import aiohttp
from loguru import logger

from config.settings import Settings
from core.exceptions import BotHaltError, MaxDrawdownError
from core.state_manager import BotState, StateManager
from data.market_data import MarketData
from database.db_manager import DatabaseManager
from database.models import BotEvent
from execution.order_manager import OrderManager
from execution.position_tracker import PositionTracker
from execution.trade_lifecycle import TradeLifecycle
from notification.notification_manager import NotificationManager
from risk.risk_manager import RiskCheckResult, RiskManager
from risk.stop_manager import StopManager
from dashboard.server import push_sse_event
from data.macro_news import MacroNewsCollector, MacroSnapshot
from data.news_collector import NewsCollector, NewsRiskSnapshot
from risk.goal_tracker import GoalTracker
from strategy.feedback_loop import FeedbackLoop
from strategy.symbol_ranker import SymbolRanker
from strategy.score_engine import MacroData
from strategy.ml_filter import MLSignalFilter
from strategy.signal_aggregator import SignalAggregator
from strategy.volatility_spike import VolatilitySpikeDetector


class BotEngine:
    """자동매매봇 최상위 오케스트레이터 (멀티 심볼 지원)"""

    def __init__(
        self,
        settings: Settings,
        market: MarketData,
        aggregator: SignalAggregator,
        risk_mgr: RiskManager,
        order_mgr: OrderManager,
        stop_mgr: StopManager,
        position_tracker: PositionTracker,
        lifecycle: TradeLifecycle,
        notifier: NotificationManager,
        db: DatabaseManager,
        params: Optional["AllStrategyParams"] = None,
    ):
        self._s = settings
        self._market = market
        self._aggregator = aggregator
        self._risk = risk_mgr
        self._orders = order_mgr
        self._stops = stop_mgr
        self._tracker = position_tracker
        self._lifecycle = lifecycle
        self._notifier = notifier
        self._db = db
        self._state = StateManager()
        self._tasks: list[asyncio.Task] = []
        self._feedback = FeedbackLoop(db)
        self._goal = GoalTracker(db, settings.goal, notifier)
        self._ml_filter = MLSignalFilter()
        # 심볼 로테이터 (주간 모멘텀 기반)
        self._symbol_ranker: Optional[SymbolRanker] = None  # REST 클라이언트 접근 후 초기화
        self._active_symbols: list[str] = list(settings.trading.symbol_list)
        self._last_symbol_rotation: Optional[datetime] = None
        self._macro_collector = MacroNewsCollector()
        self._macro_snapshot: MacroSnapshot | None = None
        # 심볼별 OI 변화율 추적
        self._prev_oi: dict[str, float | None] = {}

        # [고도화] 뉴스/스파이크/이벤트 대응 구성요소
        from config.strategy_params import AllStrategyParams
        self._params: AllStrategyParams = params or AllStrategyParams.from_yaml()

        news_params = self._params.news_watch
        vol_params = self._params.vol_spike
        self._news_collector = NewsCollector(
            cryptopanic_api_key=os.getenv("CRYPTOPANIC_API_KEY", ""),
            lookback_minutes=news_params.lookback_minutes,
            thresholds={
                "elevated": news_params.elevated_threshold,
                "high": news_params.high_threshold,
                "critical": news_params.critical_threshold,
            },
        )
        self._news_snapshot: NewsRiskSnapshot | None = None
        self._spike_detector = VolatilitySpikeDetector(
            recent_bars=vol_params.recent_bars,
            baseline_bars=vol_params.baseline_bars,
            vol_ratio_threshold=vol_params.vol_ratio_threshold,
            z_threshold=vol_params.z_threshold,
            range_ratio_threshold=vol_params.range_ratio_threshold,
        )
        # 이벤트/스파이크 후 일시 정지 해제 시각
        self._pause_until: Optional[datetime] = None
        # 이벤트별 선제 축소/청산 적용 기록 (중복 트리거 방지)
        self._event_actions_applied: dict[str, str] = {}  # event_title → "REDUCED"|"CLOSED"
        # 스파이크 중복 트리거 방지: 심볼별 마지막 스파이크 캔들 타임스탬프
        self._last_spike_candle_ts: dict[str, object] = {}  # symbol → pd.Timestamp
        # 자동 복구 카운터 (day key → count)
        self._auto_resume_counts: dict[str, int] = {}

        # [Watchdog] 시그널 루프 마지막 실행 시각 + 에러 알림 쿨다운
        self._last_signal_check: Optional[datetime] = None
        self._watchdog_alerted: bool = False          # 중복 알림 방지
        # 루프별 연속 에러 카운터 및 마지막 알림 시각
        self._loop_error_counts: dict[str, int] = {}
        self._loop_error_last_alert: dict[str, datetime] = {}
        self._loop_error_alert_threshold: int = 5    # N회 연속 에러 시 텔레그램 발송
        self._loop_error_alert_cooldown_sec: int = 600  # 같은 루프 알림 재발송 쿨다운(초)

        # [버그2] KST 자정 일일 리셋 감지용 날짜 추적
        self._current_kst_date: Optional[str] = None

        # [버그3] HALT 즉시 중단을 위한 이벤트 (set → 모든 루프 즉시 탈출)
        self._halt_event: asyncio.Event = asyncio.Event()

    @property
    def state(self) -> BotState:
        return self._state.state

    async def start(self) -> None:
        """봇을 시작하고 모든 루프를 실행한다."""
        symbols = self._s.trading.symbol_list
        logger.info("=" * 60)
        logger.info("OKX 자동매매봇 시작 (멀티 심볼)")
        logger.info(f"심볼: {', '.join(symbols)}")
        logger.info(f"레버리지: {self._s.trading.leverage}x")
        logger.info(f"모드: {'페이퍼 트레이딩' if self._s.okx.is_demo else '실거래'}")
        logger.info("=" * 60)

        # 시장 데이터 워밍업 (모든 심볼)
        logger.info("시장 데이터 로딩 중...")
        await self._market.warm_up()

        self._state.transition(BotState.WATCHING, "초기화 완료")

        # 시작 알림
        await self._notifier.on_startup(
            ", ".join(symbols),
            self._s.trading.leverage,
            self._s.okx.is_demo,
        )

        # 매크로 데이터 초기 로딩
        try:
            self._macro_snapshot = await self._macro_collector.get_snapshot()
            logger.info(f"매크로 데이터 로딩 완료: {self._macro_snapshot.summary()}")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"매크로 데이터 초기 로딩 실패 (네트워크): {e}")
        except (KeyError, ValueError) as e:
            logger.warning(f"매크로 데이터 초기 로딩 실패 (파싱): {e}")

        # [버그1] 봇 시작 시 포지션 reconciliation — 유령/고아 포지션 감지 및 복구
        await self._reconcile_positions()

        # 비동기 루프 시작
        self._tasks = [
            asyncio.create_task(self._candle_refresh_loop(), name="candle-loop"),
            asyncio.create_task(self._signal_loop(), name="signal-loop"),
            asyncio.create_task(self._position_monitor_loop(), name="position-loop"),
            asyncio.create_task(self._risk_monitor_loop(), name="risk-loop"),
            asyncio.create_task(self._feedback_loop(), name="feedback-loop"),
            asyncio.create_task(self._macro_refresh_loop(), name="macro-loop"),
        ]
        # [고도화] 뉴스/스파이크/이벤트 대응 루프
        if self._params.news_watch.enabled:
            self._tasks.append(asyncio.create_task(self._news_watch_loop(), name="news-loop"))
        if self._params.vol_spike.enabled:
            self._tasks.append(asyncio.create_task(self._volatility_spike_loop(), name="spike-loop"))
        if self._params.event_action.enabled:
            self._tasks.append(asyncio.create_task(self._event_action_loop(), name="event-loop"))
        # [Watchdog] 시그널 체크 감시 루프
        self._tasks.append(asyncio.create_task(self._watchdog_loop(), name="watchdog-loop"))
        # [10x] 주간 심볼 로테이션 루프
        rotation_cfg = getattr(self._params, 'x10_goal', None)
        if rotation_cfg and getattr(getattr(rotation_cfg, 'symbol_rotation', None), 'enabled', False):
            self._symbol_ranker = SymbolRanker(self._market._rest)
            self._tasks.append(asyncio.create_task(self._symbol_rotation_loop(), name="rotation-loop"))

        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        """봇 종료"""
        logger.info("봇 종료 중...")
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._macro_collector.close()
        await self._news_collector.close()
        await self._notifier.close()
        logger.info("봇 종료 완료")

    # ── 루프 1: 캔들 데이터 갱신 (멀티 심볼) ─────────────────────

    async def _candle_refresh_loop(self) -> None:
        """모든 심볼의 타임프레임별 캔들을 주기적으로 갱신한다."""
        # [버그3] _halt_event 기반 조건 — HALT 시 즉시 탈출
        while not self._halt_event.is_set():
            try:
                for symbol in self._market.symbols:
                    if self._halt_event.is_set():
                        return
                    for tf in self._s.trading.all_timeframes:
                        await self._market.refresh_candles(tf, symbol=symbol)
                await self._sleep_interruptible(self._s.trading.signal_check_interval_sec)
            except asyncio.CancelledError:
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"캔들 갱신 루프 네트워크 오류: {e}")
                await self._maybe_send_loop_error_alert("candle-loop", f"네트워크 오류: {e}")
                await self._sleep_interruptible(30)
            except (KeyError, ValueError) as e:
                logger.error(f"캔들 갱신 루프 데이터 오류: {e}")
                await self._maybe_send_loop_error_alert("candle-loop", f"데이터 오류: {e}")
                await self._sleep_interruptible(30)

    # ── 루프 2: 시그널 평가 (멀티 심볼) ─────────────────────────

    async def _signal_loop(self) -> None:
        """각 심볼별로 시그널을 평가하고 진입 조건 충족 시 포지션을 연다."""
        # [버그3] _halt_event 기반 조건 — HALT 시 즉시 탈출
        while not self._halt_event.is_set():
            try:
                await self._sleep_interruptible(self._s.trading.signal_check_interval_sec)
                if self._halt_event.is_set():
                    break

                # Watchdog: 루프 생존 확인
                self._last_signal_check = datetime.now(timezone.utc)
                self._watchdog_alerted = False  # 정상 동작 중 → 알림 리셋
                self._loop_error_counts["signal-loop"] = 0

                # 신규 진입 차단 조건 (PAUSED_EVENT / 스파이크 이후 쿨다운)
                if not self._state.can_trade:
                    continue
                now = datetime.now(timezone.utc)
                if self._pause_until and now < self._pause_until:
                    continue

                for symbol in self._market.symbols:
                    await self._evaluate_symbol(symbol)

            except asyncio.CancelledError:
                break
            except BotHaltError as e:
                await self._halt(str(e))
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"시그널 루프 네트워크 오류: {e}")
                self._state.transition(BotState.WATCHING, f"네트워크 예외 복구: {e}")
                await self._maybe_send_loop_error_alert("signal-loop", f"네트워크 오류: {e}")
                await self._sleep_interruptible(10)
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"시그널 루프 데이터 오류: {e}")
                self._state.transition(BotState.WATCHING, f"데이터 예외 복구: {e}")
                await self._maybe_send_loop_error_alert("signal-loop", f"데이터 오류: {e}")
                await self._sleep_interruptible(10)

    async def _evaluate_symbol(self, symbol: str) -> None:
        """단일 심볼에 대한 시그널 평가 및 진입 처리"""
        sym_tag = symbol.split("/")[0]

        if not self._market.is_ready(symbol=symbol):
            logger.debug(f"[{sym_tag}] 시장 데이터 준비 안됨 — 스킵")
            return

        # 심리 데이터 조회 (실패 시 None → 필터 스킵)
        funding_rate = None
        open_interest = None
        long_short_ratio = None
        try:
            funding_rate = await self._market.get_funding_rate(symbol=symbol)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.debug(f"[{symbol}] funding_rate 조회 실패: {e}")
        try:
            open_interest = await self._market.get_open_interest(symbol=symbol)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.debug(f"[{symbol}] open_interest 조회 실패: {e}")
        try:
            long_short_ratio = await self._market.get_long_short_ratio(symbol=symbol)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.debug(f"[{symbol}] long_short_ratio 조회 실패: {e}")

        prev_oi = self._prev_oi.get(symbol)

        # 매크로 데이터를 MacroData로 변환
        macro_data = None
        if self._macro_snapshot:
            ms = self._macro_snapshot
            # 뉴스 위험도를 매크로에 반영 (CRITICAL/HIGH면 신규 진입 스코어를 강하게 감점)
            if self._news_snapshot is not None:
                ms.news_risk_level = self._news_snapshot.level
                ms.news_risk_score = self._news_snapshot.aggregate_score
                if self._news_snapshot.top_items:
                    ms.news_top_headline = self._news_snapshot.top_items[0].title[:80]
            macro_data = MacroData(
                fear_greed_index=ms.fear_greed_index,
                fear_greed_label=ms.fear_greed_label,
                upcoming_high_impact_count=ms.upcoming_high_impact_count,
                nearest_high_impact_hours=ms.nearest_high_impact_hours,
                nearest_event_title=ms.nearest_event_title,
                btc_dominance=ms.btc_dominance,
                total_market_cap_change_24h=ms.total_market_cap_change_24h,
            )

        # 시그널 평가 (스코어링 + 3레이어 확인 + 매크로)
        signal = await self._aggregator.evaluate(
            self._market,
            funding_rate=funding_rate,
            open_interest=open_interest,
            prev_open_interest=prev_oi,
            long_short_ratio=long_short_ratio,
            symbol=symbol,
            macro=macro_data,
        )
        # OI 변화 추적 갱신
        if open_interest is not None:
            self._prev_oi[symbol] = open_interest
        if not signal.is_actionable:
            return

        # ML 시그널 필터 적용
        ml_pred = self._ml_filter.predict(
            score_total=signal.score.total if signal.score else 50.0,
            atr_pct=signal.atr_value / signal.entry_price if signal.entry_price > 0 else 0.01,
            hour_of_day=datetime.now(timezone.utc).hour,
            day_of_week=datetime.now(timezone.utc).weekday(),
            regime=signal.regime.regime if signal.regime else "TRENDING",
        )
        if not ml_pred.should_trade:
            logger.info(f"[{sym_tag}] ML 필터 차단: {ml_pred.summary()}")
            return

        self._state.transition(BotState.SIGNAL_DETECTED, f"{sym_tag} {signal.direction}")
        logger.info(f"[{sym_tag}] [시그널] {signal.summary()}")
        await push_sse_event("signal", {"symbol": symbol, "direction": signal.direction, "confidence": signal.confidence})

        # 잔고 조회
        balance_info = await self._market.get_balance()
        balance = balance_info.get("free", 0)

        # 열린 포지션 조회
        open_trades = await self._db.fetch_open_trades()

        # 같은 심볼 같은 방향 중복 방지
        same_symbol_dir = [
            t for t in open_trades
            if t.symbol == symbol and t.direction == signal.direction
        ]
        if same_symbol_dir:
            logger.info(f"[{sym_tag}] {signal.direction} 방향 포지션이 이미 열려 있음 — 스킵")
            self._state.transition(BotState.WATCHING, "중복 포지션")
            return

        # 리스크 검증
        report = await self._risk.pre_trade_check(signal, balance, open_trades)

        if report.result != RiskCheckResult.APPROVED:
            logger.info(f"[{sym_tag}] 진입 거부: {report.reason}")
            self._state.transition(BotState.WATCHING, "진입 거부")
            return

        # 포지션 진입
        trade = await self._orders.open_position(signal, report.position_size)
        await self._db.insert_trade(trade)

        await self._notifier.on_trade_opened(trade, signal.entry_price)

        # [버그4] 부분 체결 발생 시 텔레그램 알림
        is_partial = getattr(trade, "partial_fill", False)
        requested_qty = getattr(trade, "requested_qty", report.position_size.quantity)
        if is_partial:
            await self._notifier.on_alert(
                "WARNING",
                (
                    f"⚠️ <b>[부분 체결]</b> {sym_tag} {trade.direction}\n"
                    f"요청 수량: {requested_qty:.4f} → 체결 수량: {trade.quantity:.4f}\n"
                    f"미체결: {requested_qty - trade.quantity:.4f} | "
                    f"SL/TP는 체결 수량 기준으로 설정됨"
                ),
            )

        # 진입 컨텍스트(score breakdown, regime 등)를 metadata에 기록 (Trade Journal 백필 소스)
        entry_metadata = {
            "trade_id": trade.trade_id,
            "symbol": symbol,
            "direction": trade.direction,
        }
        if signal.score is not None:
            entry_metadata["score"] = {
                "total": round(signal.score.total, 2),
                "trend": round(signal.score.trend, 2),
                "momentum": round(signal.score.momentum, 2),
                "volume": round(signal.score.volume, 2),
                "volatility": round(signal.score.volatility, 2),
                "sentiment": round(signal.score.sentiment, 2),
                "macro": round(signal.score.macro, 2),
                "strength": signal.score.signal_strength,
            }
        if signal.regime is not None:
            entry_metadata["regime"] = {
                "label": signal.regime.regime,
                "adx": round(signal.regime.adx_value, 2),
                "bb_bw": round(signal.regime.bb_bandwidth, 4),
                "atr_pct": round(signal.regime.atr_pct, 4),
            }
        if signal.momentum is not None:
            entry_metadata["momentum"] = {
                "macd_cross": signal.momentum.macd_cross,
                "rsi": round(signal.momentum.rsi_value, 2),
            }
        if signal.micro is not None:
            entry_metadata["micro"] = {
                "bb_pct": round(signal.micro.bb_pct, 3),
                "volume_confirmed": signal.micro.volume_confirmed,
            }

        await self._db.log_event(BotEvent(
            event_type="ORDER",
            level="INFO",
            message=f"포지션 오픈: {trade.trade_id} {sym_tag} {trade.direction}"
                    + (f" [부분체결:{trade.quantity:.4f}/{requested_qty:.4f}]" if is_partial else ""),
            metadata=entry_metadata,
        ))

        self._state.transition(BotState.IN_TRADE, f"{sym_tag} {trade.trade_id}")
        await push_sse_event("trade", {"action": "opened", "trade_id": trade.trade_id, "symbol": symbol, "direction": trade.direction, "entry_price": trade.entry_price})

    # ── 루프 3: 포지션 모니터 ────────────────────────────────────

    async def _position_monitor_loop(self) -> None:
        """열린 포지션을 주기적으로 점검한다."""
        # [버그3] _halt_event 기반 조건 — HALT 시 즉시 탈출
        while not self._halt_event.is_set():
            try:
                await self._sleep_interruptible(self._s.trading.position_monitor_interval_sec)
                if self._halt_event.is_set():
                    break

                open_trades = await self._tracker.sync_open_positions()
                if not open_trades:
                    if self._state.state == BotState.IN_TRADE:
                        self._state.transition(BotState.WATCHING, "포지션 없음")
                    continue

                # 심볼별로 현재가 및 ATR 조회 후 모니터링
                for trade in open_trades:
                    try:
                        current_price = await self._market.get_current_price(symbol=trade.symbol)

                        # 1H ATR 조회
                        df_1h = self._market.get_candles(
                            self._s.trading.timeframe_momentum,
                            symbol=trade.symbol,
                        )
                        from strategy.indicators import calc_atr
                        import pandas as pd
                        atr_s = calc_atr(df_1h["high"], df_1h["low"], df_1h["close"])
                        atr_val = float(atr_s.iloc[-1]) if not pd.isna(atr_s.iloc[-1]) else 500.0

                        exit_reason = await self._lifecycle.monitor(trade, current_price, atr_val)
                        if exit_reason:
                            sym_tag = trade.symbol.split("/")[0]
                            logger.info(f"[{sym_tag}] 포지션 청산됨: {trade.trade_id} ({exit_reason})")
                            await push_sse_event("trade", {"action": "closed", "trade_id": trade.trade_id, "symbol": trade.symbol, "reason": exit_reason})
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        logger.error(f"포지션 모니터 ({trade.trade_id}) 네트워크 오류: {e}")
                    except (KeyError, ValueError, TypeError) as e:
                        logger.error(f"포지션 모니터 ({trade.trade_id}) 데이터 오류: {e}")

                # 포지션이 남아있으면 IN_TRADE, 아니면 WATCHING
                remaining = await self._db.fetch_open_trades()
                if remaining and self._state.state not in (BotState.IN_TRADE, BotState.PAUSED_EVENT):
                    self._state.transition(BotState.IN_TRADE)
                elif not remaining and self._state.state == BotState.IN_TRADE:
                    self._state.transition(BotState.WATCHING, "모든 포지션 청산")

                # 일시정지 만료 체크
                if self._pause_until and datetime.now(timezone.utc) >= self._pause_until:
                    self._pause_until = None
                    if self._state.is_paused:
                        self._state.unpause("이벤트/스파이크 쿨다운 종료")

            except asyncio.CancelledError:
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"포지션 모니터 루프 네트워크 오류: {e}")
                await self._maybe_send_loop_error_alert("position-loop", f"네트워크 오류: {e}")
                await self._sleep_interruptible(15)
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"포지션 모니터 루프 데이터 오류: {e}")
                await self._maybe_send_loop_error_alert("position-loop", f"데이터 오류: {e}")
                await self._sleep_interruptible(15)

    # ── 루프 4: 리스크 모니터 ────────────────────────────────────

    async def _risk_monitor_loop(self) -> None:
        """드로다운/일일 손실 감시 + [고도화] HALTED 상태에서의 자동 복구 체크.
        [버그2] KST 자정 일일 리셋 감지 및 텔레그램 알림 추가.
        """
        while True:  # HALTED 상태에서도 자동 복구 체크를 위해 계속 돌린다
            try:
                await asyncio.sleep(60)  # 1분 주기

                # [버그2] KST 자정 감지 → 일일 손실 한도 자동 리셋
                now_kst_date = datetime.now(_KST).strftime("%Y-%m-%d")
                if self._current_kst_date is None:
                    self._current_kst_date = now_kst_date  # 최초 초기화
                elif now_kst_date != self._current_kst_date:
                    prev_date = self._current_kst_date
                    self._current_kst_date = now_kst_date
                    self._risk.reset_daily()
                    logger.info(
                        f"[DailyReset] KST 날짜 전환: {prev_date} → {now_kst_date} "
                        f"— 일일 손실 한도 리셋, 거래 재개"
                    )
                    await self._notifier.on_alert(
                        "INFO",
                        (
                            f"🌅 <b>일일 리셋 완료</b>\n"
                            f"날짜 전환: {prev_date} → {now_kst_date} (KST)\n"
                            f"일일 손실 한도 초기화 — 정상 거래 재개"
                        ),
                    )
                    await self._db.log_event(BotEvent(
                        event_type="DAILY_RESET",
                        level="INFO",
                        message=f"KST 자정 일일 리셋: {prev_date} → {now_kst_date}",
                    ))

                balance_info = await self._market.get_balance()
                equity = balance_info.get("total", 0)
                free = balance_info.get("free", 0)
                used = balance_info.get("used", 0)

                await self._risk.record_equity(equity, free, used)

                # HALTED 상태: 자동 복구 시도
                if self._state.is_halted:
                    await self._maybe_auto_resume(equity)
                    continue

                if await self._risk.check_max_drawdown(equity):
                    await self._halt(
                        f"최대 드로다운 초과: 자산 ${equity:,.2f} USDT"
                    )
                    continue

                # 일일 손실 체크
                daily = await self._db.get_daily_pnl()
                loss_limit = equity * self._s.risk.daily_loss_limit_pct
                if daily.pnl_usdt < -loss_limit:
                    await self._notifier.on_alert(
                        "WARNING",
                        f"일일 손실 한도 도달: {daily.pnl_usdt:.2f} USDT (한도: -{loss_limit:.2f})"
                    )

            except asyncio.CancelledError:
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"리스크 모니터 루프 네트워크 오류: {e}")
                await asyncio.sleep(30)
            except (KeyError, ValueError) as e:
                logger.error(f"리스크 모니터 루프 데이터 오류: {e}")
                await asyncio.sleep(30)

    async def _maybe_auto_resume(self, equity: float) -> None:
        """HALTED 상태에서 드로다운이 복구 범위로 들어오면 자동 재개."""
        ar = self._params.auto_resume
        if not ar.enabled:
            return
        if self._state.halt_since is None:
            return
        halt_elapsed = (datetime.now(timezone.utc) - self._state.halt_since).total_seconds() / 60
        if halt_elapsed < ar.min_halt_minutes:
            return
        peak = await self._risk._get_peak_equity(equity)
        if peak <= 0:
            return
        drawdown = (peak - equity) / peak
        if drawdown > ar.recovery_drawdown_pct:
            return

        day_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        count = self._auto_resume_counts.get(day_key, 0)
        if count >= ar.max_auto_resumes_per_day:
            return
        self._auto_resume_counts[day_key] = count + 1

        self._state.resume(reason=f"자동 복구 (drawdown={drawdown:.2%})")
        await self._notifier.on_alert(
            "INFO",
            f"<b>자동 복구</b> — 드로다운 {drawdown:.2%} ≤ {ar.recovery_drawdown_pct:.2%}, 정지 {halt_elapsed:.0f}분 경과",
        )
        await self._db.log_event(BotEvent(
            event_type="AUTO_RESUME",
            level="INFO",
            message=f"자동 복구: drawdown={drawdown:.2%}, halt_elapsed={halt_elapsed:.0f}m",
        ))

    # ── 루프 5: 피드백 루프 (성과 분석) ────────────────────────────

    async def _feedback_loop(self) -> None:
        """KST 자정마다 하루 성과를 분석하고, 자동 튜닝 및 ML 재학습을 수행한다."""
        # [버그3] _halt_event 기반 조건
        while not self._halt_event.is_set():
            try:
                # 다음 KST 자정(00:00)까지 대기
                now_kst = datetime.now(_KST)
                next_midnight_kst = (now_kst + timedelta(days=1)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                sleep_sec = (next_midnight_kst - now_kst).total_seconds()
                await self._sleep_interruptible(sleep_sec)
                if self._halt_event.is_set():
                    break

                # 자정이 지난 직후 → 어제(KST) 날짜로 리포트 생성
                yesterday_kst = (datetime.now(_KST) - timedelta(days=1)).strftime("%Y-%m-%d")
                report = await self._feedback.generate_daily_report(date_str=yesterday_kst)

                # daily_pnl 테이블에서 집계된 당일 수익 조회
                daily_pnl = await self._db.get_daily_pnl(yesterday_kst)
                try:
                    balance_info = await self._market.get_balance()
                    equity = balance_info.get("total", 0.0)
                except Exception:
                    equity = 0.0

                if report.total_trades > 0 or daily_pnl.pnl_usdt != 0:
                    await self._notifier.on_daily_summary(
                        date=yesterday_kst,
                        pnl_usdt=daily_pnl.pnl_usdt,
                        trade_count=daily_pnl.trade_count,
                        win_rate=report.win_rate,
                        equity=equity,
                    )
                    # 상세 분석 리포트는 별도 on_alert로 전송 (제안 사항 포함)
                    if report.suggestions:
                        await self._notifier.on_alert("INFO", report.to_telegram_html())
                    logger.info(f"[피드백] 일일 리포트 전송 ({yesterday_kst})\n{report.summary()}")

                # 포트폴리오 목표 추적 업데이트 (이정표 감지 + 월간 메트릭)
                try:
                    await self._goal.update_nightly(equity, yesterday_kst)
                    # 포트폴리오 요약을 매일 자정 함께 발송
                    portfolio_html = await self._goal.get_portfolio_summary_html(equity)
                    await self._notifier.on_alert("INFO", portfolio_html)
                except Exception as e:
                    logger.error(f"GoalTracker 업데이트 실패: {e}")

                # 자동 파라미터 튜닝 (20건 이상 거래 시)
                try:
                    changes = await self._feedback.auto_tune(min_trades=20)
                    if changes:
                        change_msg = "\n".join(f"  {k}: {v}" for k, v in changes.items())
                        await self._notifier.on_alert(
                            "INFO",
                            f"<b>자동 튜닝 적용</b>\n{change_msg}",
                        )
                except Exception as e:
                    logger.error(f"자동 튜닝 실패: {e}")

                # ML 필터 재학습
                if self._ml_filter.needs_retrain():
                    try:
                        closed_trades = await self._db.fetch_closed_trades(limit=200)
                        trade_dicts = [t.to_dict() for t in closed_trades]
                        if self._ml_filter.train(trade_dicts):
                            importances = self._ml_filter.get_feature_importances()
                            if importances:
                                logger.info(f"[ML필터] 피처 중요도: {importances}")
                    except Exception as e:
                        logger.error(f"ML 필터 재학습 실패: {e}")

            except asyncio.CancelledError:
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"피드백 루프 네트워크 오류: {e}")
                await self._sleep_interruptible(300)
            except (KeyError, ValueError) as e:
                logger.error(f"피드백 루프 데이터 오류: {e}")
                await self._sleep_interruptible(300)

    # ── 루프 5b: 주간 심볼 로테이션 ───────────────────────────────

    async def _symbol_rotation_loop(self) -> None:
        """매주 일요일 KST 자정마다 모멘텀 상위 심볼로 거래 대상을 교체한다."""
        while not self._halt_event.is_set():
            try:
                # 다음 일요일 KST 자정(00:00)까지 대기
                now_kst = datetime.now(_KST)
                days_until_sunday = (6 - now_kst.weekday()) % 7  # 0 = 오늘이 일요일
                if days_until_sunday == 0 and now_kst.hour < 1:
                    days_until_sunday = 0  # 오늘 자정이 지난 지 얼마 안 됨 → 즉시 실행
                else:
                    days_until_sunday = days_until_sunday if days_until_sunday > 0 else 7
                next_sunday = (now_kst + timedelta(days=days_until_sunday)).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                sleep_sec = (next_sunday - now_kst).total_seconds()
                # 최소 10분은 대기 (무한 루프 방지)
                sleep_sec = max(sleep_sec, 600)
                await self._sleep_interruptible(sleep_sec)
                if self._halt_event.is_set():
                    break

                if self._symbol_ranker is None:
                    continue

                # 10x 목표 후보 심볼 풀에서 상위 N개 선택
                rotation_cfg = getattr(self._params, 'x10_goal', None)
                if rotation_cfg is None:
                    continue

                sym_rot = getattr(rotation_cfg, 'symbol_rotation', None)
                if sym_rot is None:
                    continue

                candidates = getattr(sym_rot, 'candidate_pool', self._active_symbols)
                top_n = getattr(sym_rot, 'top_n_symbols', 3)

                try:
                    new_symbols = await self._symbol_ranker.rank_symbols_list(
                        candidates, top_n=top_n
                    )
                except Exception as e:
                    logger.warning(f"[심볼 로테이션] 랭킹 실패 (기존 심볼 유지): {e}")
                    continue

                if not new_symbols:
                    logger.warning("[심볼 로테이션] 빈 결과 — 기존 심볼 유지")
                    continue

                old_symbols = list(self._active_symbols)
                added = [s for s in new_symbols if s not in old_symbols]
                removed = [s for s in old_symbols if s not in new_symbols]

                if not added and not removed:
                    logger.info("[심볼 로테이션] 변경 없음")
                    continue

                # 심볼 업데이트
                self._active_symbols = new_symbols
                self._market.update_symbols(new_symbols)
                self._last_symbol_rotation = datetime.now(_KST)

                # 따뜻한 업 (새 심볼 캔들 로딩)
                if added:
                    logger.info(f"[심볼 로테이션] 신규 심볼 데이터 로딩: {added}")
                    await self._market.warm_up_symbols(added)

                rotation_msg = (
                    f"🔄 <b>주간 심볼 로테이션</b>\n"
                    f"➕ 추가: {', '.join(s.split('/')[0] for s in added) or '-'}\n"
                    f"➖ 제거: {', '.join(s.split('/')[0] for s in removed) or '-'}\n"
                    f"✅ 활성: {', '.join(s.split('/')[0] for s in new_symbols)}"
                )
                logger.info(rotation_msg.replace("<b>", "").replace("</b>", ""))
                await self._notifier.on_alert("INFO", rotation_msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"심볼 로테이션 루프 오류: {e}")
                await self._sleep_interruptible(3600)  # 오류 시 1시간 후 재시도

    # ── 루프 6: 매크로 데이터 갱신 ─────────────────────────────────

    async def _macro_refresh_loop(self) -> None:
        """매크로 경제 데이터를 주기적으로 갱신한다 (30분 주기)."""
        # [버그3] _halt_event 기반 조건
        while not self._halt_event.is_set():
            try:
                await self._sleep_interruptible(1800)  # 30분마다

                self._macro_snapshot = await self._macro_collector.get_snapshot()
                logger.info(f"[매크로] 갱신 완료: {self._macro_snapshot.summary()}")

                # 이벤트 블랙아웃 시간 자동 갱신
                blackout_times = self._macro_collector.get_event_blackout_times()
                if blackout_times:
                    logger.info(f"[매크로] 이벤트 블랙아웃 시간: {blackout_times}")

            except asyncio.CancelledError:
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"매크로 데이터 갱신 실패 (네트워크): {e}")
                await self._sleep_interruptible(300)
            except (KeyError, ValueError) as e:
                logger.warning(f"매크로 데이터 갱신 실패 (파싱): {e}")
                await self._sleep_interruptible(300)

    # ── [고도화] 루프 7: 뉴스 감시 ─────────────────────────────────

    async def _news_watch_loop(self) -> None:
        """암호화폐/지정학 뉴스 실시간 감시 (HIGH → 부분 축소, CRITICAL → 전량 청산)."""
        nw = self._params.news_watch
        # [버그3] _halt_event 기반 조건
        while not self._halt_event.is_set():
            try:
                await self._sleep_interruptible(nw.poll_interval_sec)

                snap = await self._news_collector.fetch_snapshot()
                self._news_snapshot = snap
                if snap.total_items > 0:
                    logger.info(f"[뉴스] {snap.summary()}")

                # 신규 CRITICAL/HIGH 헤드라인은 알림으로 별도 푸시
                fresh = self._news_collector.get_new_critical_items(snap)
                for item in fresh[:3]:  # 최대 3건까지만
                    await self._notifier.on_alert(
                        "WARNING",
                        (
                            f"<b>[뉴스 위험]</b> score={item.risk_score} "
                            f"[{item.source}] {item.title[:120]}"
                        ),
                    )

                if snap.level == "CRITICAL":
                    if nw.close_all_on_critical:
                        await self._force_close_all(
                            reason=f"NEWS_CRITICAL(score={snap.aggregate_score})",
                        )
                    self._pause_entries(
                        minutes=max(60, nw.poll_interval_sec // 60 * 4),
                        reason=f"뉴스 CRITICAL — {snap.top_items[0].title[:60] if snap.top_items else ''}",
                    )
                elif snap.level == "HIGH":
                    if nw.reduce_on_high_ratio > 0:
                        await self._force_reduce_all(
                            ratio=nw.reduce_on_high_ratio,
                            reason=f"NEWS_HIGH(score={snap.aggregate_score})",
                        )
                    if nw.pause_entries_on_high:
                        self._pause_entries(
                            minutes=30,
                            reason=f"뉴스 HIGH — {snap.top_items[0].title[:60] if snap.top_items else ''}",
                        )

            except asyncio.CancelledError:
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"뉴스 감시 루프 네트워크 오류: {e}")
                await self._sleep_interruptible(60)
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"뉴스 감시 루프 데이터 오류: {e}")
                await self._sleep_interruptible(60)

    # ── [고도화] 루프 8: 변동성 스파이크 감시 ──────────────────────

    async def _volatility_spike_loop(self) -> None:
        """모든 심볼의 최단 타임프레임 캔들에서 변동성 급증을 감지해 즉시 청산."""
        vs = self._params.vol_spike
        check_interval = max(15, self._s.trading.position_monitor_interval_sec)
        # [버그3] _halt_event 기반 조건
        while not self._halt_event.is_set():
            try:
                await self._sleep_interruptible(check_interval)

                for symbol in self._market.symbols:
                    if not self._market.is_ready(symbol=symbol):
                        continue
                    try:
                        df = self._market.get_candles(vs.timeframe, symbol=symbol)
                    except Exception:
                        continue

                    result = self._spike_detector.detect(df)
                    if not result.is_spike:
                        continue

                    # ── 중복 트리거 방지: 같은 캔들에서 반복 감지 시 스킵 ──
                    latest_candle_ts = df.index[-1] if not df.empty else None
                    last_ts = self._last_spike_candle_ts.get(symbol)
                    if latest_candle_ts is not None and latest_candle_ts == last_ts:
                        # 이미 이 캔들에서 스파이크 처리 완료 — 중복 스킵
                        continue
                    self._last_spike_candle_ts[symbol] = latest_candle_ts

                    sym_tag = symbol.split("/")[0]
                    logger.warning(f"[{sym_tag}] 변동성 스파이크: {result.summary()}")
                    await self._notifier.on_alert(
                        "WARNING",
                        (
                            f"<b>[변동성 스파이크]</b> {sym_tag} {result.direction} "
                            f"| {result.reason}"
                        ),
                    )
                    await self._db.log_event(BotEvent(
                        event_type="VOL_SPIKE",
                        level="WARNING",
                        message=f"{sym_tag} spike {result.direction}: {result.reason}",
                    ))

                    if vs.close_all_on_spike:
                        await self._force_close_all(
                            reason=f"VOL_SPIKE({sym_tag}:{result.direction})",
                            only_symbol=None,  # 안전을 위해 전체 청산
                        )
                    self._pause_entries(
                        minutes=vs.pause_minutes_after_spike,
                        reason=f"스파이크 후 쿨다운 ({sym_tag})",
                    )
                    # 한 틱에 여러 심볼이 스파이크여도 한 번만 전체 청산 처리
                    break

            except asyncio.CancelledError:
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"스파이크 감시 네트워크 오류: {e}")
                await self._sleep_interruptible(30)
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(f"스파이크 감시 데이터 오류: {e}")
                await self._sleep_interruptible(30)

    # ── [고도화] 루프 9: 예정 이벤트 선제 대응 ─────────────────────

    async def _event_action_loop(self) -> None:
        """예정된 고영향 경제 이벤트 N시간 전부터 단계적으로 포지션을 축소/청산."""
        ea = self._params.event_action
        # [버그3] _halt_event 기반 조건
        while not self._halt_event.is_set():
            try:
                await self._sleep_interruptible(300)  # 5분마다 체크

                snap = self._macro_snapshot
                if snap is None or not snap.events:
                    continue

                now = datetime.now(timezone.utc)
                for ev in snap.events:
                    if not getattr(ev, "is_high_impact", False):
                        continue
                    delta_h = (ev.datetime_utc - now).total_seconds() / 3600
                    if delta_h <= 0:
                        # 이벤트 경과 — 재개 시각 설정
                        after = ev.datetime_utc + timedelta(minutes=ea.resume_minutes_after)
                        if now < after:
                            self._pause_entries_until(after, f"이벤트 후 대기 ({ev.title})")
                        continue

                    key = f"{ev.title}|{ev.datetime_utc.isoformat()}"
                    applied = self._event_actions_applied.get(key)

                    # 이벤트 30분 전: 전량 청산
                    if delta_h <= ea.pre_event_close_hours and applied != "CLOSED":
                        await self._force_close_all(reason=f"PRE_EVENT({ev.title})")
                        self._event_actions_applied[key] = "CLOSED"
                        self._pause_entries(
                            minutes=int(ea.pre_event_close_hours * 60) + ea.resume_minutes_after,
                            reason=f"이벤트 임박: {ev.title}",
                        )
                    # 이벤트 2시간 전: 부분 축소 (1회)
                    elif delta_h <= ea.pre_event_reduce_hours and applied is None:
                        await self._force_reduce_all(
                            ratio=ea.pre_event_reduce_ratio,
                            reason=f"PRE_EVENT_REDUCE({ev.title})",
                        )
                        self._event_actions_applied[key] = "REDUCED"
                        await self._notifier.on_alert(
                            "INFO",
                            f"<b>이벤트 선제 대응</b> — {ev.title} {delta_h:.1f}h 남음, 포지션 {int(ea.pre_event_reduce_ratio*100)}% 축소",
                        )

            except asyncio.CancelledError:
                break
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"이벤트 대응 루프 네트워크 오류: {e}")
                await self._sleep_interruptible(120)
            except (KeyError, ValueError, TypeError, AttributeError) as e:
                logger.warning(f"이벤트 대응 루프 데이터 오류: {e}")
                await self._sleep_interruptible(120)

    # ── [고도화] 공통 헬퍼: 강제 축소 / 청산 / 진입 일시정지 ──────

    async def _force_close_all(
        self,
        reason: str,
        only_symbol: Optional[str] = None,
    ) -> None:
        """열린 모든 포지션(또는 지정 심볼)에 대해 전량 강제 청산."""
        open_trades = await self._db.fetch_open_trades()
        if only_symbol:
            open_trades = [t for t in open_trades if t.symbol == only_symbol]
        if not open_trades:
            return
        logger.warning(f"강제 청산 시작: {len(open_trades)}건 (이유: {reason})")
        for trade in open_trades:
            try:
                current_price = await self._market.get_current_price(symbol=trade.symbol)
                await self._lifecycle.emergency_close(trade, current_price, reason)
                await push_sse_event("trade", {
                    "action": "closed",
                    "trade_id": trade.trade_id,
                    "symbol": trade.symbol,
                    "reason": reason,
                })
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"강제 청산 실패 ({trade.trade_id}) 네트워크: {e}")
            except (KeyError, ValueError) as e:
                logger.error(f"강제 청산 실패 ({trade.trade_id}) 데이터: {e}")
        await self._notifier.on_alert(
            "WARNING",
            f"<b>강제 청산 완료</b> — {len(open_trades)}건 | 이유: {reason}",
        )

    async def _force_reduce_all(self, ratio: float, reason: str) -> None:
        """모든 열린 포지션을 ratio 비율만큼 부분 청산."""
        open_trades = await self._db.fetch_open_trades()
        if not open_trades:
            return
        logger.info(f"강제 축소({ratio:.0%}) 시작: {len(open_trades)}건 (이유: {reason})")
        for trade in open_trades:
            try:
                current_price = await self._market.get_current_price(symbol=trade.symbol)
                await self._lifecycle.reduce_position(trade, current_price, ratio=ratio, reason=reason)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"축소 실패 ({trade.trade_id}) 네트워크: {e}")
            except (KeyError, ValueError) as e:
                logger.error(f"축소 실패 ({trade.trade_id}) 데이터: {e}")

    def _pause_entries(self, minutes: int, reason: str) -> None:
        """신규 진입을 N분 동안 차단한다."""
        if minutes <= 0:
            return
        target = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        self._pause_entries_until(target, reason)

    def _pause_entries_until(self, until: datetime, reason: str) -> None:
        already_paused = self._pause_until is not None and self._pause_until > datetime.now(timezone.utc)
        if self._pause_until is None or until > self._pause_until:
            self._pause_until = until
        if not self._state.is_paused and not self._state.is_halted:
            self._state.pause_for_event(reason)
        # 이미 해당 이유로 pause 중이면 로그 중복 출력 방지
        if not already_paused:
            logger.info(f"신규 진입 일시정지: {until.strftime('%Y-%m-%d %H:%M UTC')} ({reason})")

    # ── [Watchdog] 시그널 루프 생존 감시 ─────────────────────────────

    async def _watchdog_loop(self) -> None:
        """시그널 루프가 WATCHDOG_TIMEOUT분 이상 실행되지 않으면 텔레그램 경고 발송."""
        WATCHDOG_TIMEOUT_MINUTES = 10
        CHECK_INTERVAL_SEC = 60  # 1분마다 체크

        # [버그3] _halt_event 기반 조건
        while not self._halt_event.is_set():
            try:
                await self._sleep_interruptible(CHECK_INTERVAL_SEC)
                if self._halt_event.is_set():
                    break

                if self._last_signal_check is None:
                    # 봇 시작 직후 — 아직 첫 체크 전
                    continue

                elapsed_min = (
                    datetime.now(timezone.utc) - self._last_signal_check
                ).total_seconds() / 60

                if elapsed_min >= WATCHDOG_TIMEOUT_MINUTES:
                    if not self._watchdog_alerted:
                        self._watchdog_alerted = True
                        msg = (
                            f"⚠️ <b>[Watchdog 경고]</b>\n"
                            f"시그널 루프가 <b>{elapsed_min:.0f}분</b> 동안 응답 없음.\n"
                            f"봇 상태: {self._state.state.name}\n"
                            f"마지막 체크: {self._last_signal_check.strftime('%H:%M:%S UTC')}"
                        )
                        logger.warning(f"[Watchdog] 시그널 루프 무응답 {elapsed_min:.0f}분")
                        await self._notifier.on_alert("WARNING", msg)
                else:
                    # 정상 복구 후 재알림 준비
                    if self._watchdog_alerted:
                        self._watchdog_alerted = False
                        await self._notifier.on_alert(
                            "INFO",
                            f"✅ <b>[Watchdog 복구]</b> 시그널 루프 정상 동작 재개",
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Watchdog] 루프 오류: {e}")

    # ── [버그3] HALT 인터럽트 가능한 sleep 헬퍼 ─────────────────────

    async def _sleep_interruptible(self, seconds: float) -> None:
        """_halt_event가 set되면 sleep을 즉시 중단하고 반환한다.

        일반 asyncio.sleep 대신 이 함수를 사용하면 HALT 신호 수신 시
        최대 수십ms 내로 루프를 탈출할 수 있다.
        """
        try:
            await asyncio.wait_for(self._halt_event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass  # 정상적으로 시간이 경과한 경우

    async def _maybe_send_loop_error_alert(self, loop_name: str, error_msg: str) -> None:
        """루프에서 에러가 반복될 때 텔레그램 알림을 발송한다.

        동일 루프에서 _loop_error_alert_threshold번 연속 에러가 발생하면 첫 알림을 보내고,
        이후 _loop_error_alert_cooldown_sec 초가 지나기 전까지 중복 알림을 억제한다.
        """
        count = self._loop_error_counts.get(loop_name, 0) + 1
        self._loop_error_counts[loop_name] = count

        if count < self._loop_error_alert_threshold:
            return

        now = datetime.now(timezone.utc)
        last_alert = self._loop_error_last_alert.get(loop_name)
        if last_alert is not None:
            elapsed = (now - last_alert).total_seconds()
            if elapsed < self._loop_error_alert_cooldown_sec:
                return  # 쿨다운 중

        self._loop_error_last_alert[loop_name] = now
        msg = (
            f"🔴 <b>[루프 에러 반복]</b> <code>{loop_name}</code>\n"
            f"연속 {count}회 에러 발생\n"
            f"최근 오류: {error_msg[:200]}\n"
            f"시각: {now.strftime('%Y-%m-%d %H:%M UTC')}"
        )
        logger.error(f"[{loop_name}] 연속 {count}회 에러 — 텔레그램 알림 발송")
        try:
            await self._notifier.on_alert("ERROR", msg)
        except Exception as e:
            logger.error(f"루프 에러 알림 전송 실패: {e}")

    # ── [버그1] 시작 시 포지션 Reconciliation ────────────────────────

    async def _reconcile_positions(self) -> None:
        """봇 시작 시 OKX 실제 포지션과 DB를 비교해 불일치를 감지·복구한다.

        - DB에 없는데 거래소에 있음 → DB에 orphan 포지션으로 추가
        - DB에 있는데 거래소에 없음 → DB에서 CLOSED 처리 (유령 포지션)
        - 불일치 발생 시 텔레그램 알림 발송
        """
        logger.info("[Reconcile] 포지션 정합성 확인 시작...")
        symbols = self._s.trading.symbol_list
        discrepancies: list[str] = []

        try:
            db_open_trades = await self._db.fetch_open_trades()
            # DB 포지션 인덱스: (symbol, direction) → TradeRecord
            db_index: dict[tuple[str, str], "TradeRecord"] = {
                (t.symbol, t.direction): t for t in db_open_trades
            }

            # 거래소에서 전체 포지션 조회
            exchange_positions: dict[tuple[str, str], dict] = {}
            for symbol in symbols:
                try:
                    okx_pos_list = await self._orders._rest.fetch_positions(symbol)
                    for pos in okx_pos_list:
                        side = pos.get("side", "")
                        direction = "LONG" if side == "long" else "SHORT" if side == "short" else None
                        if direction and float(pos.get("contracts", 0) or 0) > 0:
                            exchange_positions[(symbol, direction)] = pos
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"[Reconcile] {symbol} 포지션 조회 실패 (스킵): {e}")

            # 1) DB에 있는데 거래소에 없음 → 유령 포지션 → DB closed 처리
            for (symbol, direction), trade in db_index.items():
                if (symbol, direction) not in exchange_positions:
                    sym_tag = symbol.split("/")[0]
                    msg = (
                        f"[{sym_tag}] 유령 포지션 감지: DB에 OPEN이지만 거래소에 없음 "
                        f"({trade.trade_id} {direction}) → DB를 CLOSED로 마감"
                    )
                    logger.warning(msg)
                    discrepancies.append(f"🔴 유령포지션 마감: {sym_tag} {direction} ({trade.trade_id})")
                    trade.status = "CLOSED"
                    trade.exit_time = datetime.now(timezone.utc).isoformat()
                    trade.exit_reason = "RECONCILE_GHOST"
                    await self._db.update_trade(trade)

            # 2) 거래소에 있는데 DB에 없음 → 고아 포지션 → DB에 추가
            from database.models import TradeRecord as TR
            for (symbol, direction), pos in exchange_positions.items():
                if (symbol, direction) not in db_index:
                    sym_tag = symbol.split("/")[0]
                    contracts = float(pos.get("contracts", 0) or 0)
                    avg_price = float(pos.get("entryPrice") or pos.get("info", {}).get("avgPx") or 0)
                    msg = (
                        f"[{sym_tag}] 고아 포지션 감지: 거래소에 있는데 DB에 없음 "
                        f"({direction} qty={contracts:.4f} entry=${avg_price:,.2f}) → DB에 복구"
                    )
                    logger.warning(msg)
                    discrepancies.append(
                        f"🟡 고아포지션 복구: {sym_tag} {direction} "
                        f"qty={contracts:.4f} @ ${avg_price:,.2f}"
                    )
                    orphan_trade = TR(
                        symbol=symbol,
                        direction=direction,
                        quantity=contracts,
                        leverage=self._s.trading.leverage,
                        entry_price=avg_price,
                        entry_time=datetime.now(timezone.utc).isoformat(),
                        signal_confidence=0.0,
                        atr_at_entry=0.0,
                    )
                    await self._db.insert_trade(orphan_trade)

            if discrepancies:
                summary = "\n".join(discrepancies)
                logger.warning(f"[Reconcile] 불일치 {len(discrepancies)}건 발견 및 처리 완료")
                await self._notifier.on_alert(
                    "WARNING",
                    (
                        f"⚠️ <b>[포지션 불일치 감지]</b> — 봇 시작 시 복구 완료\n"
                        f"{summary}\n"
                        f"<i>이전 세션에서 청산 실패 또는 외부 개입이 있었을 수 있습니다.</i>"
                    ),
                )
                await self._db.log_event(BotEvent(
                    event_type="RECONCILE",
                    level="WARNING",
                    message=f"포지션 불일치 {len(discrepancies)}건 복구",
                    metadata={"items": discrepancies},
                ))
            else:
                logger.info("[Reconcile] 포지션 정합성 확인 완료 — 불일치 없음")

        except Exception as e:
            logger.error(f"[Reconcile] 포지션 정합성 확인 중 오류 (봇 시작 계속): {e}")

    # ── 긴급 정지 ────────────────────────────────────────────────

    async def _halt(self, reason: str) -> None:
        """봇 긴급 정지. 모든 포지션을 강제 청산한다."""
        logger.critical(f"봇 긴급 정지: {reason}")
        self._state.halt(reason)
        # [버그3] HALT 이벤트 즉시 발동 — 모든 루프가 다음 await에서 바로 탈출한다
        self._halt_event.set()
        await push_sse_event("state", {"state": "HALTED", "reason": reason})
        await self._notifier.on_halt(reason)

        # 열린 포지션 강제 청산
        open_trades = await self._db.fetch_open_trades()
        if open_trades:
            logger.critical(f"긴급 청산: {len(open_trades)}개 포지션")
            for trade in open_trades:
                try:
                    current_price = await self._market.get_current_price(symbol=trade.symbol)
                    await self._lifecycle._close(trade, current_price, "HALT")
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.error(f"긴급 청산 실패 ({trade.trade_id}) 네트워크: {e}")
                except (KeyError, ValueError) as e:
                    logger.error(f"긴급 청산 실패 ({trade.trade_id}) 데이터: {e}")

        await self._db.log_event(BotEvent(
            event_type="HALT",
            level="CRITICAL",
            message=f"봇 정지: {reason}",
        ))
