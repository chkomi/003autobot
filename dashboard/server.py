"""
FastAPI 웹 대시보드.
봇 상태, 차트 데이터, 시그널 분석, 거래 기록을 실시간으로 조회한다.
SSE(Server-Sent Events)로 시그널/거래/상태를 실시간 푸시한다.
"""
import asyncio
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from loguru import logger
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from backtest.performance import PerformanceAnalyzer
from config.strategy_params import AllStrategyParams
from database.db_manager import DatabaseManager
from strategy.indicators import (
    calc_atr, calc_bollinger, calc_ema, calc_macd, calc_rsi,
    calc_supertrend, calc_volume_sma,
)
from strategy.trend_filter import TrendFilter
from strategy.momentum_trigger import MomentumTrigger
from strategy.micro_confirmation import MicroConfirmation

app = FastAPI(title="OKX 자동매매봇 대시보드", version="2.0.0")

_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

_db: DatabaseManager = None
_bot_engine = None
_sse_clients: list[asyncio.Queue] = []


def init_dashboard(db: DatabaseManager, engine=None):
    global _db, _bot_engine
    _db = db
    _bot_engine = engine


async def push_sse_event(event_type: str, data: dict) -> None:
    """SSE 클라이언트들에게 이벤트를 푸시한다."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    dead: list[asyncio.Queue] = []
    for q in _sse_clients:
        try:
            q.put_nowait((event_type, payload))
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _sse_clients.remove(q)


def _safe(v):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return round(float(v), 2)


def _ts(idx):
    return int(pd.Timestamp(idx).timestamp())


# ── 페이지 ────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = _static_dir / "index.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text(encoding="utf-8"))
    logger.error(f"대시보드 HTML 파일 없음: {html_file}")
    return HTMLResponse("<h1>대시보드를 로딩할 수 없습니다</h1>")


# ── 차트 데이터 API ──────────────────────────────────────────

@app.get("/api/candles")
async def get_candles(tf: str = "1h", limit: int = 100):
    if not _bot_engine:
        return {"candles": []}
    try:
        df = _bot_engine._market.get_candles(tf, limit=limit)
        candles = []
        for ts, row in df.iterrows():
            candles.append({
                "time": _ts(ts),
                "open": _safe(row["open"]),
                "high": _safe(row["high"]),
                "low": _safe(row["low"]),
                "close": _safe(row["close"]),
                "volume": _safe(row["volume"]),
            })
        return {"candles": candles}
    except (KeyError, ValueError) as e:
        logger.warning(f"캔들 API 오류: {e}")
        return {"candles": [], "error": str(e)}


@app.get("/api/indicators")
async def get_indicators(tf: str = "1h"):
    if not _bot_engine:
        return {}
    try:
        df = _bot_engine._market.get_candles(tf)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        ema21 = calc_ema(close, 21)
        ema55 = calc_ema(close, 55)
        ema200 = calc_ema(close, 200)
        st = calc_supertrend(high, low, close)
        macd_df = calc_macd(close)
        rsi = calc_rsi(close)
        bb = calc_bollinger(close)
        atr = calc_atr(high, low, close)
        vol_sma = calc_volume_sma(volume)

        n = min(120, len(df))
        sl = slice(-n, None)

        def series_to_list(s, sl=sl):
            return [{"time": _ts(ts), "value": _safe(v)} for ts, v in s.iloc[sl].items() if _safe(v) is not None]

        macd_list = []
        for ts, row in macd_df.iloc[sl].iterrows():
            m, h, s_val = _safe(row["macd"]), _safe(row["macd_hist"]), _safe(row["macd_signal"])
            if m is not None:
                macd_list.append({"time": _ts(ts), "macd": m, "hist": h, "signal": s_val})

        bb_list = []
        for ts, row in bb.iloc[sl].iterrows():
            u, mid, lo_val = _safe(row["bb_upper"]), _safe(row["bb_mid"]), _safe(row["bb_lower"])
            if u is not None:
                bb_list.append({"time": _ts(ts), "upper": u, "mid": mid, "lower": lo_val})

        st_list = []
        for ts, row in st.iloc[sl].iterrows():
            v, d = _safe(row["supertrend"]), _safe(row["supertrend_dir"])
            if v is not None:
                st_list.append({"time": _ts(ts), "value": v, "dir": int(d) if d else 0})

        return {
            "ema21": series_to_list(ema21),
            "ema55": series_to_list(ema55),
            "ema200": series_to_list(ema200),
            "supertrend": st_list,
            "macd": macd_list,
            "rsi": series_to_list(rsi),
            "bb": bb_list,
            "atr": series_to_list(atr),
            "volume_sma": series_to_list(vol_sma),
        }
    except (KeyError, ValueError) as e:
        logger.warning(f"인디케이터 API 오류: {e}")
        return {"error": str(e)}


@app.get("/api/signal-status")
async def get_signal_status():
    if not _bot_engine:
        return {"signal": "UNKNOWN"}
    try:
        market = _bot_engine._market
        params = AllStrategyParams.from_yaml()

        price = await market.get_current_price()
        bal = await market.get_balance()

        df4 = market.get_candles(params.trend.timeframe)
        df1 = market.get_candles(params.momentum.timeframe)
        df15 = market.get_candles(params.micro.timeframe)

        trend = TrendFilter(params.trend).analyze(df4)
        mom = MomentumTrigger(params.momentum).analyze(df1)
        micro = MicroConfirmation(params.micro).analyze(df15)

        # Score engine
        score_data = {"trend": 0, "momentum": 0, "volume": 0, "volatility": 0, "sentiment": 0, "total": 0}
        try:
            from strategy.score_engine import ScoreEngine, SentimentData
            se = ScoreEngine(params.scoring, params.sentiment)
            sr = se.evaluate(df4, df1, df15, sentiment=SentimentData())
            score_data = {
                "trend": _safe(sr.trend), "momentum": _safe(sr.momentum),
                "volume": _safe(sr.volume), "volatility": _safe(sr.volatility),
                "sentiment": _safe(sr.sentiment), "total": _safe(sr.total),
            }
        except (KeyError, ValueError, TypeError) as e:
            logger.debug(f"스코어 평가 실패: {e}")

        # Layer results
        l1 = {"status": "N/A", "regime": "NEUTRAL", "ema_aligned": False, "supertrend": "N/A"}
        if trend:
            l1 = {
                "status": "PASS" if trend.regime != "NEUTRAL" else "NEUTRAL",
                "regime": trend.regime,
                "ema_aligned": trend.ema_aligned,
                "supertrend": "UP" if trend.supertrend_bullish else "DOWN",
                "ema_fast": _safe(trend.ema_fast),
                "ema_mid": _safe(trend.ema_mid),
                "ema_slow": _safe(trend.ema_slow),
            }

        l2 = {"status": "N/A", "macd_cross": "NONE", "rsi": 50, "long_trigger": False, "short_trigger": False}
        if mom:
            l2_pass = (trend and trend.is_bullish and mom.long_trigger) or (trend and trend.is_bearish and mom.short_trigger)
            l2 = {
                "status": "PASS" if l2_pass else "FAIL",
                "macd_cross": mom.macd_cross,
                "rsi": _safe(mom.rsi_value),
                "macd_hist": _safe(mom.macd_hist),
                "long_trigger": mom.long_trigger,
                "short_trigger": mom.short_trigger,
            }

        l3 = {"status": "N/A", "bb_pct": 0.5, "volume_ratio": 0, "long_confirm": False, "short_confirm": False}
        if micro:
            l3_pass = (trend and trend.is_bullish and micro.long_confirm) or (trend and trend.is_bearish and micro.short_confirm)
            vol_ratio = micro.current_volume / micro.avg_volume if micro.avg_volume > 0 else 0
            l3 = {
                "status": "PASS" if l3_pass else "FAIL",
                "bb_pct": _safe(micro.bb_pct),
                "bb_bounce": micro.bb_bounce,
                "volume_ratio": _safe(vol_ratio),
                "volume_confirmed": micro.volume_confirmed,
                "long_confirm": micro.long_confirm,
                "short_confirm": micro.short_confirm,
            }

        # Probability estimate
        total_score = score_data.get("total", 0) or 0
        regime = l1.get("regime", "NEUTRAL")
        long_prob = min(int(total_score * 1.0), 100) if regime == "BULL" else min(int(total_score * 0.3), 30)
        short_prob = min(int(total_score * 1.0), 100) if regime == "BEAR" else min(int(total_score * 0.3), 30)

        any_pass = l2.get("status") == "PASS" or l3.get("status") == "PASS"
        signal_dir = "FLAT"
        if any_pass and regime == "BULL":
            signal_dir = "LONG"
        elif any_pass and regime == "BEAR":
            signal_dir = "SHORT"

        return {
            "current_price": _safe(price),
            "balance": _safe(bal.get("total", 0)),
            "free_balance": _safe(bal.get("free", 0)),
            "regime": regime,
            "signal": signal_dir,
            "layer1": l1,
            "layer2": l2,
            "layer3": l3,
            "long_probability": long_prob,
            "short_probability": short_prob,
            "score": score_data,
            "bot_state": _bot_engine.state.value if _bot_engine else "UNKNOWN",
        }
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"시그널 상태 API 오류: {e}")
        return {"signal": "ERROR", "error": str(e)}


# ── 기존 API (유지) ──────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    state = _bot_engine.state.value if _bot_engine else "UNKNOWN"
    open_trades = []
    if _db:
        trades = await _db.fetch_open_trades()
        open_trades = [t.to_dict() for t in trades]
    return {"state": state, "open_positions": open_trades}


@app.get("/api/trades")
async def get_trades(limit: int = 50):
    if not _db:
        return {"trades": []}
    trades = await _db.fetch_closed_trades(limit=limit)
    return {"trades": [t.to_dict() for t in trades]}


@app.get("/api/performance")
async def get_performance():
    if not _db:
        return {}
    trades = await _db.fetch_closed_trades(limit=500)
    trade_dicts = [t.to_dict() for t in trades]
    analyzer = PerformanceAnalyzer(trade_dicts)
    report = analyzer.calculate()
    return {
        "total_trades": report.total_trades,
        "win_rate": report.win_rate,
        "profit_factor": report.profit_factor,
        "sharpe_ratio": report.sharpe_ratio,
        "max_drawdown_pct": report.max_drawdown_pct,
        "total_pnl_usdt": report.total_pnl_usdt,
        "expectancy_usdt": report.expectancy_usdt,
    }


@app.get("/api/daily")
async def get_daily(date: str = None):
    if not _db:
        return {}
    daily = await _db.get_daily_pnl(date)
    return {"date": daily.date, "pnl_usdt": daily.pnl_usdt, "trade_count": daily.trade_count, "win_rate": daily.win_rate}


@app.get("/api/events")
async def get_events(limit: int = 50):
    if not _db:
        return {"events": []}
    return {"events": await _db.fetch_recent_events(limit=limit)}


@app.post("/api/control/halt")
async def halt_bot(reason: str = "수동 정지"):
    if _bot_engine:
        await _bot_engine._halt(reason)
        return {"status": "halted", "reason": reason}
    return {"status": "error", "message": "봇이 초기화되지 않음"}


@app.post("/api/control/resume")
async def resume_bot():
    if _bot_engine:
        _bot_engine._state.resume()
        return {"status": "resumed"}
    return {"status": "error", "message": "봇이 초기화되지 않음"}


# ── SSE (Server-Sent Events) ────────────────────────────────

@app.get("/api/stream")
async def sse_stream():
    """SSE 스트림 — 시그널, 거래, 상태 변화를 실시간으로 푸시한다."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _sse_clients.append(queue)

    async def event_generator():
        try:
            # 초기 상태 전송
            state = _bot_engine.state.value if _bot_engine else "UNKNOWN"
            yield f"event: state\ndata: {json.dumps({'state': state})}\n\n"

            while True:
                try:
                    event_type, payload = await asyncio.wait_for(queue.get(), timeout=30)
                    yield f"event: {event_type}\ndata: {payload}\n\n"
                except asyncio.TimeoutError:
                    # 30초마다 keep-alive
                    yield ": heartbeat\n\n"
        finally:
            if queue in _sse_clients:
                _sse_clients.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
