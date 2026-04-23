"""
OKX 전 종목 백테스팅 — 1년+ 기간 (2024-01-01 ~ 2026-04-23)
결과: report/backtest_results.csv

사용법:
  cd "/Users/yun/Documents/Business/003. Autobot"
  python3 run_full_backtest.py
"""
import asyncio
import csv
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import load_settings
from config.strategy_params import AllStrategyParams
from data.okx_rest_client import OKXRestClient
from backtest.backtester import Backtester
from backtest.data_loader import HistoricalDataLoader
from backtest.performance import PerformanceAnalyzer

# ─── 백테스트 설정 ────────────────────────────────────────────
FROM_DATE   = "2024-01-01"   # 2년치 이상 (EMA200 워밍업 포함)
TO_DATE     = "2026-04-23"
INITIAL_BALANCE = 1000.0
LEVERAGE    = 3              # .env에서 읽어오지만 기본값
MAX_SYMBOLS = 100            # 거래량 상위 N개로 제한 (0이면 전체)
MIN_4H_CANDLES = 220         # EMA200 최소 필요 캔들

RESULTS_DIR = Path(__file__).parent / "report"
CACHE_DIR   = Path(__file__).parent / "backtest" / "results" / "cache_full"


def recommendation(sharpe: float, win_rate: float, profit_factor: float, max_dd: float) -> str:
    """Sharpe/승률 기반 추천 등급"""
    if sharpe >= 1.5 and win_rate >= 0.55:
        return "STRONG_BUY"
    elif sharpe >= 1.0 and win_rate >= 0.50 and profit_factor >= 1.5:
        return "BUY"
    elif sharpe >= 0.5 and win_rate >= 0.45:
        return "NEUTRAL"
    else:
        return "AVOID"


async def fetch_all_swap_symbols(rest: OKXRestClient) -> list[str]:
    """OKX USDT 무기한 선물 전종목 + 거래량 정렬"""
    try:
        markets = await rest._exchange.load_markets()
        symbols = []
        for sym, info in markets.items():
            if (
                info.get("swap")
                and info.get("active")
                and info.get("quote") == "USDT"
                and info.get("settle") == "USDT"
            ):
                symbols.append(sym)
        symbols.sort()
        logger.info(f"총 {len(symbols)}개 USDT 무기한 선물 발견")

        # 거래량 상위 정렬 (tickers 배치 조회 시도)
        if MAX_SYMBOLS > 0 and len(symbols) > MAX_SYMBOLS:
            logger.info(f"거래량 상위 {MAX_SYMBOLS}개 선택 중...")
            try:
                tickers = await rest._exchange.fetch_tickers(symbols[:300])
                vol_map = {}
                for sym, tk in tickers.items():
                    vol_map[sym] = float(tk.get("quoteVolume") or 0)
                symbols_sorted = sorted(symbols, key=lambda s: vol_map.get(s, 0), reverse=True)
                symbols = symbols_sorted[:MAX_SYMBOLS]
                logger.info(f"거래량 상위 {len(symbols)}개 선택 완료")
            except Exception as e:
                logger.warning(f"거래량 정렬 실패, 알파벳 순 사용: {e}")
                symbols = symbols[:MAX_SYMBOLS]

        return symbols
    except Exception as e:
        logger.error(f"종목 조회 실패: {e}")
        return []


async def fetch_symbol_data(
    loader: HistoricalDataLoader,
    symbol: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """4H/1H/15M OHLCV 데이터 수집 (CSV 캐시 활용)"""
    # 4H: EMA200 워밍업을 위해 시작일 60일 전부터
    from_4h = (pd.Timestamp(FROM_DATE) - timedelta(days=60)).strftime("%Y-%m-%d")
    # 1H: MACD 워밍업을 위해 5일 전부터
    from_1h = (pd.Timestamp(FROM_DATE) - timedelta(days=5)).strftime("%Y-%m-%d")

    df_4h  = await loader.fetch_or_load_csv(symbol, "4h",  from_4h, TO_DATE, CACHE_DIR)
    df_1h  = await loader.fetch_or_load_csv(symbol, "1h",  from_1h, TO_DATE, CACHE_DIR)
    df_15m = await loader.fetch_or_load_csv(symbol, "15m", FROM_DATE, TO_DATE, CACHE_DIR)

    return df_4h, df_1h, df_15m


def run_backtest(
    symbol: str,
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    params: AllStrategyParams,
    leverage: int,
) -> dict:
    """단일 종목 백테스트 실행"""
    try:
        bt = Backtester(
            params,
            initial_balance=INITIAL_BALANCE,
            leverage=leverage,
        )
        report = bt.run(df_4h, df_1h, df_15m)
        trades  = bt.get_trades()

        # 거래별 수익률 계산
        trade_returns = []
        for t in trades:
            if t.get("pnl_usdt") is not None:
                trade_returns.append(t["pnl_usdt"] / INITIAL_BALANCE * 100)

        best_trade_pct  = max(trade_returns) if trade_returns else 0.0
        worst_trade_pct = min(trade_returns) if trade_returns else 0.0

        # 연간 수익률 계산
        start_dt = pd.Timestamp(FROM_DATE)
        end_dt   = pd.Timestamp(TO_DATE)
        days     = (end_dt - start_dt).days
        years    = days / 365.0
        total_return_pct = report.total_pnl_pct * 100
        annual_return    = ((1 + report.total_pnl_pct) ** (1 / years) - 1) * 100 if years > 0 else 0.0

        rec = recommendation(
            report.sharpe_ratio,
            report.win_rate,
            report.profit_factor,
            report.max_drawdown_pct,
        )

        return {
            "symbol":               symbol,
            "status":               "OK",
            "period":               f"{FROM_DATE}~{TO_DATE}",
            "total_trades":         report.total_trades,
            "win_rate":             round(report.win_rate * 100, 2),
            "total_return":         round(total_return_pct, 2),
            "annual_return":        round(annual_return, 2),
            "max_drawdown":         round(report.max_drawdown_pct * 100, 2),
            "sharpe_ratio":         round(report.sharpe_ratio, 3),
            "profit_factor":        round(report.profit_factor, 3),
            "avg_trade_duration":   round(report.avg_trade_duration_hours, 1) if report.avg_trade_duration_hours else None,
            "best_trade":           round(best_trade_pct, 2),
            "worst_trade":          round(worst_trade_pct, 2),
            "recommendation":       rec,
        }
    except Exception as e:
        logger.warning(f"  {symbol} 백테스트 오류: {e}")
        return {
            "symbol":        symbol,
            "status":        "ERROR",
            "period":        f"{FROM_DATE}~{TO_DATE}",
            "total_trades":  0,
            "win_rate":      0,
            "total_return":  0,
            "annual_return": 0,
            "max_drawdown":  0,
            "sharpe_ratio":  0,
            "profit_factor": 0,
            "avg_trade_duration": None,
            "best_trade":    0,
            "worst_trade":   0,
            "recommendation": "AVOID",
        }


def save_csv(results: list[dict], out_path: Path) -> None:
    """결과를 CSV로 저장"""
    fieldnames = [
        "symbol", "period", "total_trades", "win_rate",
        "total_return", "annual_return", "max_drawdown",
        "sharpe_ratio", "profit_factor", "avg_trade_duration",
        "best_trade", "worst_trade", "recommendation",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"CSV 저장 완료: {out_path} ({len(results)}개 종목)")


async def main():
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )
    # 로그 파일도 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(RESULTS_DIR / "backtest_run.log", level="DEBUG", rotation="50 MB")

    settings = load_settings()
    params   = AllStrategyParams.from_yaml()
    leverage = settings.trading.leverage or LEVERAGE

    rest   = OKXRestClient(settings.okx)
    loader = HistoricalDataLoader(rest)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"OKX 전종목 백테스팅 시작")
    logger.info(f"기간: {FROM_DATE} ~ {TO_DATE}")
    logger.info(f"레버리지: {leverage}x | 초기 자본: ${INITIAL_BALANCE:,.0f}")
    logger.info(f"시그널 모드: {params.backtest.signal_mode}")
    logger.info("=" * 60)

    symbols = await fetch_all_swap_symbols(rest)
    if not symbols:
        logger.error("종목 목록을 가져올 수 없습니다.")
        await rest.close()
        return

    logger.info(f"백테스트 대상: {len(symbols)}개 종목")

    all_results = []
    start_time  = time.time()

    for i, symbol in enumerate(symbols, 1):
        elapsed = time.time() - start_time
        eta     = (elapsed / i) * (len(symbols) - i) if i > 1 else 0
        logger.info(
            f"[{i:>3}/{len(symbols)}] {symbol:<25} "
            f"경과: {elapsed/60:.1f}분 | 예상 남은시간: {eta/60:.1f}분"
        )

        # 데이터 수집
        try:
            df_4h, df_1h, df_15m = await fetch_symbol_data(loader, symbol)
        except Exception as e:
            logger.warning(f"  데이터 수집 실패: {e}")
            all_results.append({
                "symbol": symbol, "status": "ERROR", "period": f"{FROM_DATE}~{TO_DATE}",
                "total_trades": 0, "win_rate": 0, "total_return": 0, "annual_return": 0,
                "max_drawdown": 0, "sharpe_ratio": 0, "profit_factor": 0,
                "avg_trade_duration": None, "best_trade": 0, "worst_trade": 0,
                "recommendation": "AVOID",
            })
            await asyncio.sleep(1.0)
            continue

        # 데이터 부족 체크
        if len(df_4h) < MIN_4H_CANDLES:
            logger.debug(f"  4H 캔들 부족 ({len(df_4h)}개) — 스킵")
            all_results.append({
                "symbol": symbol, "status": "SKIP", "period": f"{FROM_DATE}~{TO_DATE}",
                "total_trades": 0, "win_rate": 0, "total_return": 0, "annual_return": 0,
                "max_drawdown": 0, "sharpe_ratio": 0, "profit_factor": 0,
                "avg_trade_duration": None, "best_trade": 0, "worst_trade": 0,
                "recommendation": "AVOID",
            })
            continue

        # 백테스트 실행
        result = run_backtest(symbol, df_4h, df_1h, df_15m, params, leverage)
        all_results.append(result)

        if result["total_trades"] > 0:
            logger.info(
                f"  → 거래 {result['total_trades']:>3}회 | "
                f"승률 {result['win_rate']:>5.1f}% | "
                f"Sharpe {result['sharpe_ratio']:>5.2f} | "
                f"연수익 {result['annual_return']:>+6.1f}% | "
                f"[{result['recommendation']}]"
            )
        else:
            logger.info(f"  → 시그널 없음")

        # API rate limit 고려
        await asyncio.sleep(0.5)

    await rest.close()

    # CSV 저장
    csv_path = RESULTS_DIR / "backtest_results.csv"
    # OK 결과만 저장 (ERROR/SKIP도 포함, status 컬럼은 CSV에서 제외)
    save_csv(all_results, csv_path)

    # 요약 출력
    ok_results = [r for r in all_results if r.get("total_trades", 0) > 0]
    ok_results.sort(key=lambda r: r.get("sharpe_ratio", 0), reverse=True)

    total_elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"백테스팅 완료: {len(all_results)}종목 처리 | {total_elapsed/60:.1f}분 소요")
    logger.info(f"거래 발생: {len(ok_results)}종목")
    strong_buy = [r for r in ok_results if r.get("recommendation") == "STRONG_BUY"]
    buy        = [r for r in ok_results if r.get("recommendation") == "BUY"]
    logger.info(f"STRONG_BUY: {len(strong_buy)}종목 | BUY: {len(buy)}종목")
    logger.info(f"CSV 저장: {csv_path}")
    logger.info("=" * 60)

    # TOP 20 출력
    logger.info("\n📊 상위 20개 종목 (Sharpe Ratio 기준)")
    logger.info(f"{'#':>3} {'종목':<25} {'거래':>5} {'승률':>7} {'Sharpe':>7} {'연수익':>8} {'추천':<12}")
    logger.info("-" * 75)
    for i, r in enumerate(ok_results[:20], 1):
        logger.info(
            f"{i:>3} {r['symbol']:<25} {r['total_trades']:>5} "
            f"{r['win_rate']:>5.1f}% {r['sharpe_ratio']:>7.2f} "
            f"{r['annual_return']:>+6.1f}%  [{r['recommendation']}]"
        )

    return ok_results[:20]


if __name__ == "__main__":
    asyncio.run(main())
