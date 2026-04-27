"""
OKX BTC/USDT 자동매매봇 진입점.

사용법:
  python main.py              -- 봇 실행 (기본: 페이퍼 트레이딩)
  python main.py --mode test  -- API 연결 및 텔레그램 테스트만 수행
  python main.py --backtest --from 2024-01-01 --to 2024-12-31
"""
import argparse
import asyncio
import sys
from pathlib import Path

import uvloop
from loguru import logger

# 로거 설정 (파일 + 콘솔)
def _setup_logger(log_level: str = "INFO") -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    logger.add(
        "logs/bot_{time:YYYY-MM-DD}.log",
        level="DEBUG",
        rotation="00:00",
        retention="30 days",
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )


async def run_bot() -> None:
    """봇 메인 실행"""
    from config.settings import load_settings
    from config.strategy_params import AllStrategyParams
    from data.candle_store import CandleStore
    from data.market_data import MarketData
    from data.okx_rest_client import OKXRestClient
    from data.okx_ws_client import OKXWebSocketClient
    from database.db_manager import DatabaseManager
    from execution.order_manager import OrderManager
    from execution.position_tracker import PositionTracker
    from execution.trade_lifecycle import TradeLifecycle
    from notification.notification_manager import NotificationManager
    from risk.risk_manager import RiskManager
    from risk.stop_manager import StopManager
    from strategy.signal_aggregator import SignalAggregator
    from core.bot_engine import BotEngine
    from dashboard.server import app as dashboard_app, init_dashboard

    settings = load_settings()
    _setup_logger(settings.log_level)

    if not settings.okx.is_configured:
        logger.error("OKX API 키가 설정되지 않음. .env 파일을 확인하세요.")
        return

    logger.info(f"설정 로딩 완료 — 모드: {'페이퍼' if settings.okx.is_demo else '실거래'}")
    logger.info(f"거래 심볼: {', '.join(settings.trading.symbol_list)}")

    # DB 초기화
    db = await DatabaseManager.create(settings.db_path)

    # REST 클라이언트
    rest = OKXRestClient(settings.okx)

    # WebSocket 클라이언트 (선택적)
    ws = OKXWebSocketClient(
        api_key=settings.okx.api_key,
        secret_key=settings.okx.secret_key,
        passphrase=settings.okx.passphrase,
        is_demo=settings.okx.is_demo,
    )

    # 데이터 레이어
    store = CandleStore(db)
    market = MarketData(rest, store, settings.trading)

    # 전략
    params = AllStrategyParams.from_yaml()
    aggregator = SignalAggregator(params)

    # 리스크 + 목표 추적
    notifier_pre = NotificationManager(settings)  # GoalTracker 주입용 (아래에서 재사용)
    from risk.goal_tracker import GoalTracker
    goal_tracker = GoalTracker(db, settings.goal, notifier_pre)
    risk_mgr = RiskManager(settings.trading, settings.risk, db, goal_tracker=goal_tracker)
    stop_mgr = StopManager(params.risk)

    # 실행
    order_mgr = OrderManager(rest, settings.trading)
    tracker = PositionTracker(rest, db, settings.trading)
    notifier = notifier_pre  # 위에서 이미 생성
    lifecycle = TradeLifecycle(order_mgr, stop_mgr, db, notifier, settings.trading, risk_params=params.risk)

    # 봇 엔진 (params 공유로 뉴스/스파이크/이벤트/자동복구 활성화)
    engine = BotEngine(
        settings=settings,
        market=market,
        aggregator=aggregator,
        risk_mgr=risk_mgr,
        order_mgr=order_mgr,
        stop_mgr=stop_mgr,
        position_tracker=tracker,
        lifecycle=lifecycle,
        notifier=notifier,
        db=db,
        params=params,
    )

    # 대시보드 초기화
    init_dashboard(db, engine)

    # 봇 + 대시보드 동시 실행
    import uvicorn

    async def run_dashboard():
        config = uvicorn.Config(
            dashboard_app,
            host=settings.dashboard_host,
            port=settings.dashboard_port,
            log_level="warning",
        )
        server = uvicorn.Server(config)
        await server.serve()

    try:
        await asyncio.gather(
            engine.start(),
            run_dashboard(),
        )
    finally:
        await engine.stop()
        await rest.close()
        await db.close()


async def run_test() -> None:
    """API 연결 테스트 및 텔레그램 핑"""
    from config.settings import load_settings
    from data.okx_rest_client import OKXRestClient
    from notification.telegram_notifier import TelegramNotifier

    settings = load_settings()
    _setup_logger(settings.log_level)

    logger.info("=== 연결 테스트 모드 ===")

    # OKX API 연결 테스트
    if settings.okx.is_configured:
        rest = OKXRestClient(settings.okx)
        try:
            ticker = await rest.fetch_ticker(settings.trading.symbol)
            price = ticker.get("last", 0)
            logger.info(f"OKX API 연결 성공: BTC/USDT = ${price:,.2f}")

            balance = await rest.fetch_balance()
            logger.info(f"잔고 조회: {balance}")

            # 캔들 조회 테스트
            df = await rest.fetch_ohlcv(settings.trading.symbol, "1h", limit=10)
            logger.info(f"캔들 조회 성공: {len(df)}개 (1H)")

        except (KeyError, ValueError, ConnectionError) as e:
            logger.error(f"OKX API 오류: {e}")
        finally:
            await rest.close()
    else:
        logger.warning("OKX API 키 미설정 — API 테스트 스킵")

    # 텔레그램 테스트
    if settings.telegram_configured:
        notifier = TelegramNotifier(settings.telegram_bot_token, settings.telegram_chat_id)
        success = await notifier.send_test_ping()
        if success:
            logger.info("텔레그램 연결 성공")
        else:
            logger.error("텔레그램 연결 실패")
    else:
        logger.warning("텔레그램 미설정 — 테스트 스킵")

    logger.info("=== 테스트 완료 ===")


async def run_backtest(from_date: str, to_date: str) -> None:
    """백테스트 실행"""
    from config.settings import load_settings
    from config.strategy_params import AllStrategyParams
    from data.okx_rest_client import OKXRestClient
    from backtest.data_loader import HistoricalDataLoader
    from backtest.backtester import Backtester

    settings = load_settings()
    _setup_logger("INFO")

    logger.info(f"=== 백테스트 모드: {from_date} ~ {to_date} ===")

    rest = OKXRestClient(settings.okx)
    loader = HistoricalDataLoader(rest)

    symbol = settings.trading.symbol
    try:
        logger.info("과거 데이터 수집 중... (시간이 걸릴 수 있음)")
        df_4h = await loader.fetch_full_history(symbol, "4h", from_date, to_date)
        df_1h = await loader.fetch_full_history(symbol, "1h", from_date, to_date)
        df_15m = await loader.fetch_full_history(symbol, "15m", from_date, to_date)

        params = AllStrategyParams.from_yaml()
        bt = Backtester(
            params,
            initial_balance=1000.0,
            leverage=settings.trading.leverage,
        )
        report = bt.run(df_4h, df_1h, df_15m)

        print("\n" + "=" * 60)
        print("백테스트 결과")
        print("=" * 60)
        print(f"총 거래: {report.total_trades}회")
        print(f"승률:    {report.win_rate:.1%}")
        print(f"Profit Factor: {report.profit_factor:.2f}")
        print(f"Sharpe Ratio:  {report.sharpe_ratio:.2f}")
        print(f"최대 드로다운: {report.max_drawdown_pct:.1%}")
        print(f"총 P&L: {report.total_pnl_usdt:+.2f} USDT")
        print(f"최고 거래: +{report.best_trade_usdt:.2f} USDT")
        print(f"최악 거래: {report.worst_trade_usdt:.2f} USDT")
        print(f"기준 충족 여부: {'YES ✓' if report.is_acceptable() else 'NO ✗'}")
        print("=" * 60)

    finally:
        await rest.close()


def main():
    parser = argparse.ArgumentParser(description="OKX 자동매매봇")
    parser.add_argument("--mode", choices=["live", "test"], default="live", help="실행 모드")
    parser.add_argument("--backtest", action="store_true", help="백테스트 실행")
    parser.add_argument("--from", dest="from_date", default="2024-01-01")
    parser.add_argument("--to", dest="to_date", default="2024-12-31")
    args = parser.parse_args()

    # uvloop 설치 시 성능 향상
    try:
        uvloop.install()
    except (ImportError, AttributeError):
        pass

    if args.backtest:
        asyncio.run(run_backtest(args.from_date, args.to_date))
    elif args.mode == "test":
        asyncio.run(run_test())
    else:
        asyncio.run(run_bot())


if __name__ == "__main__":
    main()
