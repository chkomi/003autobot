"""
OKX 전 종목 백테스팅 실행기.
모든 USDT 무기한 선물 종목에 대해 Triple Confirmation 전략을 백테스트하고
종목별 보고서 + 종합 요약을 backtest/ 폴더에 저장한다.

개선사항:
- HistoricalDataLoader로 6.5개월 전체 데이터 수집 (CSV 캐시)
- 시그널 조건 완화 (AND→OR, 완화 RSI/BB/볼륨)
- 워크포워드 윈도우 축소 (45/15일)
- OOS 3단계 프로그레시브 검증
- 파라미터 민감도 분석

사용법: cd "003. Autobot" && python3 -m backtest.run_all_symbols
"""
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import load_settings
from config.strategy_params import AllStrategyParams
from data.okx_rest_client import OKXRestClient
from backtest.backtester import Backtester
from backtest.data_loader import HistoricalDataLoader
from backtest.performance import PerformanceAnalyzer
from backtest.walk_forward import WalkForwardValidator
from backtest.oos_validator import OutOfSampleValidator
from backtest.sensitivity import SensitivityAnalyzer

RESULTS_DIR = Path(__file__).parent / "results"
CACHE_DIR = RESULTS_DIR / "cache"
FROM_DATE = "2025-10-01"
TO_DATE = "2026-04-15"
INITIAL_BALANCE = 1000.0
MIN_CANDLES_4H = 220

# 제외 종목 (유동성 부족 등)
EXCLUDE_SYMBOLS = set()


async def fetch_all_swap_symbols(rest: OKXRestClient) -> list[str]:
    """OKX에서 USDT 무기한 선물 전 종목 목록을 가져온다."""
    try:
        markets = await rest._exchange.load_markets()
        symbols = []
        for sym, info in markets.items():
            if (
                info.get("swap")
                and info.get("active")
                and info.get("quote") == "USDT"
                and info.get("settle") == "USDT"
                and sym not in EXCLUDE_SYMBOLS
            ):
                symbols.append(sym)
        symbols.sort()
        return symbols
    except (KeyError, ValueError, AttributeError) as e:
        logger.error(f"종목 목록 조회 실패: {e}")
        return []


async def fetch_symbol_data(
    loader: HistoricalDataLoader,
    symbol: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """종목의 4H/1H/15M 데이터를 수집한다 (CSV 캐시 활용).

    4H: FROM_DATE - 60일부터 (EMA200 워밍업)
    1H: FROM_DATE - 5일부터 (MACD 워밍업)
    15M: FROM_DATE부터
    """
    # 4H: EMA200 워밍업을 위해 60일 전부터 수집
    from_4h = (pd.Timestamp(FROM_DATE) - timedelta(days=60)).strftime("%Y-%m-%d")
    # 1H: MACD(26) + 여유분을 위해 5일 전부터 수집
    from_1h = (pd.Timestamp(FROM_DATE) - timedelta(days=5)).strftime("%Y-%m-%d")

    df_4h = await loader.fetch_or_load_csv(symbol, "4h", from_4h, TO_DATE, CACHE_DIR)
    df_1h = await loader.fetch_or_load_csv(symbol, "1h", from_1h, TO_DATE, CACHE_DIR)
    df_15m = await loader.fetch_or_load_csv(symbol, "15m", FROM_DATE, TO_DATE, CACHE_DIR)

    return df_4h, df_1h, df_15m


def run_backtest_for_symbol(
    symbol: str,
    df_4h: pd.DataFrame,
    df_1h: pd.DataFrame,
    df_15m: pd.DataFrame,
    params: AllStrategyParams,
    leverage: int,
) -> dict:
    """단일 종목 백테스트를 실행하고 결과 dict를 반환한다."""
    min_trades = params.walk_forward.min_trades_per_window  # YAML에서 로드
    yaml_min_trades = 15  # strategy_params.yaml의 min_trades
    try:
        # 1단계에서 YAML의 min_trades를 읽는다
        from pathlib import Path as _P
        import yaml as _yaml
        _cfg_path = _P(__file__).parent.parent / "config" / "strategy_params.yaml"
        with open(_cfg_path) as _f:
            _cfg = _yaml.safe_load(_f)
        yaml_min_trades = _cfg.get("min_trades", 15)
    except (OSError, KeyError, ValueError):
        pass

    try:
        bt = Backtester(
            params,
            initial_balance=INITIAL_BALANCE,
            leverage=leverage,
        )
        report = bt.run(df_4h, df_1h, df_15m)
        trades = bt.get_trades()
        debug_counts = bt.get_debug_counts()

        # 워크포워드 검증 실행 (YAML 설정 사용)
        wf_params = params.walk_forward
        wf_result = None
        try:
            wf = WalkForwardValidator(
                df_4h, df_1h, df_15m,
                initial_balance=INITIAL_BALANCE,
                train_days=wf_params.train_days,
                test_days=wf_params.test_days,
                min_trades_per_window=wf_params.min_trades_per_window,
                step_days=wf_params.step_days,
            )
            wf_result = wf.run(params)
        except (KeyError, ValueError, ZeroDivisionError) as e:
            logger.debug(f"{symbol} 워크포워드 검증 실패: {e}")

        # OOS 프로그레시브 검증 (거래 수가 최소 기준 이상일 때만)
        oos_result = None
        if report.total_trades >= yaml_min_trades:
            try:
                oos = OutOfSampleValidator(
                    df_4h, df_1h, df_15m,
                    initial_balance=INITIAL_BALANCE,
                )
                oos_result = oos.run(
                    params,
                    min_trades_stage1=10,
                    train_days=wf_params.train_days,
                    test_days=wf_params.test_days,
                )
            except (KeyError, ValueError, ZeroDivisionError) as e:
                logger.debug(f"{symbol} OOS 검증 실패: {e}")

        # 민감도 분석 (OOS 통과 종목만)
        sensitivity_result = None
        if oos_result and oos_result.final_passed and params.sensitivity.enabled:
            try:
                sa = SensitivityAnalyzer(
                    df_4h, df_1h, df_15m,
                    params=params,
                    initial_balance=INITIAL_BALANCE,
                )
                sensitivity_result = sa.analyze()
            except (KeyError, ValueError, ZeroDivisionError) as e:
                logger.debug(f"{symbol} 민감도 분석 실패: {e}")

        return {
            "symbol": symbol,
            "status": "OK",
            "total_trades": report.total_trades,
            "win_rate": round(report.win_rate, 4),
            "profit_factor": round(report.profit_factor, 2),
            "sharpe_ratio": round(report.sharpe_ratio, 2),
            "sortino_ratio": round(report.sortino_ratio, 2),
            "max_drawdown_pct": round(report.max_drawdown_pct, 4),
            "total_pnl_usdt": round(report.total_pnl_usdt, 2),
            "total_pnl_pct": round(report.total_pnl_pct, 4),
            "avg_trade_duration_hours": round(report.avg_trade_duration_hours, 1) if report.avg_trade_duration_hours else None,
            "best_trade_usdt": round(report.best_trade_usdt, 2),
            "worst_trade_usdt": round(report.worst_trade_usdt, 2),
            "expectancy_usdt": round(report.expectancy_usdt, 2),
            "acceptable": report.is_acceptable(min_trades=yaml_min_trades),
            "min_trades_met": report.total_trades >= yaml_min_trades,
            "walk_forward": {
                "robustness_score": wf_result.robustness_score if wf_result else None,
                "is_robust": wf_result.is_robust() if wf_result else None,
                "avg_test_sharpe": wf_result.avg_test_sharpe if wf_result else None,
                "avg_degradation": wf_result.avg_degradation if wf_result else None,
                "test_positive_windows": wf_result.test_positive_windows if wf_result else None,
                "total_windows": wf_result.total_windows if wf_result else None,
            },
            "oos_validation": {
                "final_passed": oos_result.final_passed if oos_result else None,
                "stages": [
                    {"stage": s.stage, "passed": s.passed, "reason": s.reason}
                    for s in oos_result.stages
                ] if oos_result else None,
            },
            "sensitivity": {
                "overall_stable": sensitivity_result.overall_stable if sensitivity_result else None,
                "unstable_params": sensitivity_result.unstable_params if sensitivity_result else None,
            },
            "trades": trades,
            "candle_counts": {
                "4h": len(df_4h),
                "1h": len(df_1h),
                "15m": len(df_15m),
            },
            "debug_counts": debug_counts,
        }
    except (KeyError, ValueError, ZeroDivisionError, TypeError) as e:
        return {
            "symbol": symbol,
            "status": "ERROR",
            "error": str(e),
            "total_trades": 0,
        }


def generate_symbol_report(result: dict, out_dir: Path, min_trades: int) -> None:
    """종목별 상세 보고서를 텍스트 파일로 저장한다."""
    symbol = result["symbol"]
    safe_name = symbol.replace("/", "_").replace(":", "_")
    filepath = out_dir / f"{safe_name}.txt"

    lines = []
    lines.append(f"{'='*60}")
    lines.append(f"  백테스팅 보고서: {symbol}")
    lines.append(f"  기간: {FROM_DATE} ~ {TO_DATE}")
    lines.append(f"  초기 자본: ${INITIAL_BALANCE:,.0f}")
    lines.append(f"{'='*60}")
    lines.append("")

    if result["status"] == "ERROR":
        lines.append(f"  ❌ 오류: {result.get('error', 'Unknown')}")
        filepath.write_text("\n".join(lines), encoding="utf-8")
        return

    if result["total_trades"] == 0:
        lines.append("  ⚠️ 거래 없음 — 백테스트 기간 동안 시그널 미발생")
        filepath.write_text("\n".join(lines), encoding="utf-8")
        return

    r = result
    acceptable = r.get("acceptable", False)
    grade = "🟢 투자 가능" if acceptable else "🔴 기준 미달"

    lines.append(f"  종합 평가: {grade}")
    lines.append("")
    lines.append(f"  ┌────────────────────────────────────────┐")
    lines.append(f"  │  총 거래      : {r['total_trades']:>6}회                  │")
    lines.append(f"  │  승률          : {r['win_rate']*100:>6.1f}%   {'✓' if r['win_rate']>=0.45 else '✗'} (≥45%)    │")
    lines.append(f"  │  Profit Factor : {r['profit_factor']:>6.2f}    {'✓' if r['profit_factor']>=1.8 else '✗'} (≥1.8)    │")
    lines.append(f"  │  Sharpe Ratio  : {r['sharpe_ratio']:>6.2f}    {'✓' if r['sharpe_ratio']>=1.5 else '✗'} (≥1.5)    │")
    lines.append(f"  │  Sortino Ratio : {r['sortino_ratio']:>6.2f}                   │")
    lines.append(f"  │  최대 드로다운 : {r['max_drawdown_pct']*100:>6.1f}%   {'✓' if r['max_drawdown_pct']<=0.20 else '✗'} (≤20%)    │")
    lines.append(f"  │  총 P&L (USDT) : {r['total_pnl_usdt']:>+8.2f}                │")
    lines.append(f"  │  총 수익률     : {r['total_pnl_pct']*100:>+6.1f}%                  │")
    lines.append(f"  │  기대값/거래   : {r['expectancy_usdt']:>+8.2f} USDT           │")
    lines.append(f"  │  최고 거래     : {r['best_trade_usdt']:>+8.2f} USDT           │")
    lines.append(f"  │  최악 거래     : {r['worst_trade_usdt']:>+8.2f} USDT           │")
    if r.get("avg_trade_duration_hours"):
        lines.append(f"  │  평균 보유     : {r['avg_trade_duration_hours']:>6.1f}시간               │")
    lines.append(f"  └────────────────────────────────────────┘")
    lines.append("")

    # 기준 충족 상세
    lines.append("  기준 충족 여부:")
    checks = [
        (f"최소 거래 수 ≥ {min_trades}회", r["total_trades"] >= min_trades),
        ("승률 ≥ 45%", r["win_rate"] >= 0.45),
        ("Profit Factor ≥ 1.8", r["profit_factor"] >= 1.8),
        ("Sharpe Ratio ≥ 1.5", r["sharpe_ratio"] >= 1.5),
        ("최대 드로다운 ≤ 20%", r["max_drawdown_pct"] <= 0.20),
    ]
    for label, ok in checks:
        lines.append(f"    {'✅' if ok else '❌'} {label}")

    # 워크포워드 검증 결과
    wf = r.get("walk_forward", {})
    if wf.get("robustness_score") is not None:
        lines.append("")
        lines.append("  워크포워드 검증:")
        lines.append(f"    견고성 점수: {wf['robustness_score']:.0f}/100 {'✅' if wf.get('is_robust') else '❌'}")
        lines.append(f"    검증 평균 Sharpe: {wf.get('avg_test_sharpe', 0):.2f}")
        lines.append(f"    성능 유지율: {wf.get('avg_degradation', 0):.1%}")
        lines.append(f"    수익 윈도우: {wf.get('test_positive_windows', 0)}/{wf.get('total_windows', 0)}")

    # OOS 검증 결과
    oos = r.get("oos_validation", {})
    if oos.get("stages"):
        lines.append("")
        lines.append("  OOS 프로그레시브 검증:")
        final = "PASS" if oos.get("final_passed") else "FAIL"
        lines.append(f"    최종 판정: {final}")
        for s in oos["stages"]:
            status = "✅" if s["passed"] else "❌"
            lines.append(f"    {status} Stage {s['stage']}: {s['reason']}")

    # 민감도 분석 결과
    sens = r.get("sensitivity", {})
    if sens.get("overall_stable") is not None:
        lines.append("")
        lines.append("  파라미터 민감도:")
        if sens["overall_stable"]:
            lines.append("    ✅ 전체 안정 — 과적합 위험 낮음")
        else:
            lines.append(f"    ❌ 불안정 파라미터: {', '.join(sens.get('unstable_params', []))}")

    filepath.write_text("\n".join(lines), encoding="utf-8")


def generate_summary_report(all_results: list[dict], out_dir: Path, min_trades: int) -> None:
    """전 종목 종합 요약 보고서를 생성한다."""
    filepath = out_dir / "_SUMMARY.txt"

    ok_results = [r for r in all_results if r["status"] == "OK" and r["total_trades"] > 0]
    ok_results.sort(key=lambda r: r.get("sharpe_ratio", 0), reverse=True)

    acceptable = [r for r in ok_results if r.get("acceptable")]
    profitable = [r for r in ok_results if r.get("total_pnl_usdt", 0) > 0]
    oos_passed = [r for r in ok_results if r.get("oos_validation", {}).get("final_passed")]

    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"  OKX 전 종목 백테스팅 종합 보고서")
    lines.append(f"  기간: {FROM_DATE} ~ {TO_DATE} | 전략: Triple Confirmation (Layer1+OR)")
    lines.append(f"  초기 자본: ${INITIAL_BALANCE:,.0f} | 최소 거래: {min_trades}회 | 생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"{'='*70}")
    lines.append("")
    lines.append(f"  전체 종목: {len(all_results)}개")
    lines.append(f"  거래 발생: {len(ok_results)}개")
    lines.append(f"  수익 종목: {len(profitable)}개")
    lines.append(f"  기준 충족 (투자 가능): {len(acceptable)}개")
    lines.append(f"  OOS 검증 통과: {len(oos_passed)}개")
    lines.append("")

    if acceptable:
        lines.append(f"  ┌{'─'*66}┐")
        lines.append(f"  │  🟢 투자 가능 종목 (거래≥{min_trades}, Sharpe≥1.5, 승률≥45%, PF≥1.8, DD≤20%) │")
        lines.append(f"  ├{'─'*66}┤")
        lines.append(f"  │  {'종목':<20} {'거래':>5} {'승률':>7} {'PF':>6} {'Sharpe':>7} {'DD':>7} {'P&L':>10} │")
        lines.append(f"  ├{'─'*66}┤")
        for r in acceptable:
            sym = r["symbol"][:20]
            lines.append(
                f"  │  {sym:<20} {r['total_trades']:>5} "
                f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
                f"{r['sharpe_ratio']:>6.2f} {r['max_drawdown_pct']*100:>5.1f}% "
                f"{r['total_pnl_usdt']:>+8.2f} │"
            )
        lines.append(f"  └{'─'*66}┘")
        lines.append("")

    # Top 20 by Sharpe
    lines.append(f"  ┌{'─'*66}┐")
    lines.append(f"  │  📊 전체 순위 Top 20 (Sharpe Ratio 기준)                          │")
    lines.append(f"  ├{'─'*66}┤")
    lines.append(f"  │  {'#':>3} {'종목':<18} {'거래':>5} {'승률':>7} {'PF':>6} {'Sharpe':>7} {'P&L':>10} │")
    lines.append(f"  ├{'─'*66}┤")
    for i, r in enumerate(ok_results[:20], 1):
        sym = r["symbol"][:18]
        flag = "🟢" if r.get("acceptable") else "  "
        lines.append(
            f"  │ {flag}{i:>2} {sym:<18} {r['total_trades']:>5} "
            f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
            f"{r['sharpe_ratio']:>6.2f} {r['total_pnl_usdt']:>+8.2f} │"
        )
    lines.append(f"  └{'─'*66}┘")
    lines.append("")

    # Bottom 10
    if len(ok_results) > 10:
        worst = ok_results[-10:]
        worst.reverse()
        lines.append(f"  ┌{'─'*66}┐")
        lines.append(f"  │  🔴 최악 종목 Bottom 10                                          │")
        lines.append(f"  ├{'─'*66}┤")
        for r in worst:
            sym = r["symbol"][:18]
            lines.append(
                f"  │    {sym:<18} {r['total_trades']:>5} "
                f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} "
                f"{r['sharpe_ratio']:>6.2f} {r['total_pnl_usdt']:>+8.2f} │"
            )
        lines.append(f"  └{'─'*66}┘")

    # 워크포워드 검증 결과 섹션
    wf_results = [r for r in ok_results if r.get("walk_forward", {}).get("robustness_score") is not None]
    if wf_results:
        robust = [r for r in wf_results if r.get("walk_forward", {}).get("is_robust")]
        lines.append("")
        lines.append(f"  ┌{'─'*66}┐")
        lines.append(f"  │  🔍 워크포워드 검증 결과 (견고성 점수 ≥ 50 = 과적합 위험 낮음)     │")
        lines.append(f"  ├{'─'*66}┤")
        lines.append(f"  │  검증 완료: {len(wf_results)}종목 | 견고: {len(robust)}종목                          │")
        lines.append(f"  ├{'─'*66}┤")
        lines.append(f"  │  {'종목':<18} {'견고성':>6} {'검증Sharpe':>9} {'유지율':>7} {'수익윈도우':>9} │")
        lines.append(f"  ├{'─'*66}┤")
        wf_sorted = sorted(wf_results, key=lambda r: r["walk_forward"]["robustness_score"], reverse=True)
        for r in wf_sorted[:20]:
            wf = r["walk_forward"]
            sym = r["symbol"][:18]
            robust_flag = "✓" if wf.get("is_robust") else "✗"
            lines.append(
                f"  │  {sym:<18} {wf['robustness_score']:>5.0f}{robust_flag} "
                f"{wf['avg_test_sharpe']:>8.2f} "
                f"{wf['avg_degradation']:>6.1%} "
                f"{wf['test_positive_windows']}/{wf['total_windows']:>7} │"
            )
        lines.append(f"  └{'─'*66}┘")

    # OOS 검증 결과 섹션
    if oos_passed:
        lines.append("")
        lines.append(f"  ┌{'─'*66}┐")
        lines.append(f"  │  ✅ OOS 검증 통과 종목 (3단계 프로그레시브 검증 완료)              │")
        lines.append(f"  ├{'─'*66}┤")
        lines.append(f"  │  {'종목':<18} {'거래':>5} {'Sharpe':>7} {'P&L':>10} {'민감도':>8} │")
        lines.append(f"  ├{'─'*66}┤")
        for r in oos_passed:
            sym = r["symbol"][:18]
            sens = r.get("sensitivity", {})
            stable = "안정" if sens.get("overall_stable") else ("불안정" if sens.get("overall_stable") is not None else "미분석")
            lines.append(
                f"  │  {sym:<18} {r['total_trades']:>5} "
                f"{r['sharpe_ratio']:>6.2f} {r['total_pnl_usdt']:>+8.2f} "
                f"{stable:>8} │"
            )
        lines.append(f"  └{'─'*66}┘")

    filepath.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"종합 보고서 저장: {filepath}")

    # JSON 데이터도 저장
    json_path = out_dir / "_summary.json"
    summary_data = []
    for r in all_results:
        d = {k: v for k, v in r.items() if k != "trades"}
        summary_data.append(d)
    json_path.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8")


async def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

    settings = load_settings()
    rest = OKXRestClient(settings.okx)
    params = AllStrategyParams.from_yaml()
    loader = HistoricalDataLoader(rest)

    # YAML에서 min_trades 로드
    import yaml
    cfg_path = Path(__file__).parent.parent / "config" / "strategy_params.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    min_trades = cfg.get("min_trades", 15)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    signal_mode = params.backtest.signal_mode
    logger.info("OKX 전 종목 백테스팅 시작")
    logger.info(f"기간: {FROM_DATE} ~ {TO_DATE}")
    logger.info(f"시그널 모드: {signal_mode} | 최소 거래: {min_trades}회")
    logger.info(f"워크포워드: {params.walk_forward.train_days}일 학습 / {params.walk_forward.test_days}일 검증")

    symbols = await fetch_all_swap_symbols(rest)
    logger.info(f"총 {len(symbols)}개 USDT 무기한 선물 종목 발견")

    all_results = []

    for i, symbol in enumerate(symbols, 1):
        logger.info(f"[{i}/{len(symbols)}] {symbol} 백테스팅 중...")

        # HistoricalDataLoader로 전체 기간 데이터 수집 (CSV 캐시)
        try:
            df_4h, df_1h, df_15m = await fetch_symbol_data(loader, symbol)
        except (KeyError, ValueError, OSError) as e:
            logger.warning(f"  {symbol}: 데이터 수집 실패: {e}")
            all_results.append({"symbol": symbol, "status": "ERROR", "error": str(e), "total_trades": 0})
            continue

        # 데이터 부족 체크
        if len(df_4h) < MIN_CANDLES_4H:
            logger.debug(f"  {symbol}: 4H 캔들 부족 ({len(df_4h)}개) — 스킵")
            all_results.append({"symbol": symbol, "status": "SKIP", "reason": "데이터 부족", "total_trades": 0})
            continue

        result = run_backtest_for_symbol(symbol, df_4h, df_1h, df_15m, params, settings.trading.leverage)
        all_results.append(result)

        if result["total_trades"] > 0:
            pnl = result.get("total_pnl_usdt", 0)
            oos_status = ""
            oos_v = result.get("oos_validation", {})
            if oos_v.get("final_passed") is True:
                oos_status = " | OOS:PASS"
            elif oos_v.get("final_passed") is False:
                oos_status = " | OOS:FAIL"
            logger.info(f"  → 거래 {result['total_trades']}회 | 승률 {result['win_rate']*100:.1f}% | P&L: {pnl:+.2f}{oos_status}")
        else:
            logger.info(f"  → 시그널 없음")

        # 종목별 보고서 생성
        generate_symbol_report(result, RESULTS_DIR, min_trades)

    # 종합 보고서
    generate_summary_report(all_results, RESULTS_DIR, min_trades)

    await rest.close()

    ok = [r for r in all_results if r.get("acceptable")]
    oos_ok = [r for r in all_results if r.get("oos_validation", {}).get("final_passed")]
    traded = [r for r in all_results if r.get("total_trades", 0) > 0]
    logger.info(f"백테스팅 완료: {len(all_results)}종목 | 거래발생 {len(traded)}종목 | 기준충족 {len(ok)}종목 | OOS통과 {len(oos_ok)}종목")
    logger.info(f"결과: {RESULTS_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
