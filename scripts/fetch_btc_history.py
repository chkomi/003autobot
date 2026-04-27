"""
BTC/USDT swap 1분봉 historical 다운로드 스크립트.

모드:
    full       : 기본. 저장된 최신 ts 이후부터 forward fetch (resume).
    backfill   : --start부터 강제 fetch (resume 무시). 과거 미수집 구간 채울 때.
    gap-fill   : 저장된 데이터의 gap을 자동 감지해 누락 구간만 채움.

사용 예:
    # 1) 2020-01-01부터 backfill (과거 미수집 구간 전부)
    python3 scripts/fetch_btc_history.py --start 2020-01-01 --mode backfill

    # 2) 기본 resume (최신 데이터 이후만)
    python3 scripts/fetch_btc_history.py

    # 3) gap-fill (이미 가진 범위 안의 누락 구간만)
    python3 scripts/fetch_btc_history.py --start 2020-01-01 --mode gap-fill

    # 4) 끝일 지정 (smoke test 용)
    python3 scripts/fetch_btc_history.py --start 2026-04-01 --end 2026-04-15

    # 5) ignore-resume 단축 (--mode backfill과 동일)
    python3 scripts/fetch_btc_history.py --start 2020-01-01 --ignore-resume
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.okx_history_loader import HistoryFetchPlan, OKXHistoryLoader  # noqa: E402

DEFAULT_BASE_DIR = REPO_ROOT / "data_cache" / "ohlcv_1m"
DEFAULT_SYMBOL = "BTC/USDT:USDT"
DEFAULT_TIMEFRAME = "1m"
DEFAULT_START = "2020-01-01"


def _to_utc_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--timeframe", default=DEFAULT_TIMEFRAME)
    parser.add_argument("--start", default=DEFAULT_START, help="YYYY-MM-DD UTC")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD UTC (default: now)")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument(
        "--mode",
        choices=["full", "backfill", "gap-fill"],
        default="full",
        help="full: resume; backfill: --start부터 강제; gap-fill: 누락 구간만",
    )
    parser.add_argument(
        "--ignore-resume",
        action="store_true",
        help="--mode backfill과 동일 (편의 옵션)",
    )
    parser.add_argument(
        "--gap-min-minutes",
        type=int,
        default=5,
        help="gap-fill 모드에서 gap으로 간주할 최소 분 단위 (기본 5)",
    )
    parser.add_argument(
        "--rate-limit-sleep-ms",
        type=int,
        default=50,
        help="API 호출 간 추가 sleep (기본 50ms)",
    )
    args = parser.parse_args()

    start_ms = _to_utc_ms(args.start)
    if args.end:
        end_ms = _to_utc_ms(args.end)
    else:
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    plan = HistoryFetchPlan(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_ms=start_ms,
        end_ms=end_ms,
        base_dir=Path(args.base_dir),
    )

    mode = args.mode
    if args.ignore_resume and mode == "full":
        mode = "backfill"

    logger.info(
        f"BTC history fetch [{mode}]: {args.symbol} {args.timeframe} "
        f"{datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).isoformat()} → "
        f"{datetime.fromtimestamp(end_ms/1000, tz=timezone.utc).isoformat()}"
    )

    async with OKXHistoryLoader(rate_limit_sleep_ms=args.rate_limit_sleep_ms) as loader:
        if mode == "gap-fill":
            result = await loader.fill_gaps(plan, min_gap_minutes=args.gap_min_minutes)
            logger.info(f"[gap-fill] {result['gaps_detected']}개 감지 / {result['gaps_filled']}개 채움 / +{result['rows_added']:,} rows")
            logger.info(f"[gap-fill] 총 행수: {result.get('total_rows', '?'):,}")
            if result["remaining_gaps"]:
                logger.warning(f"[gap-fill] 채우지 못한 gap {len(result['remaining_gaps'])}개:")
                for g in result["remaining_gaps"][:10]:
                    logger.warning(f"  {g['start']} → {g['end']} (~{g['expected_bars']} bars)")
        else:
            ignore_resume = (mode == "backfill")
            meta = await loader.fetch_to_parquet(plan, ignore_resume=ignore_resume)
            logger.info(f"완료. 총 {meta['total_rows']:,} rows. 이번 실행 추가: {meta['last_run_added_rows']:,}")
            logger.info(f"월별 gap >5min 카운트: {meta['gap_counts']}")


if __name__ == "__main__":
    asyncio.run(main())
