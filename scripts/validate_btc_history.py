"""
BTC 1분봉 historical 데이터 무결성 검증 리포트.

확인 항목:
- 월별 row 수 vs 기대치 (≈ 31일 × 24h × 60min = 44,640)
- 1분 단위 gap 누락률 (1.0 - actual_rows / expected_rows)
- 중복 timestamp 수 (반드시 0)
- 시작/종료 timestamp
- 가장 큰 gap top 10

사용 예:
    # 전체 범위 검증 (저장된 데이터 모두)
    python3 scripts/validate_btc_history.py

    # 범위 지정
    python3 scripts/validate_btc_history.py --start 2020-01-01 --end 2026-04-30

    # JSON 리포트 저장
    python3 scripts/validate_btc_history.py --output data_cache/ohlcv_1m/_validate_report.json
"""
from __future__ import annotations

import argparse
import calendar
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BASE_DIR = REPO_ROOT / "data_cache" / "ohlcv_1m"
DEFAULT_SYMBOL_DIR = "BTC-USDT-SWAP"


def _expected_minutes_in_month(year: int, month: int) -> int:
    days = calendar.monthrange(year, month)[1]
    return days * 24 * 60


def _format_pct(p: float) -> str:
    return f"{p*100:.3f}%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--symbol-dir", default=DEFAULT_SYMBOL_DIR)
    parser.add_argument("--start", default=None, help="YYYY-MM-DD UTC (default: 첫 월)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD UTC (default: 마지막 월)")
    parser.add_argument("--gap-min-minutes", type=int, default=2,
                        help="이 이상의 분 차이를 gap으로 표시")
    parser.add_argument("--output", default=None, help="JSON 리포트 저장 경로")
    args = parser.parse_args()

    base_dir = Path(args.base_dir) / args.symbol_dir
    if not base_dir.exists():
        logger.error(f"디렉토리 없음: {base_dir}")
        return 1

    parquets = sorted(base_dir.glob("*.parquet"))
    if not parquets:
        logger.error(f"parquet 파일 없음: {base_dir}")
        return 1

    logger.info(f"검증 시작: {base_dir} ({len(parquets)}개 월별 파일)")

    # 1. 월별 row count + duplicate + gap 분포
    per_month: dict[str, dict] = {}
    all_diffs: list[pd.Timedelta] = []
    all_first_ts: pd.Timestamp | None = None
    all_last_ts: pd.Timestamp | None = None
    total_rows = 0
    total_dups = 0
    largest_gaps: list[tuple[str, str, str]] = []  # (gap_start, gap_end, duration)

    for path in parquets:
        month = path.stem
        df = pd.read_parquet(path, columns=["close"])
        if df.empty:
            per_month[month] = {
                "rows": 0, "expected": 0, "missing_pct": 1.0, "duplicates": 0,
                "gap_count_2min": 0, "gap_count_5min": 0, "gap_count_1h": 0,
            }
            continue

        try:
            year, mm = int(month[:4]), int(month[5:7])
        except ValueError:
            logger.warning(f"월 형식 인식 실패: {month}")
            continue

        # 중복 체크 (raw 인덱스 기준)
        n_raw = len(df)
        df_dedup = df[~df.index.duplicated(keep="first")].sort_index()
        n_dedup = len(df_dedup)
        dups = n_raw - n_dedup

        # 기대치 vs 실제
        expected = _expected_minutes_in_month(year, mm)
        # 마지막 월(현재 진행 중)이면 이미 지난 분만 기대
        now_utc = datetime.now(timezone.utc)
        if year == now_utc.year and mm == now_utc.month:
            month_start = datetime(year, mm, 1, tzinfo=timezone.utc)
            expected = int((now_utc - month_start).total_seconds() // 60)

        missing_pct = max(0.0, 1.0 - n_dedup / max(expected, 1))

        # gap 분포
        diffs = df_dedup.index.to_series().diff().dropna()
        all_diffs.extend(diffs.tolist())
        gap_2min = int((diffs > pd.Timedelta(minutes=2)).sum())
        gap_5min = int((diffs > pd.Timedelta(minutes=5)).sum())
        gap_1h = int((diffs > pd.Timedelta(hours=1)).sum())

        # 큰 gap 수집 (월별 상위 3개)
        big = diffs.nlargest(3)
        for ts_end, dur in big.items():
            if dur > pd.Timedelta(minutes=args.gap_min_minutes):
                idx = df_dedup.index.get_loc(ts_end)
                ts_start = df_dedup.index[idx - 1]
                largest_gaps.append((str(ts_start), str(ts_end), str(dur)))

        per_month[month] = {
            "rows": n_dedup,
            "expected": expected,
            "missing_pct": round(missing_pct, 6),
            "duplicates": dups,
            "gap_count_2min": gap_2min,
            "gap_count_5min": gap_5min,
            "gap_count_1h": gap_1h,
            "first_ts": str(df_dedup.index[0]),
            "last_ts": str(df_dedup.index[-1]),
        }

        total_rows += n_dedup
        total_dups += dups
        if all_first_ts is None or df_dedup.index[0] < all_first_ts:
            all_first_ts = df_dedup.index[0]
        if all_last_ts is None or df_dedup.index[-1] > all_last_ts:
            all_last_ts = df_dedup.index[-1]

    # 2. 누락된 월 감지
    missing_months: list[str] = []
    if all_first_ts is not None and all_last_ts is not None:
        cur = pd.Timestamp(year=all_first_ts.year, month=all_first_ts.month, day=1, tz="UTC")
        end = pd.Timestamp(year=all_last_ts.year, month=all_last_ts.month, day=1, tz="UTC")
        while cur <= end:
            m = f"{cur.year:04d}-{cur.month:02d}"
            if m not in per_month:
                missing_months.append(m)
            cur = cur + pd.offsets.MonthBegin(1)

    # 3. 전체 통계
    total_expected = sum(m["expected"] for m in per_month.values())
    overall_missing_pct = max(0.0, 1.0 - total_rows / max(total_expected, 1))

    # 4. 큰 gap top 10
    largest_gaps.sort(key=lambda g: pd.Timedelta(g[2]), reverse=True)
    top_gaps = largest_gaps[:10]

    # 5. 리포트 출력
    report: dict = {
        "schema_version": 1,
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "base_dir": str(base_dir),
        "first_ts": str(all_first_ts) if all_first_ts else None,
        "last_ts": str(all_last_ts) if all_last_ts else None,
        "total_rows": total_rows,
        "total_expected": total_expected,
        "overall_missing_pct": round(overall_missing_pct, 6),
        "total_duplicates": total_dups,
        "missing_months": missing_months,
        "per_month": per_month,
        "largest_gaps_top10": [
            {"prev_ts": s, "next_ts": e, "duration": d}
            for s, e, d in top_gaps
        ],
        "pass_gates": {
            "no_duplicates": total_dups == 0,
            "missing_pct_under_0p1": overall_missing_pct <= 0.001,
            "no_missing_months": len(missing_months) == 0,
        },
    }

    print()
    print("=" * 70)
    print(f"  📋 BTC 1m 데이터 무결성 검증 리포트")
    print("=" * 70)
    print(f"  범위: {all_first_ts} → {all_last_ts}")
    print(f"  총 행수: {total_rows:,} / 기대치: {total_expected:,}")
    print(f"  누락률: {_format_pct(overall_missing_pct)} (게이트: ≤ 0.1%)")
    print(f"  중복: {total_dups:,} (게이트: 0)")
    print(f"  누락 월: {len(missing_months)}개 {missing_months[:5] if missing_months else ''}")
    print()
    print("  통과 게이트:")
    for k, v in report["pass_gates"].items():
        print(f"    {'✅' if v else '❌'} {k}")
    print()

    print("  ┌─ 월별 상세 (rows / 기대 / 누락 / dup / gap_2min) ─────────┐")
    for m in sorted(per_month.keys()):
        d = per_month[m]
        flag = "✅" if d["missing_pct"] <= 0.001 and d["duplicates"] == 0 else "⚠️"
        print(
            f"  │ {flag} {m}  "
            f"{d['rows']:>6,}/{d['expected']:>6,}  "
            f"{_format_pct(d['missing_pct']):>8}  "
            f"dup={d['duplicates']:<3}  "
            f"gap2={d['gap_count_2min']:<4}  "
            f"gap5={d['gap_count_5min']:<3}"
        )
    print("  └────────────────────────────────────────────────────────────┘")

    if top_gaps:
        print()
        print("  🔍 큰 gap Top 10:")
        for s, e, d in top_gaps:
            print(f"    {s} → {e}  ({d})")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str))
        logger.info(f"JSON 리포트 저장: {out}")

    # 종료 코드: 게이트 모두 통과 → 0, 아니면 1
    return 0 if all(report["pass_gates"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
