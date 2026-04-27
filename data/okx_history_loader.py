"""
OKX history-candles 전용 deep-history loader.

Live trading 흐름과 분리된 historical 1분봉 다운로더.
ccxt async + `params={"history": True}`로 history-candles 엔드포인트를 사용.

Storage: 월별 Parquet 파티션 (zstd 압축).
  data_cache/ohlcv_1m/BTC-USDT-SWAP/YYYY-MM.parquet

Resume-on-failure: 디렉토리 내 가장 최신 timestamp를 찾아 그 다음 분부터 재개.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm


_OKX_HISTORY_LIMIT = 100  # OKX history-candles per-call limit
_BAR_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000}
_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass
class HistoryFetchPlan:
    symbol: str
    timeframe: str
    start_ms: int
    end_ms: int
    base_dir: Path

    @property
    def bar_ms(self) -> int:
        return _BAR_MS[self.timeframe]


def _ts_to_month(ts_ms: int) -> str:
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return f"{dt.year:04d}-{dt.month:02d}"


def _symbol_to_dirname(symbol: str) -> str:
    """ 'BTC/USDT:USDT' -> 'BTC-USDT-SWAP' """
    base, rest = symbol.split("/")
    quote = rest.split(":")[0]
    return f"{base}-{quote}-SWAP"


class MonthlyParquetWriter:
    """
    월별 Parquet 파티션 writer.

    - 인메모리 버퍼로 들어온 bar를 월별로 그룹핑.
    - 호출자가 결정한 시점(월 경계 넘을 때 + 종료 시)에 flush.
    - 기존 파일이 있으면 timestamp 기준 dedup 후 병합.
    """

    def __init__(self, base_dir: Path):
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._buffers: dict[str, list[list]] = {}

    def add_batch(self, bars: list[list]) -> set[str]:
        """배치 추가. bar = [ts_ms, open, high, low, close, volume]. 추가된 month 집합 반환."""
        touched: set[str] = set()
        for bar in bars:
            month = _ts_to_month(bar[0])
            self._buffers.setdefault(month, []).append(bar)
            touched.add(month)
        return touched

    def flush_months(self, months: set[str]) -> dict[str, int]:
        """주어진 월들을 디스크에 flush. {month: written_rows} 반환."""
        written = {}
        for month in sorted(months):
            buf = self._buffers.pop(month, [])
            if not buf:
                continue
            written[month] = self._write_month(month, buf)
        return written

    def flush_all(self) -> dict[str, int]:
        return self.flush_months(set(self._buffers.keys()))

    def _write_month(self, month: str, buf: list[list]) -> int:
        path = self._base_dir / f"{month}.parquet"
        new_df = pd.DataFrame(buf, columns=["timestamp", *_OHLCV_COLUMNS])
        new_df["timestamp"] = pd.to_datetime(new_df["timestamp"], unit="ms", utc=True)
        new_df = new_df.set_index("timestamp")

        if path.exists():
            old_df = pd.read_parquet(path)
            merged = pd.concat([old_df, new_df])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        else:
            merged = new_df[~new_df.index.duplicated(keep="last")].sort_index()

        merged.to_parquet(path, compression="zstd", engine="pyarrow")
        return len(merged)


class OKXHistoryLoader:
    """OKX history-candles deep-history fetcher."""

    def __init__(self, rate_limit_sleep_ms: int = 0):
        # ccxt enableRateLimit handles pacing; rate_limit_sleep_ms is extra cushion.
        self._extra_sleep = rate_limit_sleep_ms / 1000.0
        self._ex: Optional[ccxt.okx] = None

    async def __aenter__(self):
        self._ex = ccxt.okx({
            "options": {"defaultType": "swap"},
            "enableRateLimit": True,
        })
        await self._ex.load_markets()
        return self

    async def __aexit__(self, *args):
        if self._ex is not None:
            await self._ex.close()

    async def latest_bar_ts_ms(self, symbol: str, timeframe: str) -> Optional[int]:
        bars = await self._ex.fetch_ohlcv(symbol, timeframe, limit=1)
        if not bars:
            return None
        return int(bars[-1][0])

    async def fetch_chunk(
        self, symbol: str, timeframe: str, since_ms: int, limit: int = _OKX_HISTORY_LIMIT,
    ) -> list[list]:
        """history-candles 엔드포인트로 forward 1 chunk fetch. 빈 리스트면 더 없음."""
        bars = await self._ex.fetch_ohlcv(
            symbol, timeframe, since=since_ms, limit=limit,
            params={"history": True},
        )
        if self._extra_sleep:
            await asyncio.sleep(self._extra_sleep)
        return bars or []

    async def fetch_to_parquet(self, plan: HistoryFetchPlan, ignore_resume: bool = False) -> dict:
        """
        plan.start_ms ~ plan.end_ms 구간의 1m bar를 forward로 받아 월별 Parquet에 저장.
        멱등(idempotent): 이미 저장된 timestamp는 dedup 처리됨.

        Args:
            ignore_resume: True면 resume_ms 무시하고 plan.start_ms부터 강제 시작 (backfill용)
        """
        symbol_dir = plan.base_dir / _symbol_to_dirname(plan.symbol)
        writer = MonthlyParquetWriter(symbol_dir)
        bar_ms = plan.bar_ms

        # 1. resume point 결정
        if ignore_resume:
            since = plan.start_ms
            logger.info(f"[backfill] resume 무시 — plan.start_ms부터 강제 fetch: {datetime.fromtimestamp(since/1000, tz=timezone.utc).isoformat()}")
        else:
            resume_ms = self._latest_stored_ts_ms(symbol_dir)
            if resume_ms is not None and resume_ms >= plan.start_ms:
                since = resume_ms + bar_ms
                logger.info(f"[resume] 기존 Parquet에서 복구: {datetime.fromtimestamp(since/1000, tz=timezone.utc).isoformat()}")
            else:
                since = plan.start_ms

        # 2. fetch loop
        total_bars = 0
        last_month: Optional[str] = None
        approx_total_bars = max(1, (plan.end_ms - since) // bar_ms)
        pbar = tqdm(total=approx_total_bars, unit="bar", dynamic_ncols=True)
        try:
            while since < plan.end_ms:
                bars = await self.fetch_chunk(plan.symbol, plan.timeframe, since)
                if not bars:
                    logger.warning(f"[fetch] 빈 응답 since={since} — 종료")
                    break

                # filter: only bars strictly within plan window, and after `since`
                bars = [b for b in bars if since <= int(b[0]) < plan.end_ms]
                if not bars:
                    # progressed past data range
                    break

                touched = writer.add_batch(bars)
                total_bars += len(bars)
                pbar.update(len(bars))

                last_ts = int(bars[-1][0])
                next_since = last_ts + bar_ms

                # 월 경계 넘을 때 이전 월 flush (현재 월은 buffer에 유지)
                current_month = _ts_to_month(last_ts)
                if last_month is not None and current_month != last_month:
                    completed = {m for m in touched if m != current_month}
                    if completed:
                        writer.flush_months(completed)
                last_month = current_month

                if next_since <= since:
                    # safeguard against infinite loop on duplicate timestamps
                    next_since = since + bar_ms
                since = next_since

        finally:
            pbar.close()

        # 3. final flush + meta write
        flushed = writer.flush_all()
        meta = self._write_meta(symbol_dir, total_added=total_bars, by_month=flushed)
        return meta

    async def fill_gaps(self, plan: HistoryFetchPlan, min_gap_minutes: int = 5) -> dict:
        """
        plan.start_ms ~ plan.end_ms 범위에서 저장된 데이터의 gap을 감지하고,
        각 gap 구간을 forward fetch로 채운다.

        Args:
            min_gap_minutes: 이 분수 이상의 시간 간격을 gap으로 간주

        Returns:
            {"gaps_detected": N, "gaps_filled": N, "rows_added": N, "remaining_gaps": [...]}
        """
        from data.okx_history_loader import load_parquet_range  # self-import for staticness

        symbol_dir = plan.base_dir / _symbol_to_dirname(plan.symbol)
        if not symbol_dir.exists():
            logger.warning(f"[gap-fill] 디렉토리 없음 — full backfill 권장: {symbol_dir}")
            return {"gaps_detected": 0, "gaps_filled": 0, "rows_added": 0, "remaining_gaps": []}

        bar_ms = plan.bar_ms
        gap_threshold = pd.Timedelta(minutes=min_gap_minutes)

        # 1. 모든 월별 parquet의 timestamp 인덱스 모으기
        all_ts: list[pd.Timestamp] = []
        for path in sorted(symbol_dir.glob("*.parquet")):
            df = pd.read_parquet(path, columns=["close"])
            all_ts.extend(df.index.tolist())
        if not all_ts:
            logger.warning("[gap-fill] 저장된 데이터 없음")
            return {"gaps_detected": 0, "gaps_filled": 0, "rows_added": 0, "remaining_gaps": []}

        ts_series = pd.Series(sorted(set(all_ts)))
        # plan 범위로 자르기
        start_pd = pd.Timestamp(plan.start_ms, unit="ms", tz="UTC")
        end_pd = pd.Timestamp(plan.end_ms, unit="ms", tz="UTC")
        ts_series = ts_series[(ts_series >= start_pd) & (ts_series < end_pd)]
        if len(ts_series) < 2:
            logger.warning("[gap-fill] 범위 내 데이터 < 2개 — full backfill 필요")
            return {"gaps_detected": 0, "gaps_filled": 0, "rows_added": 0, "remaining_gaps": []}

        diffs = ts_series.diff().dropna()
        gap_mask = diffs > gap_threshold
        gaps: list[tuple[int, int]] = []
        for i in diffs.index[gap_mask]:
            prev_ts = ts_series.iloc[i - 1]
            cur_ts = ts_series.iloc[i]
            gap_start_ms = int(prev_ts.value // 1_000_000) + bar_ms
            gap_end_ms = int(cur_ts.value // 1_000_000)
            if gap_end_ms - gap_start_ms >= bar_ms:
                gaps.append((gap_start_ms, gap_end_ms))

        # 시작 부분 누락 (plan.start_ms ~ 첫 ts) / 끝 부분 누락 (마지막 ts ~ plan.end_ms)
        first_ts_ms = int(ts_series.iloc[0].value // 1_000_000)
        if first_ts_ms - plan.start_ms >= bar_ms:
            gaps.insert(0, (plan.start_ms, first_ts_ms))
        last_ts_ms = int(ts_series.iloc[-1].value // 1_000_000)
        if plan.end_ms - (last_ts_ms + bar_ms) >= bar_ms:
            gaps.append((last_ts_ms + bar_ms, plan.end_ms))

        if not gaps:
            logger.info("[gap-fill] 감지된 gap 없음 — 데이터 무결성 양호")
            return {"gaps_detected": 0, "gaps_filled": 0, "rows_added": 0, "remaining_gaps": []}

        logger.info(f"[gap-fill] {len(gaps)}개 gap 감지. 채우기 시작...")
        for i, (gs, ge) in enumerate(gaps[:5]):
            logger.info(
                f"  gap {i+1}: "
                f"{datetime.fromtimestamp(gs/1000, tz=timezone.utc).isoformat()} → "
                f"{datetime.fromtimestamp(ge/1000, tz=timezone.utc).isoformat()} "
                f"(~{(ge-gs)//bar_ms} bars)"
            )
        if len(gaps) > 5:
            logger.info(f"  ... 외 {len(gaps)-5}개")

        # 2. 각 gap을 forward fetch로 채움
        writer = MonthlyParquetWriter(symbol_dir)
        total_added = 0
        gaps_filled = 0
        remaining: list[dict] = []

        for gs, ge in gaps:
            since = gs
            chunk_added = 0
            attempts = 0
            while since < ge and attempts < 200:  # 200 chunks safety
                bars = await self.fetch_chunk(plan.symbol, plan.timeframe, since)
                if not bars:
                    break
                bars = [b for b in bars if since <= int(b[0]) < ge]
                if not bars:
                    break
                writer.add_batch(bars)
                chunk_added += len(bars)
                last_ts = int(bars[-1][0])
                next_since = last_ts + bar_ms
                if next_since <= since:
                    next_since = since + bar_ms
                since = next_since
                attempts += 1

            if chunk_added > 0:
                gaps_filled += 1
                total_added += chunk_added
            else:
                remaining.append({
                    "start": datetime.fromtimestamp(gs/1000, tz=timezone.utc).isoformat(),
                    "end": datetime.fromtimestamp(ge/1000, tz=timezone.utc).isoformat(),
                    "expected_bars": (ge - gs) // bar_ms,
                })

        # 3. flush + meta 갱신
        writer.flush_all()
        meta = self._write_meta(symbol_dir, total_added=total_added, by_month={})

        return {
            "gaps_detected": len(gaps),
            "gaps_filled": gaps_filled,
            "rows_added": total_added,
            "remaining_gaps": remaining,
            "total_rows": meta["total_rows"],
        }

    @staticmethod
    def _latest_stored_ts_ms(symbol_dir: Path) -> Optional[int]:
        if not symbol_dir.exists():
            return None
        parquets = sorted(symbol_dir.glob("*.parquet"))
        if not parquets:
            return None
        latest_path = parquets[-1]
        df = pd.read_parquet(latest_path, columns=["close"])
        if df.empty:
            return None
        last_ts = df.index.max()
        return int(last_ts.value // 1_000_000)  # ns -> ms

    @staticmethod
    def _write_meta(symbol_dir: Path, total_added: int, by_month: dict[str, int]) -> dict:
        """디스크 상의 모든 월별 parquet을 스캔해 권위적(authoritative) meta 작성."""
        meta_path = symbol_dir / "_meta.json"
        row_counts: dict[str, int] = {}
        gaps: dict[str, int] = {}
        for path in sorted(symbol_dir.glob("*.parquet")):
            month = path.stem
            df = pd.read_parquet(path, columns=["close"])
            row_counts[month] = len(df)
            if len(df) > 1:
                diffs = df.index.to_series().diff().dropna()
                gaps[month] = int((diffs > pd.Timedelta(minutes=5)).sum())
            else:
                gaps[month] = 0

        meta = {
            "schema_version": 1,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "row_counts": row_counts,
            "gap_counts": gaps,
            "total_rows": int(sum(row_counts.values())),
            "last_run_added_rows": total_added,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        return meta


def load_parquet_range(
    base_dir: Path,
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """저장된 월별 Parquet에서 [start, end) 구간을 단일 DataFrame으로 로드."""
    symbol_dir = base_dir / _symbol_to_dirname(symbol)
    if not symbol_dir.exists():
        raise FileNotFoundError(f"{symbol_dir} 없음")

    start = pd.Timestamp(start, tz="UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = pd.Timestamp(end, tz="UTC") if end.tzinfo is None else end.tz_convert("UTC")

    months = []
    cur = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    end_month = pd.Timestamp(year=end.year, month=end.month, day=1, tz="UTC")
    while cur <= end_month:
        months.append(f"{cur.year:04d}-{cur.month:02d}")
        cur = (cur + pd.offsets.MonthBegin(1)).tz_localize("UTC") if cur.tzinfo is None else cur + pd.offsets.MonthBegin(1)

    frames = []
    for m in months:
        p = symbol_dir / f"{m}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return pd.DataFrame(columns=list(_OHLCV_COLUMNS))

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.loc[(df.index >= start) & (df.index < end)]
