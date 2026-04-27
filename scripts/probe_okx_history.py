"""
OKX history-candles API probe.

Goal: confirm the earliest 1-minute bar available for BTC-USDT-SWAP.

OKX provides two endpoints:
  - /api/v5/market/candles         : recent candles (~1500 bars max)
  - /api/v5/market/history-candles : deep history (paginated by ts)

ccxt routes to history-candles when `params={"history": True}` is passed
(verified for okx exchange in ccxt 4.5+).

Usage:
    python3 scripts/probe_okx_history.py
"""
import asyncio
import time
from datetime import datetime, timezone

import ccxt.async_support as ccxt


SYMBOL = "BTC/USDT:USDT"  # ccxt unified symbol for BTC-USDT-SWAP
TIMEFRAME = "1m"


async def probe():
    ex = ccxt.okx({"options": {"defaultType": "swap"}, "enableRateLimit": True})
    try:
        await ex.load_markets()

        # 1. Recent candle endpoint - most recent bar
        recent = await ex.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=1)
        if recent:
            ts = datetime.fromtimestamp(recent[-1][0] / 1000, tz=timezone.utc)
            print(f"[recent] latest 1m bar = {ts.isoformat()} (close={recent[-1][4]})")

        # 2. Walk back via history endpoint to find earliest bar.
        # Strategy: start from epoch ms = 0 (or 2018-01-01), keep stepping
        # forward until we get data, then refine.
        candidates = [
            ("2018-01-01", 1514764800000),
            ("2019-01-01", 1546300800000),
            ("2019-09-01", 1567296000000),
            ("2020-01-01", 1577836800000),
            ("2021-01-01", 1609459200000),
            ("2021-06-01", 1622505600000),
        ]

        first_hit = None
        for label, since_ms in candidates:
            await asyncio.sleep(0.3)  # rate-limit courtesy
            try:
                bars = await ex.fetch_ohlcv(
                    SYMBOL, TIMEFRAME, since=since_ms, limit=10,
                    params={"history": True},
                )
            except Exception as e:
                print(f"[probe {label}] error: {e}")
                continue
            if bars:
                ts = datetime.fromtimestamp(bars[0][0] / 1000, tz=timezone.utc)
                print(f"[probe since={label}] got {len(bars)} bars, first={ts.isoformat()}")
                if first_hit is None or bars[0][0] < first_hit[0]:
                    first_hit = (bars[0][0], ts)
            else:
                print(f"[probe since={label}] empty")

        # 3. Pull the very first historical batch with no `since`, in reverse
        # Some OKX history endpoints return earliest if you don't pass since.
        await asyncio.sleep(0.3)
        try:
            bars = await ex.fetch_ohlcv(
                SYMBOL, TIMEFRAME, since=1, limit=100,
                params={"history": True},
            )
            if bars:
                ts = datetime.fromtimestamp(bars[0][0] / 1000, tz=timezone.utc)
                print(f"[probe since=1] earliest from epoch query = {ts.isoformat()}")
                if first_hit is None or bars[0][0] < first_hit[0]:
                    first_hit = (bars[0][0], ts)
        except Exception as e:
            print(f"[probe since=1] error: {e}")

        if first_hit:
            print(f"\nEARLIEST 1m bar found: {first_hit[1].isoformat()}")
        else:
            print("\nNo earliest bar identified — investigate.")

        # 4. Measure rate-limit: 5 sequential calls, time delta
        print("\n[rate-limit] 5 sequential calls...")
        t0 = time.monotonic()
        for i in range(5):
            await ex.fetch_ohlcv(
                SYMBOL, TIMEFRAME, since=1622505600000, limit=100,
                params={"history": True},
            )
        elapsed = time.monotonic() - t0
        print(f"[rate-limit] 5 calls took {elapsed:.2f}s ({elapsed / 5:.2f}s/call avg)")

    finally:
        await ex.close()


if __name__ == "__main__":
    asyncio.run(probe())
