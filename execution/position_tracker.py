"""
포지션 추적기.
OKX에서 실시간 포지션을 조회하고 DB와 동기화한다.
멀티 심볼: 모든 오픈 트레이드의 심볼에 대해 포지션을 동기화한다.
"""
from datetime import datetime, timezone
from typing import Optional

import aiohttp
from loguru import logger

from config.settings import TradingConfig
from core.exceptions import APIError
from data.okx_rest_client import OKXRestClient
from database.db_manager import DatabaseManager
from database.models import TradeRecord


class PositionTracker:
    """OKX 포지션 ↔ DB 동기화 (멀티 심볼)"""

    def __init__(self, rest: OKXRestClient, db: DatabaseManager, config: TradingConfig):
        self._rest = rest
        self._db = db
        self._config = config

    async def sync_open_positions(self) -> list[TradeRecord]:
        """OKX에서 현재 열린 포지션을 조회해 DB 오픈 트레이드와 비교한다.
        외부에서 청산된 포지션을 감지하고 DB를 업데이트한다.

        Returns:
            현재 오픈 TradeRecord 목록
        """
        db_open = await self._db.fetch_open_trades()
        if not db_open:
            return []

        # DB에 열린 트레이드가 있는 심볼 목록 추출
        open_symbols = set(trade.symbol for trade in db_open)

        # 심볼별 OKX 포지션 조회
        okx_positions_map: dict[str, set[str]] = {}  # symbol → {LONG, SHORT}
        for symbol in open_symbols:
            try:
                okx_positions = await self._rest.fetch_positions(symbol)
                directions = set()
                for pos in okx_positions:
                    side = pos.get("side", "")
                    if side == "long":
                        directions.add("LONG")
                    elif side == "short":
                        directions.add("SHORT")
                okx_positions_map[symbol] = directions
            except (aiohttp.ClientError, APIError) as e:
                logger.error(f"[{symbol.split('/')[0]}] 포지션 동기화 실패: {e}")
                okx_positions_map[symbol] = set()  # 조회 실패 시 빈 집합 (외부 청산 감지 가능)

        # DB에는 OPEN이지만 OKX에 없는 포지션 → 외부 청산 감지
        for trade in db_open:
            okx_dirs = okx_positions_map.get(trade.symbol, set())
            if trade.direction not in okx_dirs:
                sym_tag = trade.symbol.split("/")[0]
                logger.warning(
                    f"[{sym_tag}] 외부 청산 감지: {trade.trade_id} {trade.direction} — DB 업데이트"
                )
                trade.status = "CLOSED"
                trade.exit_time = datetime.now(timezone.utc).isoformat()
                trade.exit_reason = "EXTERNAL"
                await self._db.update_trade(trade)

        return await self._db.fetch_open_trades()

    async def get_okx_position(self, symbol: str, direction: str) -> Optional[dict]:
        """OKX에서 특정 심볼/방향의 포지션 원시 데이터를 반환한다."""
        try:
            positions = await self._rest.fetch_positions(symbol)
            target_side = "long" if direction == "LONG" else "short"
            for pos in positions:
                if pos.get("side") == target_side:
                    return pos
        except (aiohttp.ClientError, APIError) as e:
            logger.error(f"OKX 포지션 조회 실패: {e}")
        return None

    async def get_unrealized_pnl(self, symbol: str, direction: str) -> float:
        """OKX에서 미실현 손익을 직접 조회한다."""
        pos = await self.get_okx_position(symbol, direction)
        if pos:
            return float(pos.get("unrealizedPnl", 0))
        return 0.0
