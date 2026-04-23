"""
DB 모델 정의 (dataclass 기반).
SQLite 로우 ↔ Python 객체 변환을 담당한다.
"""
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_trade_id() -> str:
    return f"TRD-{uuid.uuid4().hex[:12].upper()}"


@dataclass
class TradeRecord:
    """거래 기록 모델"""
    symbol: str
    direction: str            # "LONG" | "SHORT"
    quantity: float
    leverage: int

    trade_id: str = field(default_factory=_new_trade_id)
    status: str = "OPEN"     # OPEN | CLOSED | CANCELLED

    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None

    pnl_usdt: Optional[float] = None
    pnl_pct: Optional[float] = None

    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    exit_reason: Optional[str] = None  # TP1/TP2/SL/TRAILING/TIME/MANUAL/HALT

    signal_confidence: Optional[float] = None
    atr_at_entry: Optional[float] = None

    sl_algo_id: Optional[str] = None   # OKX 알고 주문 ID (취소용)
    tp_algo_id: Optional[str] = None

    created_at: str = field(default_factory=_now_utc)

    @property
    def is_open(self) -> bool:
        return self.status == "OPEN"

    @property
    def realized_pnl(self) -> float:
        if self.pnl_usdt is not None:
            return self.pnl_usdt
        return 0.0

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "direction": self.direction,
            "status": self.status,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "leverage": self.leverage,
            "stop_loss": self.stop_loss,
            "take_profit_1": self.take_profit_1,
            "take_profit_2": self.take_profit_2,
            "pnl_usdt": self.pnl_usdt,
            "pnl_pct": self.pnl_pct,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "exit_reason": self.exit_reason,
            "signal_confidence": self.signal_confidence,
            "atr_at_entry": self.atr_at_entry,
            "sl_algo_id": self.sl_algo_id,
            "tp_algo_id": self.tp_algo_id,
            "created_at": self.created_at,
        }

    @classmethod
    def from_row(cls, row: dict) -> "TradeRecord":
        return cls(
            trade_id=row["trade_id"],
            symbol=row["symbol"],
            direction=row["direction"],
            status=row["status"],
            entry_price=row.get("entry_price"),
            exit_price=row.get("exit_price"),
            quantity=row["quantity"],
            leverage=row["leverage"],
            stop_loss=row.get("stop_loss"),
            take_profit_1=row.get("take_profit_1"),
            take_profit_2=row.get("take_profit_2"),
            pnl_usdt=row.get("pnl_usdt"),
            pnl_pct=row.get("pnl_pct"),
            entry_time=row.get("entry_time"),
            exit_time=row.get("exit_time"),
            exit_reason=row.get("exit_reason"),
            signal_confidence=row.get("signal_confidence"),
            atr_at_entry=row.get("atr_at_entry"),
            sl_algo_id=row.get("sl_algo_id"),
            tp_algo_id=row.get("tp_algo_id"),
            created_at=row.get("created_at", _now_utc()),
        )


@dataclass
class DailyPnL:
    """일일 P&L 집계 모델"""
    date: str
    pnl_usdt: float = 0.0
    trade_count: int = 0
    win_count: int = 0
    peak_equity: Optional[float] = None
    min_equity: Optional[float] = None
    updated_at: str = field(default_factory=_now_utc)

    @property
    def win_rate(self) -> float:
        if self.trade_count == 0:
            return 0.0
        return self.win_count / self.trade_count


@dataclass
class EquitySnapshot:
    """계좌 자산 스냅샷"""
    equity: float
    free: Optional[float] = None
    used: Optional[float] = None
    created_at: str = field(default_factory=_now_utc)


@dataclass
class BotEvent:
    """봇 이벤트 로그"""
    event_type: str   # SIGNAL/ORDER/HALT/ALERT/INFO
    level: str        # INFO/WARNING/ERROR/CRITICAL
    message: str
    metadata: Optional[dict] = None
    created_at: str = field(default_factory=_now_utc)

    def metadata_json(self) -> Optional[str]:
        if self.metadata:
            return json.dumps(self.metadata, ensure_ascii=False)
        return None
