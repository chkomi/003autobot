"""
봇 상태 머신.
IDLE → WATCHING → (SIGNAL_DETECTED → IN_TRADE) → (WATCHING | HALTED | PAUSED_EVENT)

PAUSED_EVENT: 거시 이벤트/뉴스/변동성 스파이크로 신규 진입만 일시 차단한 상태.
              기존 포지션 모니터링은 계속되며, 조건 해제 시 자동 WATCHING 복귀.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from loguru import logger


class BotState(Enum):
    IDLE = "IDLE"                      # 초기화 중
    WATCHING = "WATCHING"              # 시그널 대기 중
    SIGNAL_DETECTED = "SIGNAL_DETECTED"  # 시그널 감지, 진입 검토 중
    IN_TRADE = "IN_TRADE"              # 포지션 보유 중
    PAUSED_EVENT = "PAUSED_EVENT"      # 이벤트/뉴스로 신규 진입 일시 중단
    HALTED = "HALTED"                  # 긴급 정지 (드로다운/치명적 장애)


class StateManager:
    def __init__(self):
        self._state = BotState.IDLE
        self._halt_reason: Optional[str] = None
        self._pause_reason: Optional[str] = None
        self._pause_since: Optional[datetime] = None
        self._halt_since: Optional[datetime] = None
        self._state_history: list[tuple[BotState, str]] = []

    @property
    def state(self) -> BotState:
        return self._state

    @property
    def halt_reason(self) -> Optional[str]:
        return self._halt_reason

    @property
    def pause_reason(self) -> Optional[str]:
        return self._pause_reason

    @property
    def halt_since(self) -> Optional[datetime]:
        return self._halt_since

    @property
    def pause_since(self) -> Optional[datetime]:
        return self._pause_since

    @property
    def is_running(self) -> bool:
        return self._state not in (BotState.IDLE, BotState.HALTED)

    @property
    def can_trade(self) -> bool:
        """신규 진입 가능 상태 (PAUSED_EVENT / HALTED 는 차단)."""
        return self._state in (BotState.WATCHING, BotState.SIGNAL_DETECTED)

    @property
    def is_halted(self) -> bool:
        return self._state == BotState.HALTED

    @property
    def is_paused(self) -> bool:
        return self._state == BotState.PAUSED_EVENT

    def transition(self, new_state: BotState, reason: str = "") -> None:
        old_state = self._state
        self._state = new_state
        self._state_history.append((new_state, reason))
        logger.info(f"상태 전환: {old_state.value} → {new_state.value}" + (f" ({reason})" if reason else ""))

    def halt(self, reason: str) -> None:
        self._halt_reason = reason
        self._halt_since = datetime.now(timezone.utc)
        self.transition(BotState.HALTED, reason)

    def resume(self, reason: str = "수동 재시작") -> None:
        """HALTED → WATCHING 복귀."""
        if self._state == BotState.HALTED:
            self._halt_reason = None
            self._halt_since = None
            self.transition(BotState.WATCHING, reason)

    def pause_for_event(self, reason: str) -> None:
        """거시 이벤트/뉴스/스파이크로 신규 진입만 일시 차단."""
        if self._state == BotState.HALTED:
            return
        self._pause_reason = reason
        self._pause_since = datetime.now(timezone.utc)
        self.transition(BotState.PAUSED_EVENT, reason)

    def unpause(self, reason: str = "이벤트 해제") -> None:
        """PAUSED_EVENT → WATCHING 복귀."""
        if self._state == BotState.PAUSED_EVENT:
            self._pause_reason = None
            self._pause_since = None
            self.transition(BotState.WATCHING, reason)