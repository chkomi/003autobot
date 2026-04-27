"""
봇 전체에서 사용하는 커스텀 예외 계층.
모든 예외는 BotError를 상속한다.
"""


class BotError(Exception):
    """봇 베이스 예외"""
    pass


# ── API / 연결 관련 ────────────────────────────────────────────
class APIError(BotError):
    """OKX API 호출 실패"""
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(APIError):
    """API 레이트 리밋 초과"""
    pass


class AuthenticationError(APIError):
    """API 인증 실패 (키 오류, passphrase 불일치 등)"""
    pass


class WebSocketError(BotError):
    """WebSocket 연결/메시지 오류"""
    pass


# ── 데이터 관련 ───────────────────────────────────────────────
class DataError(BotError):
    """시장 데이터 수집/처리 오류"""
    pass


class InsufficientDataError(DataError):
    """지표 계산에 필요한 캔들 수 부족"""
    def __init__(self, required: int, available: int, timeframe: str):
        super().__init__(
            f"지표 계산 불가: {timeframe} 캔들 {required}개 필요, {available}개만 사용 가능"
        )
        self.required = required
        self.available = available
        self.timeframe = timeframe


# ── 전략 관련 ─────────────────────────────────────────────────
class StrategyError(BotError):
    """전략 시그널 생성 오류"""
    pass


# ── 리스크 관련 ───────────────────────────────────────────────
class RiskError(BotError):
    """리스크 검증 오류"""
    pass


class DailyLossLimitError(RiskError):
    """일일 손실 한도 초과"""
    def __init__(self, current_loss_pct: float, limit_pct: float):
        super().__init__(
            f"일일 손실 한도 초과: 현재 {current_loss_pct:.2%} / 한도 {limit_pct:.2%}"
        )
        self.current_loss_pct = current_loss_pct
        self.limit_pct = limit_pct


class MaxDrawdownError(RiskError):
    """최대 드로다운 한도 초과 - 봇 자동 정지"""
    def __init__(self, current_dd_pct: float, limit_pct: float):
        super().__init__(
            f"최대 드로다운 초과: 현재 {current_dd_pct:.2%} / 한도 {limit_pct:.2%} — 봇 정지"
        )
        self.current_dd_pct = current_dd_pct
        self.limit_pct = limit_pct


class InsufficientBalanceError(RiskError):
    """잔고 부족"""
    def __init__(self, required_usdt: float, available_usdt: float):
        super().__init__(
            f"잔고 부족: {required_usdt:.2f} USDT 필요, {available_usdt:.2f} USDT 사용 가능"
        )
        self.required_usdt = required_usdt
        self.available_usdt = available_usdt


class MaxPositionsError(RiskError):
    """최대 포지션 수 초과"""
    def __init__(self, current: int, maximum: int):
        super().__init__(f"최대 포지션 초과: 현재 {current}개 / 최대 {maximum}개")
        self.current = current
        self.maximum = maximum


# ── 주문/실행 관련 ─────────────────────────────────────────────
class OrderError(BotError):
    """주문 생성/취소/수정 오류"""
    pass


class OrderNotFoundError(OrderError):
    """주문 ID를 찾을 수 없음"""
    def __init__(self, order_id: str):
        super().__init__(f"주문을 찾을 수 없음: {order_id}")
        self.order_id = order_id


class ClosePositionFailedError(OrderError):
    """청산 주문 최대 재시도 후 전량 실패 — 거래소 포지션 수동 확인 필요.

    fallback_price: 내부 P&L 계산에 사용할 현재가 폴백 값.
    거래소에 포지션이 여전히 열려있을 수 있으므로 반드시 텔레그램 긴급 알림 발송 필요.
    """
    def __init__(self, symbol: str, trade_id: str, fallback_price: float):
        super().__init__(
            f"청산 완전 실패 ({symbol} / {trade_id}): 거래소 포지션이 아직 열려있을 수 있음 "
            f"— 폴백 가격 ${fallback_price:,.2f} 적용"
        )
        self.fallback_price = fallback_price
        self.symbol = symbol
        self.trade_id = trade_id


class PositionError(BotError):
    """포지션 조회/관리 오류"""
    pass


# ── 시스템 관련 ───────────────────────────────────────────────
class ConfigError(BotError):
    """설정 오류 (필수 환경변수 누락 등)"""
    pass


class DatabaseError(BotError):
    """DB 읽기/쓰기 오류"""
    pass


class NotificationError(BotError):
    """알림 전송 오류 (Telegram 등)"""
    pass


class BotHaltError(BotError):
    """봇 긴급 정지 신호"""
    def __init__(self, reason: str):
        super().__init__(f"봇 긴급 정지: {reason}")
        self.reason = reason
