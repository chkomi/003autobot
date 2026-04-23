"""
봇 전체 설정을 담당하는 Pydantic Settings 모델.
.env 파일에서 자동으로 값을 로딩한다.
"""
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OKXConfig(BaseSettings):
    """OKX API 인증 및 연결 설정"""
    model_config = SettingsConfigDict(env_prefix="OKX_", env_file=".env", extra="ignore")

    api_key: str = Field(default="", description="OKX API Key")
    secret_key: str = Field(default="", description="OKX Secret Key")
    passphrase: str = Field(default="", description="OKX API Passphrase")
    is_demo: bool = Field(default=True, description="True = 페이퍼 트레이딩 모드")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key and self.passphrase)


class TradingConfig(BaseSettings):
    """트레이딩 파라미터 설정"""
    model_config = SettingsConfigDict(env_prefix="TRADING_", env_file=".env", extra="ignore")

    symbol: str = Field(default="BTC/USDT:USDT", description="(레거시) 단일 심볼 — symbols 우선")
    symbols: str = Field(default="", description="쉼표로 구분된 멀티 심볼 (예: BTC/USDT:USDT,GMT/USDT:USDT,SUI/USDT:USDT)")
    leverage: int = Field(default=3, ge=1, le=10, description="레버리지 배수 (기본 3x 격리마진)")
    max_position_pct: float = Field(default=0.10, gt=0, le=0.5, description="포지션당 최대 자본 비율")
    max_open_positions: int = Field(default=3, ge=1, le=5, description="최대 동시 포지션 수")

    # 타임프레임 설정 (전략 레이어별)
    timeframe_trend: str = Field(default="4h", description="Layer 1 추세 필터 타임프레임")
    timeframe_momentum: str = Field(default="1h", description="Layer 2 모멘텀 트리거 타임프레임")
    timeframe_micro: str = Field(default="15m", description="Layer 3 미시 확인 타임프레임")

    # 루프 인터벌
    signal_check_interval_sec: int = Field(default=60, description="시그널 체크 주기 (초)")
    position_monitor_interval_sec: int = Field(default=10, description="포지션 모니터 주기 (초)")
    candle_fetch_lookback: int = Field(default=300, description="캔들 조회 개수")
    max_trade_duration_hours: int = Field(default=48, description="최대 포지션 보유 시간")

    @property
    def symbol_list(self) -> List[str]:
        """거래 대상 심볼 목록. TRADING_SYMBOLS가 있으면 우선, 없으면 TRADING_SYMBOL 사용."""
        if self.symbols.strip():
            return [s.strip() for s in self.symbols.split(",") if s.strip()]
        return [self.symbol]

    @property
    def all_timeframes(self) -> List[str]:
        return [self.timeframe_trend, self.timeframe_momentum, self.timeframe_micro]

    @staticmethod
    def to_okx_inst_id(ccxt_symbol: str) -> str:
        """ccxt 심볼을 OKX WebSocket instId로 변환 (예: BTC/USDT:USDT → BTC-USDT-SWAP)"""
        base = ccxt_symbol.split("/")[0]
        return f"{base}-USDT-SWAP"

    @property
    def okx_symbol(self) -> str:
        """(레거시) 첫 번째 심볼의 OKX instId"""
        return self.to_okx_inst_id(self.symbol_list[0])


class RiskConfig(BaseSettings):
    """리스크 관리 설정"""
    model_config = SettingsConfigDict(env_prefix="RISK_", env_file=".env", extra="ignore")

    daily_loss_limit_pct: float = Field(default=0.05, gt=0, le=0.2, description="일일 손실 한도 비율")
    weekly_loss_limit_pct: float = Field(default=0.10, gt=0, le=0.5, description="주간 손실 한도 비율")
    max_drawdown_pct: float = Field(default=0.15, gt=0, le=0.5, description="최대 드로다운 한도 비율")
    kelly_fraction: float = Field(default=0.25, gt=0, le=1.0, description="Kelly Criterion 적용 비율")

    # ATR 기반 SL/TP 설정
    atr_period: int = Field(default=14, description="ATR 계산 기간")
    atr_sl_multiplier: float = Field(default=1.0, description="SL 거리 = ATR × 이 값")
    atr_tp1_multiplier: float = Field(default=2.0, description="TP1 거리 = ATR × 이 값")
    atr_tp2_multiplier: float = Field(default=3.5, description="TP2 거리 = ATR × 이 값")
    tp1_exit_pct: float = Field(default=0.5, description="TP1 도달 시 청산 비율")
    tp2_exit_pct: float = Field(default=0.3, description="TP2 도달 시 추가 청산 비율")
    partial_exit_pct: float = Field(default=0.5, description="(legacy) TP1 청산 비율")
    trailing_atr_multiplier: float = Field(default=2.0, description="트레일링 ATR 승수")


class Settings(BaseSettings):
    """최상위 설정 - 모든 서브설정을 포함"""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Telegram
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    # Discord
    discord_webhook_url: str = Field(default="", alias="DISCORD_WEBHOOK_URL")

    # System
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    db_path: Path = Field(default=Path("data_cache/trading.db"), alias="DB_PATH")
    dashboard_host: str = Field(default="0.0.0.0", alias="DASHBOARD_HOST")
    dashboard_port: int = Field(default=8000, alias="DASHBOARD_PORT")

    # Sub-configs (단독 로딩)
    okx: OKXConfig = Field(default_factory=OKXConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)

    @property
    def telegram_configured(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def discord_configured(self) -> bool:
        return bool(self.discord_webhook_url)


def _check_env_permissions() -> None:
    """`.env` 파일의 권한이 너무 열려 있으면 경고한다."""
    import os
    import stat
    from loguru import logger

    env_path = Path(".env")
    if not env_path.exists():
        return
    try:
        mode = env_path.stat().st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            logger.warning(
                f".env 파일 권한이 너무 열려 있습니다 ({oct(mode & 0o777)}). "
                f"chmod 600 .env 를 권장합니다."
            )
    except OSError:
        pass


def load_settings() -> Settings:
    """설정 인스턴스를 로딩한다. 앱 전체에서 이 함수를 사용할 것."""
    _check_env_permissions()
    return Settings()
