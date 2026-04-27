"""
포지션 사이즈 계산기.
25% Kelly Criterion + 하드 리밋 적용.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

from config.settings import TradingConfig, RiskConfig
from core.exceptions import InsufficientBalanceError


def _load_symbol_settings() -> dict:
    """strategy_params.yaml에서 심볼별 수량 설정을 로드한다."""
    yaml_path = Path(__file__).parent.parent / "config" / "strategy_params.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("symbol_settings", {})


@dataclass
class PositionSize:
    quantity: float        # BTC 수량
    notional_usdt: float   # 명목 가치 (quantity × price)
    leverage: int
    margin_required: float # 실제 증거금 = notional / leverage


class PositionSizer:
    """리스크 기반 포지션 사이즈 계산"""

    # BTC 전용 기본값 (yaml에 설정이 없는 심볼 폴백)
    _DEFAULT_QTY_PRECISION = 4
    _DEFAULT_MIN_QTY = 0.001

    def __init__(self, trading_cfg: TradingConfig, risk_cfg: RiskConfig):
        self._t = trading_cfg
        self._r = risk_cfg
        self._symbol_settings: dict = _load_symbol_settings()

    def _get_qty_params(self, symbol: str) -> tuple[int, float]:
        """심볼별 qty_precision과 min_qty를 반환한다."""
        cfg = self._symbol_settings.get(symbol, {})
        precision = cfg.get("qty_precision", self._DEFAULT_QTY_PRECISION)
        min_qty = cfg.get("min_qty", self._DEFAULT_MIN_QTY)
        return precision, min_qty

    def calculate(
        self,
        balance_usdt: float,
        entry_price: float,
        stop_price: float,
        symbol: str = "BTC/USDT:USDT",
        win_rate: Optional[float] = None,
        avg_win_ratio: Optional[float] = None,
        avg_loss_ratio: Optional[float] = None,
        drawdown_pct: float = 0.0,
        goal_kelly_scale: float = 1.0,
    ) -> PositionSize:
        """포지션 사이즈를 계산한다.

        Args:
            balance_usdt: 현재 사용 가능 잔고
            entry_price: 진입 예상가
            stop_price: 손절 가격
            win_rate, avg_win_ratio, avg_loss_ratio: Kelly 계산용 성과 데이터
                (None이면 max_position_pct 기반 고정 사이징 사용)

        Returns:
            PositionSize
        """
        if balance_usdt <= 0:
            raise InsufficientBalanceError(1.0, balance_usdt)

        # 1. Kelly 기반 사이징 (과거 성과 데이터가 있을 때)
        if win_rate is not None and avg_win_ratio is not None and avg_loss_ratio is not None:
            kelly_f = self._kelly_fraction(win_rate, avg_win_ratio, avg_loss_ratio)
        else:
            # 초기 운영: 고정 max_position_pct 사용
            kelly_f = self._r.kelly_fraction

        # 드로다운 기반 포지션 축소
        if drawdown_pct > 0:
            max_dd = self._r.max_drawdown_pct or 0.15
            scale = max(1.0 - (drawdown_pct / max_dd), 0.2)  # 최소 20% 유지
            kelly_f *= scale
            logger.debug(f"드로다운 스케일링: DD={drawdown_pct:.2%}, 배수={scale:.2f}, 조정Kelly={kelly_f:.4f}")

        # 목표 달성률 기반 포지션 조정 (GoalTracker 제공)
        if goal_kelly_scale != 1.0:
            kelly_f *= goal_kelly_scale
            logger.debug(f"목표 달성 스케일링: 배수={goal_kelly_scale:.2f}, 조정Kelly={kelly_f:.4f}")

        # 2. 리스크 기반 사이징 (손절폭 대비 자본 비율)
        risk_pct = abs(entry_price - stop_price) / entry_price  # SL 거리 비율
        if risk_pct == 0:
            logger.warning("SL 거리가 0 — max_position_pct 사용")
            risk_pct = 0.01

        # 허용 손실 금액 = 자본 × Kelly 비율
        risk_amount_usdt = balance_usdt * kelly_f
        # 해당 손실을 견딜 수 있는 포지션 명목 가치
        notional_from_risk = risk_amount_usdt / risk_pct

        # 3. 하드 리밋 적용
        max_notional = balance_usdt * self._t.max_position_pct * self._t.leverage
        notional = min(notional_from_risk, max_notional)

        # 4. 잔고 기반 최대치 재검증
        max_margin = balance_usdt * self._t.max_position_pct
        margin_required = notional / self._t.leverage
        if margin_required > max_margin:
            margin_required = max_margin
            notional = margin_required * self._t.leverage

        # 5. 심볼별 수량 계산 (qty_precision, min_qty를 yaml 설정에서 읽음)
        qty_precision, min_qty = self._get_qty_params(symbol)
        quantity = round(notional / entry_price, qty_precision)
        quantity = max(quantity, min_qty)

        # 최종 검증
        actual_margin = (quantity * entry_price) / self._t.leverage
        if actual_margin > balance_usdt * 0.95:
            raise InsufficientBalanceError(actual_margin, balance_usdt)

        logger.debug(
            f"포지션 사이즈: {quantity} {symbol} | "
            f"명목: ${quantity * entry_price:,.2f} | "
            f"증거금: ${actual_margin:,.2f} | "
            f"Kelly: {kelly_f:.3f}"
        )

        return PositionSize(
            quantity=quantity,
            notional_usdt=quantity * entry_price,
            leverage=self._t.leverage,
            margin_required=actual_margin,
        )

    def _kelly_fraction(
        self,
        win_rate: float,
        avg_win_ratio: float,
        avg_loss_ratio: float,
    ) -> float:
        """25% Kelly Criterion 계산.

        Returns:
            0~max_position_pct 사이의 비율 (보수적 상한 적용)
        """
        if avg_win_ratio == 0:
            return self._t.max_position_pct * 0.5

        full_kelly = (win_rate * avg_win_ratio - (1 - win_rate) * avg_loss_ratio) / avg_win_ratio
        adjusted = full_kelly * self._r.kelly_fraction  # 25% Kelly

        # 음수(기대값 손실) 또는 과도한 값 방지
        return max(0.01, min(adjusted, self._t.max_position_pct))
