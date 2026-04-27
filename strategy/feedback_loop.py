"""
피드백 루프 모듈.
매매 성과를 분석하고 스코어링 파라미터 조정을 제안한다.
주기적으로 실행되어 전략의 자가 개선을 지원한다.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from loguru import logger

from database.db_manager import DatabaseManager


@dataclass
class PerformanceReport:
    """주간/일간 성과 리포트"""
    period: str                 # "daily" | "weekly"
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_consecutive_losses: int = 0
    sharpe_approx: float = 0.0
    max_drawdown_pct: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0

    # 스코어 기준별 분석
    high_score_win_rate: float = 0.0  # 75점 이상 시그널 승률
    low_score_win_rate: float = 0.0   # 60~75점 시그널 승률

    # 시간대별 분석
    best_hour_utc: Optional[int] = None
    worst_hour_utc: Optional[int] = None

    # 파라미터 조정 제안
    suggestions: list = None

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []

    def summary(self) -> str:
        lines = [
            f"=== {self.period.upper()} 성과 리포트 ===",
            f"총 거래: {self.total_trades} (승: {self.wins}, 패: {self.losses})",
            f"승률: {self.win_rate:.1%}",
            f"총 수익: ${self.total_pnl:,.2f}",
            f"평균 수익/거래: ${self.avg_pnl_per_trade:,.2f}",
            f"Profit Factor: {self.profit_factor:.2f}",
            f"최대 연속 손실: {self.max_consecutive_losses}회",
            f"샤프(근사): {self.sharpe_approx:.2f}",
            f"MDD: {self.max_drawdown_pct:.2%}",
        ]
        if self.suggestions:
            lines.append("\n[조정 제안]")
            for s in self.suggestions:
                lines.append(f"  • {s}")
        return "\n".join(lines)

    def to_telegram_html(self) -> str:
        """텔레그램 알림용 HTML 포맷"""
        pnl_emoji = "+" if self.total_pnl >= 0 else ""
        lines = [
            f"<b>{self.period.upper()} 성과 리포트</b>",
            f"거래: {self.total_trades}회 | 승률: {self.win_rate:.1%}",
            f"수익: {pnl_emoji}${self.total_pnl:,.2f}",
            f"PF: {self.profit_factor:.2f} | 샤프: {self.sharpe_approx:.2f}",
            f"MDD: {self.max_drawdown_pct:.2%}",
        ]
        if self.high_score_win_rate > 0 or self.low_score_win_rate > 0:
            lines.append(
                f"고스코어 승률: {self.high_score_win_rate:.1%} | "
                f"저스코어 승률: {self.low_score_win_rate:.1%}"
            )
        if self.suggestions:
            lines.append("\n<b>조정 제안:</b>")
            for s in self.suggestions:
                lines.append(f"• {s}")
        return "\n".join(lines)


class FeedbackLoop:
    """매매 성과 분석 및 파라미터 조정 제안 엔진"""

    def __init__(self, db: DatabaseManager):
        self._db = db

    # KST = UTC+9
    _KST = timezone(timedelta(hours=9))

    async def generate_daily_report(self, date_str: Optional[str] = None) -> PerformanceReport:
        """오늘(또는 지정 KST 날짜)의 성과 리포트를 생성한다."""
        date_str = date_str or datetime.now(self._KST).strftime("%Y-%m-%d")
        return await self._generate_report("daily", date_str=date_str)

    async def generate_weekly_report(self) -> PerformanceReport:
        """주간 성과 리포트를 생성한다. (전체 누적 통계 사용)"""
        return await self._generate_report("weekly")

    async def _generate_report(self, period: str, date_str: Optional[str] = None) -> PerformanceReport:
        """기간별 성과 리포트 생성"""
        report = PerformanceReport(period=period)

        # 기본 통계 조회 — daily는 해당 KST 날짜의 거래만, weekly는 전체
        if period == "daily" and date_str:
            stats = await self._db.fetch_trade_stats_for_date(date_str)
        else:
            stats = await self._db.fetch_trade_stats()
        if not stats or stats.get("total", 0) == 0:
            report.suggestions.append("거래 데이터가 없습니다. 전략 실행 후 분석 가능합니다.")
            return report

        total = stats.get("total", 0)
        wins = stats.get("wins", 0) or 0
        losses = stats.get("losses", 0) or 0

        report.total_trades = total
        report.wins = wins
        report.losses = losses
        report.win_rate = wins / total if total > 0 else 0
        report.total_pnl = stats.get("total_pnl", 0) or 0
        report.avg_pnl_per_trade = report.total_pnl / total if total > 0 else 0
        report.avg_win = abs(stats.get("avg_win", 0) or 0)
        report.avg_loss = abs(stats.get("avg_loss", 0) or 0)
        report.best_trade = stats.get("best_trade", 0) or 0
        report.worst_trade = stats.get("worst_trade", 0) or 0

        # Profit Factor
        total_wins = report.avg_win * wins if wins > 0 else 0
        total_losses = report.avg_loss * losses if losses > 0 else 0
        report.profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")

        # 조정 제안 생성
        self._generate_suggestions(report)

        logger.info(f"[피드백] {period} 리포트 생성: {total}건, 승률 {report.win_rate:.1%}")
        return report

    def _generate_suggestions(self, report: PerformanceReport) -> None:
        """성과 기반 파라미터 조정 제안"""
        suggestions = report.suggestions

        # 1. 승률 기반 제안
        if report.total_trades >= 10:
            if report.win_rate < 0.40:
                suggestions.append(
                    "승률이 40% 미만입니다. "
                    "strong_signal_threshold를 80으로 올려 진입 기준을 강화하세요."
                )
            elif report.win_rate > 0.65:
                suggestions.append(
                    "승률이 높습니다. "
                    "weak_signal_threshold를 55로 낮춰 거래 빈도를 늘릴 수 있습니다."
                )

        # 2. Profit Factor 기반 제안
        if report.total_trades >= 10:
            if report.profit_factor < 1.0:
                suggestions.append(
                    f"Profit Factor가 {report.profit_factor:.2f}로 손실 구간입니다. "
                    "SL을 좁히거나 TP를 늘려 손익비를 개선하세요."
                )
            elif report.profit_factor < 1.5:
                suggestions.append(
                    f"Profit Factor {report.profit_factor:.2f}. "
                    "atr_tp1_multiplier를 2.5로, atr_sl_multiplier를 0.8로 조정 검토."
                )

        # 3. 연속 손실 기반 제안
        if report.max_consecutive_losses >= 5:
            suggestions.append(
                f"최대 {report.max_consecutive_losses}연패. "
                "시장 레짐이 변했을 수 있습니다. 전략 일시 중지 후 재검토를 권합니다."
            )

        # 4. 평균 손실 vs 평균 이익 비율
        if report.avg_loss > 0 and report.avg_win > 0:
            rr = report.avg_win / report.avg_loss
            if rr < 1.5:
                suggestions.append(
                    f"손익비({rr:.2f})가 낮습니다. "
                    "atr_tp1_multiplier 상향 또는 atr_sl_multiplier 하향을 검토하세요."
                )

        # 5. 고/저 스코어 승률 비교
        if report.high_score_win_rate > 0 and report.low_score_win_rate > 0:
            if report.low_score_win_rate < 0.35:
                suggestions.append(
                    f"저스코어(60~75) 시그널 승률이 {report.low_score_win_rate:.1%}로 낮습니다. "
                    "weak_signal_threshold를 70으로 올려 저품질 시그널을 제거하세요."
                )

        if not suggestions:
            suggestions.append("현재 전략 성과가 양호합니다. 파라미터 유지를 권합니다.")

    async def auto_tune(self, min_trades: int = 20) -> dict:
        """성과 기반 파라미터 자동 조정.

        최근 거래 데이터를 분석하여 strategy_params.yaml의 파라미터를
        안전 범위 내에서 자동 조정한다.

        Args:
            min_trades: 자동 조정에 필요한 최소 거래 수

        Returns:
            변경된 파라미터 딕셔너리 (변경 없으면 빈 딕셔너리)
        """
        from pathlib import Path
        import yaml

        stats = await self._db.fetch_trade_stats()
        total = stats.get("total", 0)

        if total < min_trades:
            logger.info(f"[자동 튜닝] 거래 수 부족 ({total}/{min_trades}) — 스킵")
            return {}

        wins = stats.get("wins", 0) or 0
        win_rate = wins / total if total > 0 else 0
        avg_win = abs(stats.get("avg_win", 0) or 0)
        avg_loss = abs(stats.get("avg_loss", 0) or 0)

        # 현재 파라미터 로드
        yaml_path = Path(__file__).parent.parent / "config" / "strategy_params.yaml"
        with open(yaml_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        changes = {}
        MAX_CHANGE_PCT = 0.20  # 1회 조정 최대 변경폭 ±20%

        # 1. 승률 기반 시그널 임계값 조정
        current_weak = cfg.get("scoring", {}).get("weak_signal_threshold", 70)
        if win_rate < 0.40:
            new_weak = min(current_weak + 5, 85)
            if new_weak != current_weak:
                cfg["scoring"]["weak_signal_threshold"] = new_weak
                changes["weak_signal_threshold"] = f"{current_weak} → {new_weak} (승률 {win_rate:.1%} < 40%)"
        elif win_rate > 0.65 and total >= 30:
            new_weak = max(current_weak - 5, 55)
            if new_weak != current_weak:
                cfg["scoring"]["weak_signal_threshold"] = new_weak
                changes["weak_signal_threshold"] = f"{current_weak} → {new_weak} (승률 {win_rate:.1%} > 65%)"

        # 2. 손익비 기반 SL/TP 조정
        if avg_loss > 0:
            rr_ratio = avg_win / avg_loss
            current_sl = cfg.get("risk", {}).get("atr_sl_multiplier", 1.5)
            current_tp1 = cfg.get("risk", {}).get("atr_tp1_multiplier", 2.0)

            if rr_ratio < 1.2 and win_rate < 0.50:
                # 손익비 낮음: SL 좁히고 TP 확대
                new_sl = round(max(current_sl * 0.9, 0.8), 2)
                new_tp1 = round(min(current_tp1 * 1.1, 3.0), 2)
                if abs(new_sl - current_sl) / current_sl <= MAX_CHANGE_PCT:
                    cfg["risk"]["atr_sl_multiplier"] = new_sl
                    changes["atr_sl_multiplier"] = f"{current_sl} → {new_sl} (손익비 {rr_ratio:.2f} < 1.2)"
                if abs(new_tp1 - current_tp1) / current_tp1 <= MAX_CHANGE_PCT:
                    cfg["risk"]["atr_tp1_multiplier"] = new_tp1
                    changes["atr_tp1_multiplier"] = f"{current_tp1} → {new_tp1}"
            elif rr_ratio > 2.5 and win_rate < 0.35:
                # 손익비 높지만 승률 너무 낮음: SL 넓혀서 승률 개선
                new_sl = round(min(current_sl * 1.1, 2.5), 2)
                if abs(new_sl - current_sl) / current_sl <= MAX_CHANGE_PCT:
                    cfg["risk"]["atr_sl_multiplier"] = new_sl
                    changes["atr_sl_multiplier"] = f"{current_sl} → {new_sl} (승률 {win_rate:.1%} 개선 시도)"

        # 3. 변경 적용
        if changes:
            # 백업 (이전 설정 보존)
            backup_path = yaml_path.with_suffix(".yaml.bak")
            with open(yaml_path, encoding="utf-8") as f:
                backup_content = f.read()
            with open(backup_path, "w", encoding="utf-8") as f:
                f.write(backup_content)

            # 새 설정 저장
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

            logger.info(f"[자동 튜닝] 파라미터 변경 적용: {changes}")
            logger.info(f"[자동 튜닝] 이전 설정 백업: {backup_path}")
        else:
            logger.info("[자동 튜닝] 변경 사항 없음 — 현재 파라미터 유지")

        return changes

    async def rollback(self) -> bool:
        """이전 파라미터로 롤백한다.

        Returns:
            롤백 성공 여부
        """
        from pathlib import Path
        import shutil

        yaml_path = Path(__file__).parent.parent / "config" / "strategy_params.yaml"
        backup_path = yaml_path.with_suffix(".yaml.bak")

        if not backup_path.exists():
            logger.warning("[롤백] 백업 파일 없음 — 롤백 불가")
            return False

        shutil.copy2(backup_path, yaml_path)
        logger.info(f"[롤백] 이전 설정 복원: {backup_path} → {yaml_path}")
        return True
