"""
포트폴리오 목표 추적기 (CAPR — Capital Allocation Portfolio Risk).

역할:
  - 월간 목표 수익률 달성 현황 추적 (10x 스케줄 기반)
  - 누적 수익률 이정표(milestone) 감지 및 텔레그램 알림
  - 4단계 Kelly 스케일 팩터 제공 (AHEAD/ON_PACE/BEHIND/CHASE)
  - KST 자정 nightly 업데이트 진입점

4단계 페이즈:
  AHEAD   : 월 목표 120% 이상 달성 → 수익 보호 (Kelly × 0.60)
  ON_PACE : 월 목표 80~119% 달성  → 정상 운영 (Kelly × 1.00)
  BEHIND  : 월 목표 50~79% 달성   → 약간 공격 (Kelly × 1.15)
  CHASE   : 50% 미만 + 월말 10일  → 추격 모드 (Kelly × 1.30)

사용 방법:
    tracker = GoalTracker(db, goal_cfg, notifier)
    await tracker.update_nightly(current_equity, "2026-04-24")   # bot_engine _feedback_loop
    scale = await tracker.get_kelly_scale(current_equity)         # position_sizer에 전달
"""
import math
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from loguru import logger

from config.settings import GoalConfig
from database.db_manager import DatabaseManager
from notification.notification_manager import NotificationManager

# KST
_KST = timezone(timedelta(hours=9))

# 추적할 누적 수익률 이정표 (%) — initial_capital 대비
_MILESTONES = [10.0, 25.0, 50.0, 100.0, 200.0, 300.0, 500.0]

# 10x 목표 상수
_10X_MONTHLY_RATE = 0.2144   # 10^(1/12) - 1
_10X_TARGET_MULTIPLIER = 10.0


class GoalPhase(str, Enum):
    """월간 목표 대비 달성 단계."""
    AHEAD = "AHEAD"       # 120% 이상 — 수익 보호
    ON_PACE = "ON_PACE"   # 80~119%  — 정상 운영
    BEHIND = "BEHIND"     # 50~79%   — 약간 공격적
    CHASE = "CHASE"       # 50% 미만 + 월말 10일 이내 — 추격


# 페이즈별 Kelly 스케일
_PHASE_KELLY_SCALE: dict[GoalPhase, float] = {
    GoalPhase.AHEAD: 0.60,
    GoalPhase.ON_PACE: 1.00,
    GoalPhase.BEHIND: 1.15,
    GoalPhase.CHASE: 1.30,
}

# 페이즈별 설명 (텔레그램용)
_PHASE_LABEL: dict[GoalPhase, str] = {
    GoalPhase.AHEAD: "🟢 목표 초과 달성 (수익 보호)",
    GoalPhase.ON_PACE: "🔵 정상 페이스",
    GoalPhase.BEHIND: "🟡 목표 미달 (약간 공격적)",
    GoalPhase.CHASE: "🔴 추격 모드 (월말 리스크 증가)",
}


class GoalTracker:
    """월간 목표 + 포트폴리오 이정표 추적"""

    def __init__(
        self,
        db: DatabaseManager,
        goal_cfg: GoalConfig,
        notifier: NotificationManager,
    ):
        self._db = db
        self._cfg = goal_cfg
        self._notifier = notifier

    # ── 공개 API ──────────────────────────────────────────────────

    async def update_nightly(self, current_equity: float, date_str: str) -> None:
        """KST 자정 직후 호출 (bot_engine _feedback_loop).

        date_str: 어제 KST 날짜 'YYYY-MM-DD' (자정 이후 어제 날짜를 집계).
        """
        month_str = date_str[:7]  # 'YYYY-MM'

        # 1. 월간 메트릭 초기화 (해당 월 최초 호출 시)
        await self._db.get_or_init_monthly_metric(
            month_str=month_str,
            start_equity=current_equity,
            target_pct=self._cfg.monthly_target_pct,
        )

        # 2. 어제 daily_pnl 조회 후 월간 메트릭 업데이트
        daily = await self._db.get_daily_pnl(date_str)
        await self._db.update_monthly_metric(
            month_str=month_str,
            end_equity=current_equity,
            daily_pnl=daily.pnl_usdt,
            daily_trades=daily.trade_count,
            daily_wins=daily.win_count,
        )

        # 3. 이정표 체크 (초과 달성한 milestone 알림)
        await self._check_milestones(current_equity)

        # 4. 월초 변환 체크 — 새 달이 시작된 경우 지난달 리포트 발송
        now_kst = datetime.now(_KST)
        if now_kst.day == 1:
            prev_month = (now_kst - timedelta(days=1)).strftime("%Y-%m")
            await self._send_monthly_close_report(prev_month, current_equity)

        logger.info(
            f"[GoalTracker] 야간 업데이트 완료: {date_str} | 자산 ${current_equity:,.2f} | 월: {month_str}"
        )

    async def get_kelly_scale(self, current_equity: float) -> float:
        """이번 달 목표 달성 단계(GoalPhase)에 따른 Kelly 스케일 팩터를 반환한다.

        4단계 페이즈:
          AHEAD   : 120%+    → 0.60 (수익 보호)
          ON_PACE : 80~119%  → 1.00 (정상)
          BEHIND  : 50~79%   → 1.15 (약간 공격)
          CHASE   : <50% + 월말 → 1.30 (추격)
        """
        phase = await self.get_current_phase(current_equity)
        scale = _PHASE_KELLY_SCALE[phase]
        logger.debug(f"[GoalTracker] 페이즈: {phase.value} → Kelly 스케일 {scale:.2f}")
        return scale

    async def get_current_phase(self, current_equity: float) -> GoalPhase:
        """현재 GoalPhase를 판단한다."""
        month_str = datetime.now(_KST).strftime("%Y-%m")
        day_of_month = datetime.now(_KST).day

        metrics_list = await self._db.fetch_monthly_metrics(limit=1)
        if not metrics_list or metrics_list[0]["month"] != month_str:
            return GoalPhase.ON_PACE

        m = metrics_list[0]
        target_pct = m.get("target_pct") or self._cfg.monthly_target_pct
        actual_pct = m.get("actual_pct") or 0.0

        if target_pct <= 0:
            return GoalPhase.ON_PACE

        progress = actual_pct / target_pct  # 1.0 = 100% 달성

        if progress >= 1.20:
            return GoalPhase.AHEAD
        elif progress >= 0.80:
            return GoalPhase.ON_PACE
        elif progress >= 0.50:
            return GoalPhase.BEHIND
        else:
            # 50% 미만: 월말 10일 이내면 CHASE, 아니면 BEHIND
            days_in_month = _days_in_month(datetime.now(_KST).year, datetime.now(_KST).month)
            days_left = days_in_month - day_of_month
            if days_left <= 10:
                return GoalPhase.CHASE
            return GoalPhase.BEHIND

    async def get_10x_schedule_status(self, current_equity: float) -> dict:
        """10x 스케줄 대비 현재 상태를 반환한다.

        Returns:
            {
              "elapsed_months": int,
              "schedule_equity": float,      # 이 시점의 목표 자산
              "actual_equity": float,
              "vs_schedule": float,          # actual / schedule
              "projected_months_to_10x": float,
              "on_track": bool,              # schedule 90% 이상이면 on track
            }
        """
        initial = self._cfg.initial_capital
        target = initial * _10X_TARGET_MULTIPLIER
        start_date_str = self._cfg.start_date

        if not start_date_str:
            return {"on_track": True, "vs_schedule": 1.0, "elapsed_months": 0,
                    "schedule_equity": initial, "actual_equity": current_equity,
                    "projected_months_to_10x": 12.0}

        try:
            start_dt = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=_KST)
        except ValueError:
            return {"on_track": True, "vs_schedule": 1.0, "elapsed_months": 0,
                    "schedule_equity": initial, "actual_equity": current_equity,
                    "projected_months_to_10x": 12.0}

        now = datetime.now(_KST)
        elapsed_months = (now.year - start_dt.year) * 12 + (now.month - start_dt.month)
        elapsed_months = max(0, elapsed_months)

        schedule_equity = initial * (1 + _10X_MONTHLY_RATE) ** elapsed_months
        vs_schedule = current_equity / schedule_equity if schedule_equity > 0 else 1.0

        # 현재까지 월 평균 수익률 추정
        if elapsed_months > 0 and initial > 0:
            monthly_actual = (current_equity / initial) ** (1 / elapsed_months) - 1
        else:
            monthly_actual = _10X_MONTHLY_RATE

        # 남은 기간 추정
        if monthly_actual > 0 and current_equity < target:
            remaining = math.log(target / current_equity) / math.log(1 + monthly_actual)
        elif current_equity >= target:
            remaining = 0.0
        else:
            remaining = float("inf")

        return {
            "elapsed_months": elapsed_months,
            "schedule_equity": round(schedule_equity, 2),
            "actual_equity": round(current_equity, 2),
            "vs_schedule": round(vs_schedule, 3),
            "projected_months_to_10x": round(remaining, 1) if remaining != float("inf") else 999.0,
            "on_track": vs_schedule >= 0.90,
        }

    async def get_portfolio_summary_html(self, current_equity: float) -> str:
        """텔레그램용 포트폴리오 현황 HTML 요약 (페이즈 + 10x 스케줄 포함)."""
        stats = await self._db.fetch_portfolio_stats(
            initial_capital=self._cfg.initial_capital,
            start_date=self._cfg.start_date or None,
        )
        month_str = datetime.now(_KST).strftime("%Y-%m")
        metrics_list = await self._db.fetch_monthly_metrics(limit=1)
        m = metrics_list[0] if metrics_list and metrics_list[0]["month"] == month_str else {}

        cum_ret = stats.get("cumulative_return", 0.0)
        cagr = stats.get("cagr", 0.0)
        elapsed = stats.get("elapsed_days", 0)
        initial = stats.get("initial_capital", self._cfg.initial_capital)

        monthly_actual = m.get("actual_pct", 0.0) or 0.0
        monthly_target = m.get("target_pct", self._cfg.monthly_target_pct) or self._cfg.monthly_target_pct
        monthly_progress = monthly_actual / monthly_target if monthly_target > 0 else 0
        monthly_pnl = m.get("pnl_usdt", 0.0) or 0.0

        progress_bar = _make_progress_bar(monthly_progress)
        cum_emoji = "🟢" if cum_ret >= 0 else "🔴"
        monthly_emoji = "🟢" if monthly_actual >= 0 else "🔴"

        # 현재 페이즈 + Kelly 스케일
        phase = await self.get_current_phase(current_equity)
        kelly_scale = _PHASE_KELLY_SCALE[phase]
        phase_label = _PHASE_LABEL[phase]

        # 10x 스케줄
        schedule = await self.get_10x_schedule_status(current_equity)
        vs_sched = schedule.get("vs_schedule", 1.0)
        sched_equity = schedule.get("schedule_equity", initial)
        proj_months = schedule.get("projected_months_to_10x", 12.0)
        on_track = schedule.get("on_track", True)
        sched_emoji = "✅" if on_track else "⚠️"

        lines = [
            "📊 <b>포트폴리오 현황</b>",
            "",
            f"💰 현재 자산: <b>${current_equity:,.2f}</b> USDT",
            f"🏁 시작 자본: ${initial:,.2f} | 운용 {elapsed}일",
            f"{cum_emoji} 누적 수익률: <b>{cum_ret:+.2%}</b> (연환산 {cagr:+.1%})",
            "",
            f"🎯 <b>10x 목표 스케줄</b> (${initial:,.0f} → ${initial*10:,.0f})",
            f"{sched_emoji} 스케줄 자산: ${sched_equity:,.2f} | 달성률: <b>{vs_sched:.1%}</b>",
        ]
        if proj_months < 999:
            lines.append(f"📈 현재 페이스로 10x 달성: <b>약 {proj_months:.1f}개월</b> 예상")
        else:
            lines.append("📈 10x 달성: 현재 수익률로는 어려움 — 전략 점검 필요")

        lines += [
            "",
            f"📅 <b>이번 달 목표</b>: {monthly_target:.1%}",
            f"{monthly_emoji} 달성: <b>{monthly_actual:+.2%}</b>  ({monthly_pnl:+.2f} USDT)",
            f"진행률: {progress_bar} {monthly_progress:.0%}",
            "",
            f"⚡ <b>운용 페이즈</b>: {phase_label}",
            f"   Kelly 스케일: ×{kelly_scale:.2f}",
        ]

        # 월간 기록
        win_m = stats.get("win_months", 0)
        loss_m = stats.get("loss_months", 0)
        achieved_m = stats.get("achieved_months", 0)
        total_m = stats.get("total_months", 0)
        if total_m > 0:
            lines.extend([
                "",
                f"📆 월간 기록: {win_m}수익 / {loss_m}손실 | 목표달성 {achieved_m}/{total_m}달",
            ])

        return "\n".join(lines)

    # ── 내부 헬퍼 ────────────────────────────────────────────────

    async def _check_milestones(self, current_equity: float) -> None:
        """새로 달성한 누적 수익률 이정표를 감지하고 알림을 보낸다."""
        initial = self._cfg.initial_capital
        if initial <= 0:
            return
        cum_ret_pct = (current_equity - initial) / initial * 100  # %값

        already_hit = set(await self._db.get_hit_milestones())
        for ms in _MILESTONES:
            if ms in already_hit:
                continue
            if cum_ret_pct >= ms:
                await self._db.record_milestone(ms, current_equity)
                msg = (
                    f"🏆 <b>이정표 달성!</b>\n"
                    f"누적 수익률 <b>+{ms:.0f}%</b> 도달\n"
                    f"시작 자본: ${initial:,.2f} → 현재: ${current_equity:,.2f} USDT\n"
                    f"수익금: ${current_equity - initial:,.2f}"
                )
                logger.info(f"[GoalTracker] 이정표 달성: +{ms:.0f}%")
                try:
                    await self._notifier.on_alert("INFO", msg)
                except Exception as e:
                    logger.error(f"이정표 알림 실패: {e}")

    async def _send_monthly_close_report(
        self, month_str: str, current_equity: float
    ) -> None:
        """전월 마감 리포트 발송 (매월 1일 자정 트리거)."""
        metrics_list = await self._db.fetch_monthly_metrics(limit=12)
        prev = next((m for m in metrics_list if m["month"] == month_str), None)
        if prev is None:
            return

        actual_pct = prev.get("actual_pct") or 0.0
        target_pct = prev.get("target_pct") or self._cfg.monthly_target_pct
        pnl_usdt = prev.get("pnl_usdt") or 0.0
        trades = prev.get("trade_count") or 0
        wins = prev.get("win_count") or 0
        achieved = prev.get("goal_achieved") == 1

        win_rate = wins / trades if trades > 0 else 0.0
        result_emoji = "✅" if achieved else "❌"

        msg = (
            f"📅 <b>{month_str} 월간 결산</b>\n"
            f"\n"
            f"수익률: <b>{actual_pct:+.2%}</b>  목표: {target_pct:.1%}  {result_emoji}\n"
            f"P&L: <b>{pnl_usdt:+,.2f} USDT</b>\n"
            f"거래: {trades}회 | 승률: {win_rate:.1%}\n"
            f"현재 자산: ${current_equity:,.2f}"
        )
        try:
            await self._notifier.on_alert("INFO", msg)
        except Exception as e:
            logger.error(f"월간 결산 알림 실패: {e}")


def _make_progress_bar(ratio: float, length: int = 10) -> str:
    """간단한 텍스트 프로그레스 바 생성. 예: [██████░░░░]"""
    ratio = max(0.0, min(ratio, 1.0))
    filled = round(ratio * length)
    return "[" + "█" * filled + "░" * (length - filled) + "]"


def _days_in_month(year: int, month: int) -> int:
    """주어진 연/월의 일 수를 반환한다."""
    import calendar
    return calendar.monthrange(year, month)[1]
