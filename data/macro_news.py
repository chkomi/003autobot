"""
거시경제 매크로 뉴스 데이터 수집기.
주요 경제 이벤트(FOMC, CPI, NFP 등)와 크립토 Fear & Greed Index를 수집해
전략 의사결정에 반영한다.
"""
import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiohttp
from loguru import logger


# ── 데이터 모델 ──────────────────────────────────────────────

@dataclass
class MacroEvent:
    """거시경제 이벤트"""
    title: str
    datetime_utc: datetime
    impact: str  # "HIGH" | "MEDIUM" | "LOW"
    currency: str = "USD"
    actual: Optional[str] = None
    forecast: Optional[str] = None
    previous: Optional[str] = None

    @property
    def is_high_impact(self) -> bool:
        return self.impact == "HIGH"

    @property
    def hours_until(self) -> float:
        now = datetime.now(timezone.utc)
        delta = self.datetime_utc - now
        return delta.total_seconds() / 3600


@dataclass
class MacroSnapshot:
    """매크로 환경 스냅샷 — 스코어링에 사용"""
    fear_greed_index: Optional[int] = None       # 0~100 (0=극단공포, 100=극단탐욕)
    fear_greed_label: Optional[str] = None       # "Extreme Fear" 등
    upcoming_high_impact_count: int = 0          # 24시간 내 고영향 이벤트 수
    nearest_high_impact_hours: Optional[float] = None  # 가장 가까운 고영향 이벤트까지 시간
    nearest_event_title: Optional[str] = None
    btc_dominance: Optional[float] = None        # BTC 도미넌스 (%)
    total_market_cap_change_24h: Optional[float] = None  # 전체 시장 시총 변화율
    events: list = field(default_factory=list)    # MacroEvent 목록
    # 뉴스 기반 위험도 (news_collector.NewsCollector가 채움)
    news_risk_level: str = "NORMAL"              # NORMAL | ELEVATED | HIGH | CRITICAL
    news_risk_score: int = 0                     # 종합 점수
    news_top_headline: Optional[str] = None      # 최고 위험 헤드라인

    def summary(self) -> str:
        parts = []
        if self.fear_greed_index is not None:
            parts.append(f"F&G={self.fear_greed_index}({self.fear_greed_label})")
        if self.upcoming_high_impact_count > 0:
            parts.append(f"고영향이벤트={self.upcoming_high_impact_count}건")
        if self.nearest_high_impact_hours is not None:
            parts.append(f"최근이벤트={self.nearest_high_impact_hours:.1f}h({self.nearest_event_title})")
        if self.btc_dominance is not None:
            parts.append(f"BTC.D={self.btc_dominance:.1f}%")
        if self.news_risk_level != "NORMAL":
            parts.append(f"뉴스={self.news_risk_level}({self.news_risk_score})")
        return " | ".join(parts) if parts else "매크로 데이터 없음"


# ── 매크로 뉴스 수집기 ────────────────────────────────────────

# 주요 경제 이벤트 키워드 (고영향 판별용)
# 거시지표 + 지정학 + 암호화폐 고유 이슈를 포괄하도록 확장됨.
HIGH_IMPACT_KEYWORDS = [
    # 미국 거시지표
    "FOMC", "Federal Reserve", "Interest Rate Decision",
    "CPI", "Consumer Price Index",
    "NFP", "Non-Farm Payroll", "Nonfarm",
    "GDP", "Gross Domestic Product",
    "PCE", "Personal Consumption",
    "PPI", "Producer Price Index",
    "Unemployment Rate",
    "Retail Sales",
    "PMI",
    "Jackson Hole",
    # 중앙은행/규제
    "ECB", "BOJ", "SEC decision", "CFTC",
    "Spot ETF", "ETF Approval", "ETF Decision",
    # 지정학 (전쟁/제재/테러)
    "War", "Invasion", "Sanctions", "Missile",
    "Ukraine", "Russia", "Iran", "Israel", "Taiwan", "China",
    "OPEC", "Nuclear",
    # 암호화폐 고유
    "Halving", "Halvening", "Hard Fork",
    "Exchange Hack", "Stablecoin Depeg",
    "Satoshi", "Nakamoto", "Bitcoin Creator",
]


class MacroNewsCollector:
    """거시경제 뉴스 및 시장 심리 데이터 수집기"""

    CACHE_DIR = Path(__file__).parent.parent / "data_cache" / "macro"
    FEAR_GREED_URL = "https://api.alternative.me/fng/"
    COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"

    # 캐시 TTL (초)
    FEAR_GREED_TTL = 3600       # 1시간
    GLOBAL_MARKET_TTL = 1800    # 30분
    EVENTS_TTL = 14400          # 4시간

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: dict = {}
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── 공개 API ──────────────────────────────────────────────

    async def get_snapshot(self) -> MacroSnapshot:
        """현재 매크로 환경 스냅샷을 반환한다."""
        snapshot = MacroSnapshot()

        # 병렬로 데이터 수집
        results = await asyncio.gather(
            self._fetch_fear_greed(),
            self._fetch_global_market(),
            self._fetch_scheduled_events(),
            return_exceptions=True,
        )

        # Fear & Greed Index
        if isinstance(results[0], dict):
            snapshot.fear_greed_index = results[0].get("value")
            snapshot.fear_greed_label = results[0].get("label")

        # Global Market (BTC dominance, market cap change)
        if isinstance(results[1], dict):
            snapshot.btc_dominance = results[1].get("btc_dominance")
            snapshot.total_market_cap_change_24h = results[1].get("market_cap_change_24h")

        # Scheduled Events
        if isinstance(results[2], list):
            snapshot.events = results[2]
            now = datetime.now(timezone.utc)
            future_24h = now + timedelta(hours=24)

            high_impact = [
                e for e in results[2]
                if e.is_high_impact and now <= e.datetime_utc <= future_24h
            ]
            snapshot.upcoming_high_impact_count = len(high_impact)

            if high_impact:
                nearest = min(high_impact, key=lambda e: e.datetime_utc)
                snapshot.nearest_high_impact_hours = nearest.hours_until
                snapshot.nearest_event_title = nearest.title

        logger.debug(f"[매크로] {snapshot.summary()}")
        return snapshot

    def get_event_blackout_times(self, hours_ahead: int = 48) -> list[str]:
        """향후 N시간 내 고영향 이벤트의 UTC 시각 리스트 반환 (블랙아웃용)."""
        cache_file = self.CACHE_DIR / "events_cache.json"
        if not cache_file.exists():
            return []

        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            events = data.get("events", [])
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(hours=hours_ahead)

            blackout_times = []
            for ev in events:
                if ev.get("impact") != "HIGH":
                    continue
                dt = datetime.fromisoformat(ev["datetime_utc"])
                if now <= dt <= cutoff:
                    blackout_times.append(dt.strftime("%Y-%m-%d %H:%M"))
            return blackout_times
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"이벤트 블랙아웃 시간 조회 실패: {e}")
            return []

    # ── Fear & Greed Index ────────────────────────────────────

    async def _fetch_fear_greed(self) -> Optional[dict]:
        """Crypto Fear & Greed Index 조회"""
        cached = self._check_cache("fear_greed", self.FEAR_GREED_TTL)
        if cached is not None:
            return cached

        try:
            session = await self._get_session()
            async with session.get(self.FEAR_GREED_URL, params={"limit": 1}) as resp:
                if resp.status != 200:
                    logger.warning(f"Fear & Greed API 응답 {resp.status}")
                    return None
                data = await resp.json()

            if data and data.get("data"):
                entry = data["data"][0]
                result = {
                    "value": int(entry["value"]),
                    "label": entry["value_classification"],
                    "timestamp": int(entry["timestamp"]),
                }
                self._set_cache("fear_greed", result)
                return result
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Fear & Greed 조회 실패 (네트워크): {e}")
        except (KeyError, ValueError) as e:
            logger.warning(f"Fear & Greed 조회 실패 (파싱): {e}")
        return None

    # ── Global Market Data ────────────────────────────────────

    async def _fetch_global_market(self) -> Optional[dict]:
        """CoinGecko Global Market 데이터 조회"""
        cached = self._check_cache("global_market", self.GLOBAL_MARKET_TTL)
        if cached is not None:
            return cached

        try:
            session = await self._get_session()
            async with session.get(self.COINGECKO_GLOBAL_URL) as resp:
                if resp.status != 200:
                    logger.warning(f"CoinGecko Global API 응답 {resp.status}")
                    return None
                data = await resp.json()

            gd = data.get("data", {})
            result = {
                "btc_dominance": gd.get("market_cap_percentage", {}).get("btc"),
                "market_cap_change_24h": gd.get("market_cap_change_percentage_24h_usd"),
            }
            self._set_cache("global_market", result)
            return result
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Global Market 조회 실패 (네트워크): {e}")
        except (KeyError, ValueError) as e:
            logger.warning(f"Global Market 조회 실패 (파싱): {e}")
        return None

    # ── Scheduled Economic Events ─────────────────────────────

    async def _fetch_scheduled_events(self) -> list[MacroEvent]:
        """예정된 경제 이벤트 조회 (로컬 캐시 + 정적 캘린더 혼합)"""
        cached = self._check_cache("events", self.EVENTS_TTL)
        if cached is not None:
            return cached

        events = []

        # 정적 주요 이벤트 캘린더 로드
        events.extend(self._load_static_calendar())

        # 캐시에 저장
        self._set_cache("events", events)

        # 파일 캐시도 갱신
        self._save_events_cache(events)

        return events

    def _load_static_calendar(self) -> list[MacroEvent]:
        """로컬 JSON 캘린더 파일에서 이벤트를 로드한다."""
        calendar_file = self.CACHE_DIR / "economic_calendar.json"
        if not calendar_file.exists():
            # 기본 캘린더 생성
            self._create_default_calendar(calendar_file)

        try:
            data = json.loads(calendar_file.read_text(encoding="utf-8"))
            events = []
            now = datetime.now(timezone.utc)
            for item in data.get("events", []):
                try:
                    dt = datetime.fromisoformat(item["datetime_utc"]).replace(tzinfo=timezone.utc)
                    # 과거 이벤트는 스킵
                    if dt < now - timedelta(hours=1):
                        continue
                    events.append(MacroEvent(
                        title=item["title"],
                        datetime_utc=dt,
                        impact=item.get("impact", "MEDIUM"),
                        currency=item.get("currency", "USD"),
                    ))
                except (KeyError, ValueError) as e:
                    logger.debug(f"경제 캘린더 이벤트 파싱 실패: {e}")
                    continue
            return events
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"경제 캘린더 로드 실패: {e}")
            return []

    def _create_default_calendar(self, path: Path) -> None:
        """기본 경제 캘린더 JSON 생성 (주요 정기 이벤트)"""
        # 2026년 주요 예정 이벤트 (수동 관리 또는 외부 동기화 필요)
        calendar = {
            "_comment": "주요 경제 이벤트 캘린더. 수동 업데이트 또는 외부 동기화 권장.",
            "_updated": datetime.now(timezone.utc).isoformat(),
            "events": [
                # FOMC 2026 (예상 일정)
                {"title": "FOMC Interest Rate Decision", "datetime_utc": "2026-05-06T18:00:00", "impact": "HIGH", "currency": "USD"},
                {"title": "FOMC Interest Rate Decision", "datetime_utc": "2026-06-17T18:00:00", "impact": "HIGH", "currency": "USD"},
                {"title": "FOMC Interest Rate Decision", "datetime_utc": "2026-07-29T18:00:00", "impact": "HIGH", "currency": "USD"},
                {"title": "FOMC Interest Rate Decision", "datetime_utc": "2026-09-16T18:00:00", "impact": "HIGH", "currency": "USD"},
                {"title": "FOMC Interest Rate Decision", "datetime_utc": "2026-11-04T18:00:00", "impact": "HIGH", "currency": "USD"},
                {"title": "FOMC Interest Rate Decision", "datetime_utc": "2026-12-16T18:00:00", "impact": "HIGH", "currency": "USD"},
                # CPI (매월 둘째 주 화요일/수요일 08:30 ET = 12:30 UTC)
                {"title": "US CPI (Apr)", "datetime_utc": "2026-05-13T12:30:00", "impact": "HIGH", "currency": "USD"},
                {"title": "US CPI (May)", "datetime_utc": "2026-06-10T12:30:00", "impact": "HIGH", "currency": "USD"},
                {"title": "US CPI (Jun)", "datetime_utc": "2026-07-14T12:30:00", "impact": "HIGH", "currency": "USD"},
                # NFP (매월 첫째 주 금요일 08:30 ET = 12:30 UTC)
                {"title": "US Non-Farm Payrolls (Apr)", "datetime_utc": "2026-05-08T12:30:00", "impact": "HIGH", "currency": "USD"},
                {"title": "US Non-Farm Payrolls (May)", "datetime_utc": "2026-06-05T12:30:00", "impact": "HIGH", "currency": "USD"},
                {"title": "US Non-Farm Payrolls (Jun)", "datetime_utc": "2026-07-02T12:30:00", "impact": "HIGH", "currency": "USD"},
            ],
        }
        path.write_text(json.dumps(calendar, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(f"기본 경제 캘린더 생성: {path}")

    def _save_events_cache(self, events: list[MacroEvent]) -> None:
        """이벤트 캐시 파일 저장"""
        cache_file = self.CACHE_DIR / "events_cache.json"
        try:
            data = {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "events": [
                    {
                        "title": e.title,
                        "datetime_utc": e.datetime_utc.isoformat(),
                        "impact": e.impact,
                        "currency": e.currency,
                    }
                    for e in events
                ],
            }
            cache_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except (OSError, TypeError) as e:
            logger.warning(f"이벤트 캐시 저장 실패: {e}")

    # ── 캐시 유틸 ─────────────────────────────────────────────

    def _check_cache(self, key: str, ttl: int):
        entry = self._cache.get(key)
        if entry and (time.time() - entry["ts"]) < ttl:
            return entry["data"]
        return None

    def _set_cache(self, key: str, data) -> None:
        self._cache[key] = {"data": data, "ts": time.time()}
