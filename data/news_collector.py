"""
암호화폐/지정학 뉴스 수집기.
주요 공용 피드(RSS/CryptoPanic)에서 헤드라인을 수집하고,
키워드 매칭 기반으로 위험도(Risk Score)를 계산해 봇의 전략 의사결정에 반영한다.

의도:
  - 로컬 경제 캘린더(macro_news.py)가 미리 알 수 없는 돌발 이슈
    (전쟁/제재/해킹/ETF/사토시 관련/하드포크 등)를 자동 감지한다.
  - API 키가 없어도 동작 가능하도록 RSS를 기본 소스로 사용한다.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import aiohttp
from loguru import logger


# ── 키워드 카테고리 & 가중치 ──────────────────────────────────────
# 가중치: 기본 점수 (뉴스 1건이 등장하면 이만큼 가산)
# CRITICAL(매우 큼) > HIGH(큼) > MEDIUM(주의) 순

CRITICAL_KEYWORDS: dict[str, int] = {
    # 지정학 — 전쟁/핵/테러
    "nuclear": 40, "missile strike": 35, "war declaration": 40,
    "invasion": 30, "world war": 50, "global war": 45,
    "terror attack": 30, "assassination": 25,
    # 암호화폐 시스템 리스크
    "exchange collapse": 45, "exchange hack": 40, "bridge hack": 30,
    "stablecoin depeg": 40, "usdt depeg": 45, "usdc depeg": 45,
    "bitcoin ban": 40, "crypto ban": 40,
    # 사토시 관련 (정체 공개/자산 이동)
    "satoshi revealed": 40, "satoshi identity": 35,
    "satoshi wallet moved": 45, "satoshi coins moved": 45,
    "genesis wallet": 30,
    # 거래소 파산/출금중단
    "halts withdrawals": 35, "withdrawal suspended": 30,
    "files for bankruptcy": 35, "chapter 11": 25,
}

HIGH_KEYWORDS: dict[str, int] = {
    # 지정학
    "war": 15, "military": 10, "conflict": 10, "sanctions": 15,
    "invasion": 20, "troops": 10, "ceasefire": 10,
    "russia": 8, "ukraine": 8, "iran": 10, "israel": 10,
    "china taiwan": 15, "taiwan": 10,
    "middle east": 10, "opec": 8,
    # 규제
    "sec lawsuit": 20, "sec charges": 20, "sec sues": 20,
    "cftc charges": 15, "doj charges": 15,
    "regulation": 8, "regulatory": 8, "crackdown": 15,
    "etf approved": 20, "etf rejected": 20, "etf denied": 20,
    "spot etf": 15,
    # 중앙은행/거시
    "emergency rate": 25, "rate hike": 10, "rate cut": 10,
    "fomc minutes": 8, "fed chair": 8,
    "inflation surge": 12, "recession": 10,
    # 암호화폐 고유
    "hack": 15, "exploit": 15, "rug pull": 10,
    "hard fork": 12, "halving": 10, "halvening": 10,
    "mining ban": 15, "stablecoin": 8,
    "sec approves": 18, "blackrock": 8,
    # 사토시 / BTC 고유
    "satoshi": 18, "nakamoto": 18, "bitcoin creator": 20,
    "bitcoin founder": 20, "dormant bitcoin": 15,
}

MEDIUM_KEYWORDS: dict[str, int] = {
    "volatility": 3, "selloff": 5, "crash": 8,
    "liquidation": 5, "whale": 3, "dump": 5, "pump": 3,
    "bearish": 3, "bullish": 3,
    "delisting": 5, "listing": 3,
    "outflow": 3, "inflow": 3,
    "approved": 3, "rejected": 3,
    "probe": 5, "investigation": 5,
    "warning": 3, "alert": 3,
}


# ── 데이터 모델 ──────────────────────────────────────────────────

@dataclass
class NewsItem:
    title: str
    source: str
    url: str
    published_at: datetime
    summary: str = ""
    matched_keywords: list[str] = field(default_factory=list)
    risk_score: int = 0

    @property
    def uid(self) -> str:
        """중복 판정용 고유 ID (URL 해시)."""
        return hashlib.sha1(self.url.encode("utf-8")).hexdigest()[:16]


@dataclass
class NewsRiskSnapshot:
    """최근 수집된 뉴스들의 종합 위험도 스냅샷."""
    fetched_at: datetime
    total_items: int = 0
    top_score: int = 0
    aggregate_score: int = 0
    level: str = "NORMAL"             # NORMAL | ELEVATED | HIGH | CRITICAL
    top_items: list[NewsItem] = field(default_factory=list)

    @property
    def is_critical(self) -> bool:
        return self.level == "CRITICAL"

    @property
    def is_high(self) -> bool:
        return self.level in ("HIGH", "CRITICAL")

    def summary(self) -> str:
        if self.total_items == 0:
            return "뉴스 없음"
        head = self.top_items[0].title[:60] if self.top_items else ""
        return (
            f"level={self.level} agg={self.aggregate_score} "
            f"top={self.top_score} n={self.total_items} | {head}"
        )


# ── 수집기 ────────────────────────────────────────────────────────

class NewsCollector:
    """공용 RSS + CryptoPanic(선택) 기반 뉴스 수집·위험도 평가.

    기본 소스 (무료, API 키 불필요):
      - CoinDesk RSS
      - Cointelegraph RSS
    선택 소스 (CRYPTOPANIC_API_KEY 설정 시):
      - CryptoPanic Hot Posts
    """

    DEFAULT_RSS_SOURCES: list[tuple[str, str]] = [
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Cointelegraph", "https://cointelegraph.com/rss"),
        ("Bitcoin Magazine", "https://bitcoinmagazine.com/.rss/full/"),
    ]
    CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/"

    CACHE_DIR = Path(__file__).parent.parent / "data_cache" / "news"

    # 임계값 (필요시 params로 override)
    DEFAULT_THRESHOLDS = {
        "elevated": 20,   # aggregate_score
        "high": 40,
        "critical": 70,
    }

    def __init__(
        self,
        cryptopanic_api_key: Optional[str] = None,
        max_items_per_source: int = 30,
        lookback_minutes: int = 120,
        thresholds: Optional[dict[str, int]] = None,
    ):
        self._api_key = cryptopanic_api_key or ""
        self._max_items = max_items_per_source
        self._lookback_min = lookback_minutes
        self._thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}
        self._session: Optional[aiohttp.ClientSession] = None
        self._seen_uids: set[str] = set()
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15),
                headers={"User-Agent": "autobot-news-collector/1.0"},
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── 공개 API ──────────────────────────────────────────────────

    async def fetch_snapshot(self) -> NewsRiskSnapshot:
        """현재 시각 기준 뉴스 스냅샷을 수집·평가해 반환한다."""
        now = datetime.now(timezone.utc)
        tasks: list = [self._fetch_rss(name, url) for name, url in self.DEFAULT_RSS_SOURCES]
        if self._api_key:
            tasks.append(self._fetch_cryptopanic())

        results = await asyncio.gather(*tasks, return_exceptions=True)
        items: list[NewsItem] = []
        for r in results:
            if isinstance(r, Exception):
                logger.debug(f"[뉴스] 소스 수집 실패: {r}")
                continue
            if isinstance(r, list):
                items.extend(r)

        # 최근 N분 이내로 필터 + 키워드 스코어 계산
        cutoff_ts = now.timestamp() - self._lookback_min * 60
        scored: list[NewsItem] = []
        for it in items:
            if it.published_at.timestamp() < cutoff_ts:
                continue
            score, kws = _score_text(f"{it.title} {it.summary}")
            if score <= 0:
                continue
            it.risk_score = score
            it.matched_keywords = kws
            scored.append(it)

        # 중복 제거 (같은 URL) + 점수 내림차순
        dedup: dict[str, NewsItem] = {}
        for it in scored:
            if it.uid in dedup:
                continue
            dedup[it.uid] = it
        scored = sorted(dedup.values(), key=lambda x: x.risk_score, reverse=True)

        aggregate = sum(i.risk_score for i in scored)
        top_score = scored[0].risk_score if scored else 0
        level = self._classify(aggregate, top_score)

        snapshot = NewsRiskSnapshot(
            fetched_at=now,
            total_items=len(scored),
            top_score=top_score,
            aggregate_score=aggregate,
            level=level,
            top_items=scored[:10],
        )
        self._save_cache(snapshot)
        return snapshot

    def get_new_critical_items(self, snapshot: NewsRiskSnapshot) -> list[NewsItem]:
        """이전 호출 이후 처음 보는 CRITICAL 항목 리스트 반환 (중복 알림 방지)."""
        fresh: list[NewsItem] = []
        for it in snapshot.top_items:
            if it.risk_score < self._thresholds["high"]:
                continue
            if it.uid in self._seen_uids:
                continue
            self._seen_uids.add(it.uid)
            fresh.append(it)
        # seen_uids가 과도하게 커지지 않도록 제한
        if len(self._seen_uids) > 2000:
            self._seen_uids = set(list(self._seen_uids)[-1000:])
        return fresh

    # ── 소스별 수집 ────────────────────────────────────────────────

    async def _fetch_rss(self, source_name: str, url: str) -> list[NewsItem]:
        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                if resp.status != 200:
                    return []
                text = await resp.text()
            return _parse_rss(source_name, text, self._max_items)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.debug(f"[뉴스] {source_name} RSS 실패: {e}")
            return []

    async def _fetch_cryptopanic(self) -> list[NewsItem]:
        try:
            session = await self._get_session()
            params = {"auth_token": self._api_key, "filter": "hot", "public": "true"}
            async with session.get(self.CRYPTOPANIC_URL, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.debug(f"[뉴스] CryptoPanic 실패: {e}")
            return []

        items: list[NewsItem] = []
        for post in data.get("results", [])[: self._max_items]:
            title = post.get("title", "") or ""
            src = (post.get("source") or {}).get("title", "CryptoPanic")
            url_ = post.get("url") or post.get("original_url") or ""
            published = post.get("published_at") or post.get("created_at")
            try:
                dt = datetime.fromisoformat(published.replace("Z", "+00:00")) if published else datetime.now(timezone.utc)
            except (ValueError, AttributeError):
                dt = datetime.now(timezone.utc)
            if not url_:
                continue
            items.append(NewsItem(
                title=title, source=src, url=url_, published_at=dt,
                summary="",
            ))
        return items

    # ── 평가 유틸 ──────────────────────────────────────────────────

    def _classify(self, aggregate: int, top: int) -> str:
        t = self._thresholds
        if top >= t["critical"] or aggregate >= t["critical"] + 30:
            return "CRITICAL"
        if top >= t["high"] or aggregate >= t["high"] + 20:
            return "HIGH"
        if aggregate >= t["elevated"] or top >= 15:
            return "ELEVATED"
        return "NORMAL"

    def _save_cache(self, snap: NewsRiskSnapshot) -> None:
        try:
            path = self.CACHE_DIR / "latest_snapshot.json"
            payload = {
                "fetched_at": snap.fetched_at.isoformat(),
                "level": snap.level,
                "aggregate_score": snap.aggregate_score,
                "top_score": snap.top_score,
                "total_items": snap.total_items,
                "top_items": [
                    {
                        "title": i.title, "source": i.source, "url": i.url,
                        "published_at": i.published_at.isoformat(),
                        "risk_score": i.risk_score,
                        "keywords": i.matched_keywords,
                    }
                    for i in snap.top_items
                ],
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except (OSError, TypeError) as e:
            logger.debug(f"[뉴스] 캐시 저장 실패: {e}")


# ── 내부 헬퍼 ──────────────────────────────────────────────────────

_RSS_ITEM_RE = re.compile(r"<item[^>]*>(.*?)</item>", re.DOTALL | re.IGNORECASE)
_RSS_TAG_RE = re.compile(
    r"<(title|link|description|pubDate)[^>]*>(.*?)</\1>",
    re.DOTALL | re.IGNORECASE,
)
_ATOM_ENTRY_RE = re.compile(r"<entry[^>]*>(.*?)</entry>", re.DOTALL | re.IGNORECASE)
_ATOM_TAG_RE = re.compile(
    r"<(title|summary|updated|published)[^>]*>(.*?)</\1>",
    re.DOTALL | re.IGNORECASE,
)
_ATOM_LINK_RE = re.compile(r'<link[^>]*href="([^"]+)"', re.IGNORECASE)
_CDATA_RE = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.DOTALL)
_TAG_STRIP_RE = re.compile(r"<[^>]+>")


def _strip_html(s: str) -> str:
    if not s:
        return ""
    cd = _CDATA_RE.search(s)
    if cd:
        s = cd.group(1)
    s = _TAG_STRIP_RE.sub("", s)
    return s.strip()


def _parse_rfc822(ts: str) -> datetime:
    """RSS pubDate(RFC822) 또는 ISO8601을 datetime(UTC)로 파싱. 실패 시 현재 시각."""
    ts = (ts or "").strip()
    if not ts:
        return datetime.now(timezone.utc)
    # RFC822
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError, IndexError):
        pass
    # ISO8601
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return datetime.now(timezone.utc)


def _parse_rss(source_name: str, xml_text: str, max_items: int) -> list[NewsItem]:
    """RSS 2.0 + Atom 혼합 파서 (가벼운 구현, 외부 의존성 없음)."""
    items: list[NewsItem] = []

    # RSS 2.0
    for m in _RSS_ITEM_RE.finditer(xml_text):
        block = m.group(1)
        data = {"title": "", "link": "", "description": "", "pubDate": ""}
        for tag in _RSS_TAG_RE.finditer(block):
            key = tag.group(1).lower()
            if key == "pubdate":
                key = "pubDate"
            data[key] = _strip_html(tag.group(2))
        if not data["link"] and not data["title"]:
            continue
        items.append(NewsItem(
            title=data["title"],
            source=source_name,
            url=data["link"],
            published_at=_parse_rfc822(data["pubDate"]),
            summary=data["description"][:400],
        ))
        if len(items) >= max_items:
            return items

    # Atom (CoinDesk 등 일부 피드가 Atom 혼합)
    for m in _ATOM_ENTRY_RE.finditer(xml_text):
        block = m.group(1)
        data = {"title": "", "summary": "", "updated": "", "published": ""}
        for tag in _ATOM_TAG_RE.finditer(block):
            key = tag.group(1).lower()
            data[key] = _strip_html(tag.group(2))
        link_m = _ATOM_LINK_RE.search(block)
        link = link_m.group(1) if link_m else ""
        if not link and not data["title"]:
            continue
        items.append(NewsItem(
            title=data["title"],
            source=source_name,
            url=link,
            published_at=_parse_rfc822(data.get("published") or data.get("updated") or ""),
            summary=data["summary"][:400],
        ))
        if len(items) >= max_items:
            break

    return items


def _score_text(text: str) -> tuple[int, list[str]]:
    """텍스트에서 키워드 매칭 → 총점과 매칭 키워드 리스트 반환."""
    if not text:
        return 0, []
    t = text.lower()
    score = 0
    matched: list[str] = []
    for kw, weight in CRITICAL_KEYWORDS.items():
        if kw in t:
            score += weight
            matched.append(kw)
    for kw, weight in HIGH_KEYWORDS.items():
        if kw in t:
            score += weight
            matched.append(kw)
    for kw, weight in MEDIUM_KEYWORDS.items():
        if kw in t:
            score += weight
            matched.append(kw)
    return score, matched
