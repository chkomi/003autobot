#!/usr/bin/env python3
"""RQ Worker 진입점 (plan v2 §3.3 + §4).

사용법:
    # 로컬 직접 실행
    python worker.py

    # docker-compose
    docker-compose run --rm worker

환경변수:
    AUTOBOT_API_REDIS_URL   Redis URL (기본: redis://localhost:6379/0)
    RQ_QUEUE_NAMES          처리할 큐 이름 (기본: backtest)
    RQ_BURST               "1" 이면 큐가 빌 때 즉시 종료 (배치 처리용)

Redis 없이 실행하면 ImportError 와 함께 종료된다.
백테스트를 Redis 없이 실행하려면 API 서버가 BackgroundTask 로 처리한다.
"""
import os
import socket
import sys
from pathlib import Path

# ── 프로젝트 루트를 sys.path에 추가 ──────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

try:
    import redis
    from rq import Worker, Queue, Connection
except ImportError:
    logger.error(
        "rq / redis 패키지가 설치되어 있지 않습니다. "
        "`pip install rq redis` 후 다시 실행하세요."
    )
    sys.exit(1)


def main() -> None:
    redis_url = os.environ.get("AUTOBOT_API_REDIS_URL", "redis://localhost:6379/0")
    queue_names_raw = os.environ.get("RQ_QUEUE_NAMES", "backtest")
    queue_names = [q.strip() for q in queue_names_raw.split(",") if q.strip()]
    burst = os.environ.get("RQ_BURST", "0") == "1"

    logger.info(f"RQ Worker 시작: redis={redis_url} queues={queue_names} burst={burst}")

    conn = redis.from_url(redis_url)

    # Fix #22: hostname 기반 워커 이름으로 멀티 워커 충돌 방지
    worker_name = f"autobot-worker-{socket.gethostname()}"
    logger.info(f"RQ Worker 이름: {worker_name}")

    with Connection(conn):
        worker = Worker(
            queues=[Queue(name=q, connection=conn) for q in queue_names],
            connection=conn,
            name=worker_name,
            log_job_description=True,
        )
        worker.work(burst=burst, with_scheduler=True)


if __name__ == "__main__":
    main()
