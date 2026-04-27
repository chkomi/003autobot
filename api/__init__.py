"""Autobot REST API service (FastAPI).

이 패키지는 `plan_codex.v2.md` 3.2절의 Python API 서비스 구조를 구현한다.
기존 `dashboard/` FastAPI 앱과는 독립적으로 동작하며, Vercel `web/`에서
호출하는 `/api/v1/...` 경로를 제공한다.
"""
