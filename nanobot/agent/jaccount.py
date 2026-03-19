"""Authentication helper for course.sjtu.plus (SJTU 选课社区)."""

from __future__ import annotations

import httpx
from loguru import logger

COURSE_SJTU_BASE = "https://course.sjtu.plus"


class CourseSJTUSession:
    """Authenticated session for course.sjtu.plus.

    Uses email + password login via POST /oauth/login/.
    Keeps session cookies for subsequent API calls.
    """

    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password
        self._client: httpx.AsyncClient | None = None
        self._authenticated = False

    async def __aenter__(self) -> "CourseSJTUSession":
        self._client = httpx.AsyncClient(follow_redirects=True, timeout=30.0)
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        self._authenticated = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def login(self) -> bool:
        """Login to course.sjtu.plus with email + password.

        Returns True on success.
        """
        client = self._ensure_client()
        try:
            # Get CSRF cookie first (Django sets it on any request)
            await client.get(f"{COURSE_SJTU_BASE}/")
            csrftoken = client.cookies.get("csrftoken", "")

            resp = await client.post(
                f"{COURSE_SJTU_BASE}/oauth/email/login/",
                json={"account": self.username, "password": self.password},
                headers={
                    "X-CSRFToken": csrftoken,
                    "Referer": f"{COURSE_SJTU_BASE}/login",
                    "Origin": COURSE_SJTU_BASE,
                },
            )

            if resp.status_code == 200:
                self._authenticated = True
                logger.info("course.sjtu.plus login successful")
                return True

            logger.error(
                "course.sjtu.plus login failed ({}): {}",
                resp.status_code,
                resp.text[:200],
            )
            return False

        except Exception as exc:
            logger.error("course.sjtu.plus login error: {}", exc)
            return False

    async def get(self, url: str, **kwargs: object) -> httpx.Response:
        """Authenticated GET. Re-authenticates once on 401/403."""
        client = self._ensure_client()
        if not self._authenticated:
            await self.login()

        csrftoken = client.cookies.get("csrftoken", "")
        headers = dict(kwargs.pop("headers", {}))  # type: ignore[arg-type]
        headers.setdefault("X-CSRFToken", csrftoken)

        resp = await client.get(url, headers=headers, **kwargs)  # type: ignore[arg-type]

        if resp.status_code in (401, 403):
            self._authenticated = False
            if await self.login():
                csrftoken = client.cookies.get("csrftoken", "")
                headers["X-CSRFToken"] = csrftoken
                resp = await client.get(url, headers=headers, **kwargs)  # type: ignore[arg-type]

        return resp

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(follow_redirects=True, timeout=30.0)
        return self._client


# Keep old name as alias so existing imports don't break
JAccountSession = CourseSJTUSession
