"""Canvas (oc.sjtu.edu.cn) OAuth2 helper via JAccount SSO.

OAuth2 授权码流程：

  1. Bot 调用 get_authorize_url() 生成 JAccount 授权链接
  2. 用户在浏览器打开该链接，用 JAccount 登录（含验证码由用户处理）
  3. JAccount 把用户重定向回 Canvas 的 callback URL：
       https://oc.sjtu.edu.cn/login/oauth2/callback?code=AUTH_CODE&state=...
  4. 用户把浏览器地址栏的完整 URL 粘贴给 bot
  5. Bot 调用 complete_login(callback_url)，跟随该 URL 完成 OAuth 换 session
  6. 拿到 Canvas session cookie，存入 config 供后续 API 调用使用
"""

from __future__ import annotations

import httpx
from loguru import logger

CANVAS_BASE = "https://oc.sjtu.edu.cn"


async def get_authorize_url() -> str:
    """Return the JAccount OAuth2 authorize URL for Canvas login.

    Follows the initial Canvas login redirect to extract the full
    JAccount authorize URL (with client_id, redirect_uri, state, etc.).
    """
    async with httpx.AsyncClient(follow_redirects=False, timeout=15.0) as client:
        r = await client.get(f"{CANVAS_BASE}/login/openid_connect")
        if r.status_code in (301, 302, 303, 307, 308):
            loc = r.headers.get("location", "")
            if "jaccount.sjtu.edu.cn" in loc:
                return loc
        raise RuntimeError(f"Unexpected response from Canvas login: {r.status_code}")


async def complete_login(callback_url: str) -> str | None:
    """Follow the OAuth2 callback URL to obtain a Canvas session cookie.

    Args:
        callback_url: The full URL from the browser address bar after JAccount
                      login, e.g. https://oc.sjtu.edu.cn/login/oauth2/callback?code=...

    Returns:
        The Canvas ``_normandy_session`` cookie value on success, or None.
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
        try:
            r = await client.get(callback_url)
            # Canvas sets several cookies; the session cookie is typically
            # _normandy_session (or similar). Try common names.
            for name in ("_normandy_session", "_canvas_session", "canvas_session"):
                val = client.cookies.get(name)
                if val:
                    logger.info("Canvas OAuth2 login successful (cookie: {})", name)
                    return val
            # Fallback: return any cookie that looks like a session
            for name, val in client.cookies.items():
                if "session" in name.lower():
                    logger.info("Canvas OAuth2 login successful (cookie: {})", name)
                    return val
            logger.warning(
                "Canvas login completed but no session cookie found. "
                "Cookies: {}", list(client.cookies.keys())
            )
            return None
        except Exception as exc:
            logger.error("Canvas OAuth2 complete_login failed: {}", exc)
            return None


class CanvasSession:
    """Authenticated httpx client for Canvas API calls.

    Usage::

        session = CanvasSession(canvas_session_cookie)
        async with session:
            resp = await session.get("/api/v1/courses")
    """

    def __init__(self, session_cookie: str) -> None:
        self._session_cookie = session_cookie
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "CanvasSession":
        self._client = httpx.AsyncClient(
            base_url=CANVAS_BASE,
            follow_redirects=True,
            timeout=30.0,
            cookies={"_normandy_session": self._session_cookie},
        )
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def get(self, path: str, **kwargs: object) -> httpx.Response:
        assert self._client, "Use as async context manager"
        return await self._client.get(path, **kwargs)  # type: ignore[arg-type]
