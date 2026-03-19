"""Canvas LMS API tool for the agent."""

from __future__ import annotations

import json
from typing import Any

import httpx
from loguru import logger

from nanobot.agent.tools.base import Tool

CANVAS_BASE = "https://oc.sjtu.edu.cn"


class CanvasAPITool(Tool):
    """Call Canvas LMS REST API endpoints using the stored session cookie.

    Supports GET and POST requests to /api/v1/... endpoints.
    Returns the raw JSON response as a string.
    """

    def __init__(self, session_cookie: str) -> None:
        self._cookie = session_cookie

    @property
    def name(self) -> str:
        return "canvas_api"

    @property
    def description(self) -> str:
        return (
            "Call the SJTU Canvas LMS REST API (oc.sjtu.edu.cn/api/v1/...). "
            "Use this to list courses, assignments, files, announcements, grades, etc. "
            "Common paths: /api/v1/courses, /api/v1/courses/{id}/assignments, "
            "/api/v1/courses/{id}/files, /api/v1/users/self/profile. "
            "IMPORTANT: Always pass per_page=100 to get all results in one call. "
            "For file downloads use the file's 'url' field from the files API."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "API path, e.g. /api/v1/courses or /api/v1/courses/87068/assignments",
                },
                "params": {
                    "type": "object",
                    "description": "Optional query parameters as key-value pairs, e.g. {\"per_page\": 50}",
                },
                "method": {
                    "type": "string",
                    "enum": ["GET", "POST"],
                    "description": "HTTP method, default GET",
                },
                "body": {
                    "type": "object",
                    "description": "JSON body for POST requests",
                },
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        params: dict | None = None,
        method: str = "GET",
        body: dict | None = None,
    ) -> str:
        try:
            async with httpx.AsyncClient(
                base_url=CANVAS_BASE,
                follow_redirects=True,
                timeout=30.0,
                cookies={"_normandy_session": self._cookie},
            ) as client:
                if method.upper() == "POST":
                    r = await client.post(path, params=params, json=body)
                else:
                    r = await client.get(path, params=params)

                if r.status_code == 401:
                    return "❌ Canvas session 已过期，请运行 /config /canvas 重新登录。"
                if r.status_code >= 400:
                    return f"❌ Canvas API 返回 {r.status_code}: {r.text[:500]}"

                try:
                    data = r.json()
                except Exception:
                    return r.text[:4000]

                # Slim down course list responses — they contain many unused fields
                if isinstance(data, list) and data and "course_code" in data[0]:
                    data = [
                        {
                            "id": c.get("id"),
                            "name": c.get("name"),
                            "course_code": c.get("course_code"),
                            "workflow_state": c.get("workflow_state"),
                            "enrollment_term_id": c.get("enrollment_term_id"),
                        }
                        for c in data
                    ]

                result = json.dumps(data, ensure_ascii=False, indent=2)
                # Truncate very large responses
                if len(result) > 12000:
                    result = result[:12000] + "\n... (truncated)"
                return result

        except Exception as exc:
            logger.error("canvas_api tool error: {}", exc)
            return f"❌ 请求失败: {exc}"
