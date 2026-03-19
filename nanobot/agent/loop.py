"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import httpx

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryConsolidator
from nanobot.agent.subagent import SubagentManager
from nanobot.agent.tools.arxiv import ArxivGetTool, ArxivSearchTool
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.skills import BUILTIN_SKILLS_DIR
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.pdf import PDFInfoTool, ReadPDFTool
from nanobot.agent.tools.summarize_pdf import SummaryFileCommandTool, SummaryPDFFileTool
from nanobot.agent.tools.message import MessageTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.spawn import SpawnTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, WebSearchConfig
    from nanobot.cron.service import CronService


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        web_search_config: WebSearchConfig | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict | None = None,
        channels_config: ChannelsConfig | None = None,
    ):
        from nanobot.config.schema import ExecToolConfig, WebSearchConfig

        self.bus = bus
        self.channels_config = channels_config
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            web_search_config=self.web_search_config,
            web_proxy=web_proxy,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )

        self._running = False
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._background_tasks: list[asyncio.Task] = []
        self._processing_lock = asyncio.Lock()
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
        )
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read))
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ReadPDFTool(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(PDFInfoTool(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(SummaryPDFFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        self.tools.register(ArxivSearchTool(proxy=self.web_proxy))
        self.tools.register(ArxivGetTool(proxy=self.web_proxy))
        self.tools.register(MessageTool(send_callback=self.bus.publish_outbound))
        self.tools.register(SpawnTool(manager=self.subagents))
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """Connect to configured MCP servers (one-time, lazy)."""
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except BaseException as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """Update context for all tools that need routing info."""
        for name in ("message", "spawn", "cron"):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id, *([message_id] if name == "message" else []))

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            tool_defs = self.tools.get_definitions()

            response = await self.provider.chat_with_retry(
                messages=messages,
                tools=tool_defs,
                model=self.model,
            )

            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    tool_hint = self._tool_hint(response.tool_calls)
                    tool_hint = self._strip_think(tool_hint)
                    await on_progress(tool_hint, tool_hint=True)

                tool_call_dicts = [
                    tc.to_openai_tool_call()
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        await self._connect_mcp()
        logger.info("Agent loop started")

        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.warning("Error consuming inbound message: {}, continuing...", e)
                continue

            cmd = msg.content.strip().lower()
            if cmd == "/stop":
                await self._handle_stop(msg)
            elif cmd == "/restart":
                await self._handle_restart(msg)
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks and subagents for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        sub_cancelled = await self.subagents.cancel_by_session(msg.session_key)
        total = cancelled + sub_cancelled
        content = f"Stopped {total} task(s)." if total else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _handle_restart(self, msg: InboundMessage) -> None:
        """Restart the process in-place via os.execv."""
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content="Restarting...",
        ))

        async def _do_restart():
            await asyncio.sleep(1)
            # Use -m nanobot instead of sys.argv[0] for Windows compatibility
            # (sys.argv[0] may be just "nanobot" without full path on Windows)
            os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

        asyncio.create_task(_do_restart())

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    async def close_mcp(self) -> None:
        """Drain pending background archives, then close MCP connections."""
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
            self._background_tasks.clear()
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass  # MCP SDK cancel scope cleanup is noisy but harmless
            self._mcp_stack = None

    def _schedule_background(self, coro) -> None:
        """Schedule a coroutine as a tracked background task (drained on shutdown)."""
        task = asyncio.create_task(coro)
        self._background_tasks.append(task)
        task.add_done_callback(self._background_tasks.remove)

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            history = session.get_history(max_messages=0)
            # Subagent results should be assistant role, other system messages use user role
            current_role = "assistant" if msg.sender_id == "subagent" else "user"
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                current_role=current_role,
            )
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # Slash commands
        cmd = msg.content.strip().lower()
        if cmd == "/new":
            snapshot = session.messages[session.last_consolidated:]
            session.clear()
            self.sessions.save(session)
            self.sessions.invalidate(session.key)

            if snapshot:
                self._schedule_background(self.memory_consolidator.archive_messages(snapshot))

            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        if cmd == "/help":
            lines = [
                "🐈 nanobot commands:",
                "/new — Start a new conversation",
                "/stop — Stop the current task",
                "/restart — Restart the bot",
                "/xuanke <课程名/课号/教师> — 查询 SJTU 选课社区评价",
                "/summaryfile <pdf_path> [--exam-date YYYY-MM-DD] [--focus 'topic'] — 期末考试复习助手",
                "/help — Show available commands",
            ]
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines),
            )
        
        # /xuanke command - SJTU 选课社区查询
        if cmd.startswith("/xuanke"):
            return await self._handle_xuanke_command(msg)
        
        # /summaryfile command - 期末考试复习助手
        if cmd.startswith("/summaryfile"):
            return await self._handle_summaryfile_command(msg)
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
        self._schedule_background(self.memory_consolidator.maybe_consolidate_by_tokens(session))

        if (mt := self.tools.get("message")) and isinstance(mt, MessageTool) and mt._sent_in_turn:
            return None

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue  # Strip runtime context from multimodal messages
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            path = (c.get("_meta") or {}).get("path", "")
                            placeholder = f"[image: {path}]" if path else "[image]"
                            filtered.append({"type": "text", "text": placeholder})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def _handle_summaryfile_command(
        self, msg: InboundMessage
    ) -> OutboundMessage:
        """Handle /summaryfile command - 期末考试复习助手"""
        content = msg.content.strip()
        parts = content.split(maxsplit=2)
        
        if len(parts) < 2:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=(
                    "❌ 用法错误\n\n"
                    "📚 /summaryfile - 期末考试复习助手\n"
                    "帮你快速总结PDF讲义，生成结构化复习材料\n\n"
                    "用法：\n"
                    "  /summaryfile /path/to/file.pdf\n"
                    "  /summaryfile ~/Downloads/lecture.pdf --exam-date 2024-01-20\n"
                    "  /summaryfile ./notes.pdf --focus '第三章积分'\n\n"
                    "功能包括：\n"
                    "  ✅ 核心要点速览\n"
                    "  ✅ 重点复习区域（必考/高频/了解）\n"
                    "  ✅ 易混淆概念辨析\n"
                    "  ✅ 公式定理速记\n"
                    "  ✅ 模拟练习题\n"
                    "  ✅ 知识图谱\n"
                    "  ✅ 记忆技巧\n"
                    "  ✅ 考试陷阱提醒\n"
                    "  ✅ 复习时间规划"
                ),
            )
        
        # 解析参数
        file_path = parts[1]
        extra_args = parts[2] if len(parts) > 2 else ""
        
        exam_date = None
        focus_areas = None
        
        # 解析 --exam-date 和 --focus
        if "--exam-date" in extra_args:
            try:
                idx = extra_args.index("--exam-date")
                date_part = extra_args[idx + len("--exam-date"):].strip()
                exam_date = date_part.split()[0] if date_part else None
            except:
                pass
        
        if "--focus" in extra_args:
            try:
                idx = extra_args.index("--focus")
                focus_part = extra_args[idx + len("--focus"):].strip()
                # 提取引号内的内容或第一个单词
                if focus_part.startswith('"') or focus_part.startswith("'"):
                    quote_char = focus_part[0]
                    end_idx = focus_part.find(quote_char, 1)
                    if end_idx > 0:
                        focus_areas = focus_part[1:end_idx]
                else:
                    focus_areas = focus_part.split()[0] if focus_part else None
            except:
                pass
        
        # 发送处理中消息
        progress_msg = f"📚 正在为你生成复习材料...\n📄 文件: {file_path}"
        if exam_date:
            progress_msg += f"\n📅 考试日期: {exam_date}"
        if focus_areas:
            progress_msg += f"\n🎯 重点: {focus_areas}"
        
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=progress_msg,
            )
        )
        
        # 调用 summary_pdf_file 工具读取并准备总结内容
        summary_tool = self.tools.get("summary_pdf_file")
        if not summary_tool:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="❌ 错误：summary_pdf_file 工具未加载",
            )
        
        # 获取PDF内容（会返回一个大型prompt）
        pdf_content = await summary_tool.execute(
            file_path=file_path,
            exam_date=exam_date,
            focus_areas=focus_areas,
        )
        
        if pdf_content.startswith("❌") or pdf_content.startswith("Error"):
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=pdf_content,
            )
        
        # 使用 LLM 生成结构化总结
        from nanobot.agent.context import ContextBuilder
        
        session = self.sessions.get_or_create(f"{msg.channel}:{msg.chat_id}")
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位专业的学习助手和考试辅导专家。\n"
                    "你的任务是帮助学生高效复习，准备期末考试。\n"
                    "请基于提供的PDF内容，生成结构化、易读的复习材料。\n"
                    "使用 emoji、表格、列表等格式增强可读性。"
                ),
            },
            {
                "role": "user",
                "content": pdf_content,
            },
        ]
        
        try:
            final_content = await self._run_agent_loop_simple(messages)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=final_content or "生成复习材料时出错，请重试",
            )
        except Exception as e:
            logger.error("Failed to generate summary: {}", e)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"❌ 生成复习材料失败: {e}",
            )
    
    async def _handle_xuanke_command(
        self, msg: InboundMessage
    ) -> OutboundMessage:
        """Handle /xuanke command - SJTU 选课社区查询"""
        content = msg.content.strip()
        parts = content.split(maxsplit=1)
        
        if len(parts) < 2:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=(
                    "❌ 用法错误\n\n"
                    "📚 /xuanke - 查询 SJTU 选课社区评价\n"
                    "查询上海交通大学课程评价和老师评分\n\n"
                    "用法：\n"
                    "  /xuanke <课程名>     - 按课程名称搜索\n"
                    "  /xuanke <课号>       - 按课程代码搜索\n"
                    "  /xuanke <教师名>     - 按教师姓名搜索\n\n"
                    "示例：\n"
                    "  /xuanke 机器学习\n"
                    "  /xuanke AI3604\n"
                    "  /xuanke 李旭东"
                ),
            )
        
        query = parts[1].strip()
        
        # 发送处理中消息
        await self.bus.publish_outbound(
            OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"🔍 正在查询「{query}」的课程评价...",
            )
        )
        
        try:
            # 第一阶段：快速搜索前100页
            courses_first = await self._search_courses_batch(query, max_pages=100)
            
            if not courses_first:
                # 第一阶段没找到，继续搜索全部
                await self.bus.publish_outbound(
                    OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"🔍 正在深入搜索「{query}」...",
                    )
                )
                courses_all = await self._search_courses_batch(query, max_pages=668)
                courses_first = courses_all
            
            if not courses_first:
                return OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=f"❌ 未找到与「{query}」相关的课程",
                )
            
            # 取前5个获取评价
            top_courses = courses_first[:5]
            results = []
            for course in top_courses:
                course_info = await self._get_course_reviews(course)
                results.append(course_info)
            
            # 生成总结
            summary = await self._summarize_courses(query, results, total_found=len(courses_first))
            
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=summary,
            )
            
        except Exception as e:
            logger.error("Xuanke command failed: {}", e)
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content=f"❌ 查询失败: {e}",
            )
    
    async def _search_courses_batch(self, query: str, max_pages: int = 100) -> list[dict]:
        """Search courses from SJTU course mirror API with specified page limit"""
        base_url = "http://nas.oct0pus.top:41735"
        query_lower = query.lower()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 先获取第1页
            response = await client.get(
                f"{base_url}/api/course/",
                params={"page": 1, "page_size": 20}
            )
            response.raise_for_status()
            data = response.json()
            
            all_courses = data.get("results", [])
            total_count = data.get("count", 0)
            total_pages = (total_count + 19) // 20
            
            # 限制搜索页数
            max_pages = min(max_pages, total_pages)
            
            # 并发获取多页
            async def fetch_page(page: int) -> tuple[int, list[dict]]:
                try:
                    resp = await client.get(
                        f"{base_url}/api/course/",
                        params={"page": page, "page_size": 20},
                        timeout=15.0
                    )
                    if resp.status_code == 200:
                        return page, resp.json().get("results", [])
                except Exception:
                    pass
                return page, []
            
            # 分批并发获取，每批50个请求
            batch_size = 50
            for batch_start in range(2, max_pages + 1, batch_size):
                batch_end = min(batch_start + batch_size, max_pages + 1)
                tasks = [fetch_page(p) for p in range(batch_start, batch_end)]
                completed = await asyncio.gather(*tasks)
                
                for _, courses in completed:
                    all_courses.extend(courses)
            
            # 最终过滤并返回所有匹配结果
            results = []
            for course in all_courses:
                course_name = course.get("name", "").lower()
                course_code = course.get("code", "").lower()
                teacher = course.get("teacher", "").lower()
                
                if (query_lower in course_name or 
                    query_lower in course_code or 
                    query_lower in teacher):
                    results.append(course)
            
            return results
    
    async def _get_course_reviews(self, course: dict) -> dict:
        """Get reviews for a specific course"""
        base_url = "http://nas.oct0pus.top:41735"
        course_id = course["id"]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 获取课程评价
            response = await client.get(
                f"{base_url}/api/course/{course_id}/review/",
                params={"page_size": 10}
            )
            
            reviews = []
            if response.status_code == 200:
                data = response.json()
                reviews = data.get("results", [])
            
            return {
                "course": course,
                "reviews": reviews,
            }
    
    async def _summarize_courses(self, query: str, results: list[dict], total_found: int = 0) -> str:
        """Use LLM to summarize course information and reviews"""
        if not results:
            return f"未找到与「{query}」相关的课程评价"
        
        # 使用 total_found 如果提供，否则使用 results 长度
        display_total = total_found if total_found > len(results) else len(results)
        
        # 构建提示
        prompt = f"""请根据以下 SJTU 选课社区的查询结果，为用户生成一份详细的课程评价总结。

## 用户查询
「{query}」

## 查询结果
共找到 {display_total} 门相关课程，以下是前 {len(results)} 门的详细信息：
"""
        
        for idx, result in enumerate(results, 1):
            course = result["course"]
            reviews = result["reviews"]
            
            rating = course.get("rating") or {}
            avg_rating = rating.get("avg") or 0
            review_count = rating.get("count") or 0
            
            prompt += f"\n### 课程 {idx}\n"
            prompt += f"- 课程名称：{course.get('name', 'N/A')}\n"
            prompt += f"- 课程代码：{course.get('code', 'N/A')}\n"
            prompt += f"- 授课教师：{course.get('teacher', 'N/A')}\n"
            prompt += f"- 开课院系：{course.get('department', 'N/A')}\n"
            prompt += f"- 学分：{course.get('credit', 'N/A')}\n"
            prompt += f"- 综合评分：{avg_rating:.1f}/5.0（基于 {review_count} 条评价）\n"
            
            if reviews:
                prompt += f"\n#### 学生评价（{len(reviews)} 条）：\n"
                for i, review in enumerate(reviews[:5], 1):
                    semester = review.get('semester', '未知学期')
                    rating_val = review.get('rating', 0)
                    comment = review.get('comment', '无内容')
                    score = review.get('score', None)
                    
                    prompt += f"\n**评价 {i}** ({semester}, 评分: {rating_val}/5"
                    if score:
                        prompt += f", 成绩: {score}"
                    prompt += f")\n{comment}\n"
            else:
                prompt += "\n暂无学生评价\n"
        
        prompt += f"""
## 请生成以下格式的总结报告：

1. **查询结果概览** - 说明共找到 {display_total} 门相关课程，显示前 {len(results)} 门
2. **各课程详细分析** - 对每个课程：
   - 基本信息（名称、代码、教师、院系）
   - 综合评分和推荐指数
   - 学生评价要点总结（优缺点、给分情况、课程难度、作业量等）
3. **选课建议** - 根据不同需求（如：给分好、学到东西、轻松等）给出建议
4. **更多课程提示** - 如果 {display_total} > {len(results)}，提示用户"还有 {display_total - len(results)} 门相关课程，可使用更精确的关键词（如教师名或课号）查看更多"
5. **总体评价** - 一句话总结

请使用 emoji、表格等格式增强可读性，用中文回答。
"""
        
        messages = [
            {
                "role": "system",
                "content": "你是一位熟悉上海交通大学课程的选课助手，擅长分析课程评价并为同学提供选课建议。"
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        
        try:
            summary = await self._run_agent_loop_simple(messages)
            return summary or "生成总结时出错，请重试"
        except Exception as e:
            logger.error("Failed to summarize courses: {}", e)
            # 如果 LLM 总结失败，返回原始数据
            return self._format_simple_results(query, results)
    
    def _format_simple_results(self, query: str, results: list[dict]) -> str:
        """Format results in simple text when LLM summarization fails"""
        lines = [f"📚 「{query}」的查询结果：\n"]
        
        for idx, result in enumerate(results, 1):
            course = result["course"]
            reviews = result["reviews"]
            
            rating = course.get("rating") or {}
            avg_rating = rating.get("avg") or 0
            review_count = rating.get("count") or 0
            
            stars = "⭐" * int(round(avg_rating))
            
            lines.append(f"\n**{idx}. {course.get('name', 'N/A')}** ({course.get('code', 'N/A')})")
            lines.append(f"👨‍🏫 教师：{course.get('teacher', 'N/A')}")
            lines.append(f"🏫 院系：{course.get('department', 'N/A')}")
            lines.append(f"⭐ 评分：{avg_rating:.1f}/5.0 {stars} ({review_count} 条评价)")
            lines.append(f"📖 学分：{course.get('credit', 'N/A')}")
            
            if reviews:
                lines.append(f"\n📝 最新评价：")
                for review in reviews[:3]:
                    semester = review.get('semester', '未知学期')
                    rating_val = review.get('rating', 0)
                    comment = review.get('comment', '无内容')[:150]
                    lines.append(f"  • [{semester}] {rating_val}⭐: {comment}...")
            
            lines.append("")
        
        lines.append("\n💡 使用 `/xuanke <课程名/课号/教师>` 查询更多课程")
        return "\n".join(lines)

    async def _run_agent_loop_simple(self, messages: list[dict]) -> str:
        """Simplified agent loop for single-turn generation"""
        try:
            from nanobot.providers.base import GenerationSettings
            
            response = await self.provider.chat(
                messages=messages,
                tools=None,  # 不启用工具，纯文本生成
                model=self.model,
                max_tokens=8192,
                temperature=0.3,
            )
            
            return response.content or "生成内容为空"
            
        except Exception as e:
            logger.error("Simple agent loop failed: {}", e)
            return f"生成失败: {e}"

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or cron usage)."""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
