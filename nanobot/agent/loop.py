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
from nanobot.agent.tools.claude_tools import MultiEditFileTool, SearchFilesTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.pdf import PDFInfoTool, ReadPDFTool
from nanobot.agent.tools.zotero_tools import ListZoteroCollectionsTool, SearchZoteroLibraryTool
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
    from nanobot.config.schema import ChannelsConfig, ExecToolConfig, SJTUConfig, VoiceConfig, WebSearchConfig
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
        sjtu_config: SJTUConfig | None = None,
        voice_config: VoiceConfig | None = None,
        embedding_model: str | None = None,
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

        # Auto-derive embedding model from main model when not explicitly set.
        if embedding_model is None:
            from nanobot.agent.semantic_memory import resolve_embedding_model
            embedding_model = resolve_embedding_model(self.model)
            if embedding_model:
                logger.info("Semantic memory: auto-selected embedding model {}", embedding_model)

        self._embedding_model = embedding_model
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

        self._sjtu_config = sjtu_config
        self._voice_config = voice_config
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
            embedding_model=embedding_model,
        )
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        extra_read = [BUILTIN_SKILLS_DIR] if allowed_dir else None
        self.tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir, extra_allowed_dirs=extra_read))
        for cls in (WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(SearchFilesTool(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(MultiEditFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(ReadPDFTool(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(PDFInfoTool(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(SummaryPDFFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
        self.tools.register(SearchZoteroLibraryTool())
        self.tools.register(ListZoteroCollectionsTool())
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

    async def _semantic_search(self, query: str) -> list[str] | None:
        """Return semantically relevant history entries, or None if disabled."""
        semantic = self.memory_consolidator.store.semantic
        if semantic is None or not query.strip():
            return None
        results = await semantic.search(query, k=5)
        return results if results else None

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
            relevant_history = await self._semantic_search(msg.content)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
                current_role=current_role,
                relevant_history=relevant_history,
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

        # /config and /voice multi-step state — intercept BEFORE slash-command check so
        # the password message never reaches the LLM or session history.
        metadata = session.metadata if isinstance(getattr(session, "metadata", None), dict) else {}
        if metadata.get("_config_step"):
            return await self._handle_config_step(msg, session)

        if metadata.get("_voice_config_step"):
            return await self._handle_voice_config_step(msg, session)

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
                "/vocab <words> — Store vocabulary words",
                "/word <word> — Query word details",
                "/paragraph <words> — Generate contextual paragraph",
                "/review — Review vocabulary",
                "/stats — Show learning statistics",
                "/config /xuanke — 设置选课社区账号（邮箱+密码）",
                "/config /canvas — 通过 JAccount OAuth2 登录 Canvas",
                "/xuanke <课程名/课号/教师> — 查询 SJTU 选课社区评价",
                "/canvas <问题> — 查询 Canvas 课程、作业、文件等（需先 /config /canvas）",
                "/summaryfile <pdf_path> [--exam-date YYYY-MM-DD] [--focus 'topic'] — 期末考试复习助手",
                "/voice — 语音播报设置（选择音色、开启/关闭）",
                "/profile — 查看你的个人画像（长期记忆摘要）",
                "/proactive — 查看/触发主动助手任务（HEARTBEAT.md）",
                "/help — Show available commands",
            ]
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines),
            )

        # /voice command
        if cmd == "/voice":
            return await self._handle_voice_config_start(msg, session)

        # /config subcommands
        if cmd == "/config /xuanke":
            return await self._handle_config_xuanke_start(msg, session)

        if cmd == "/config /canvas":
            return await self._handle_config_canvas_start(msg, session)

        # /xuanke command - SJTU 选课社区查询
        if cmd.startswith("/xuanke"):
            return await self._handle_xuanke_command(msg)

        # /canvas command - Canvas LMS
        if cmd.startswith("/canvas"):
            return await self._handle_canvas_command(msg, session)

        # /summaryfile command - 期末考试复习助手
        if cmd.startswith("/summaryfile"):
            return await self._handle_summaryfile_command(msg)

        # /vocab commands - vocabulary learning (hard intercept for 0 latency)
        vocab_commands = ("/vocab ", "/word ", "/paragraph", "/review", "/stats")
        if cmd.startswith(vocab_commands):
            return await self._handle_vocab_command(msg)

        # /profile command - 个人画像
        if cmd == "/profile":
            memory_content = self.memory_consolidator.store.read_long_term()
            if not memory_content.strip():
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="还没有积累到足够的对话，画像暂时为空。多聊几次就会有了！",
                )
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=f"## 你的个人画像\n\n{memory_content}",
            )

        # /proactive command - 主动助手任务列表
        if cmd == "/proactive":
            heartbeat_file = self.workspace / "HEARTBEAT.md"
            if heartbeat_file.exists():
                content = heartbeat_file.read_text(encoding="utf-8")
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"## 主动助手任务 (HEARTBEAT.md)\n\n{content}\n\n> 每隔一段时间自动检查并执行上述任务。",
                )
            else:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=(
                        "HEARTBEAT.md 还不存在。\n\n"
                        "在工作区创建 `HEARTBEAT.md`，写入你希望助手定期做的事，例如：\n\n"
                        "```markdown\n"
                        "- 检查今天的 Canvas 作业截止情况\n"
                        "- 提醒我复习昨天学过的知识点\n"
                        "- 如果工作区有新 PDF 就自动摘要\n"
                        "```"
                    ),
                )

        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))
        if message_tool := self.tools.get("message"):
            if isinstance(message_tool, MessageTool):
                message_tool.start_turn()

        history = session.get_history(max_messages=0)
        relevant_history = await self._semantic_search(msg.content)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
            relevant_history=relevant_history,
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

        if self._voice_config and self._voice_config.activate:
            self._schedule_background(self._speak(final_content))

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

    async def _handle_vocab_command(
        self, msg: InboundMessage
    ) -> OutboundMessage:
        """Handle vocabulary learning commands with hard interception."""
        from nanobot.skills.vocab import VocabHandler, VocabularyStore, VocabGenerator
        from nanobot.skills.vocab.handler import (
            format_word_result,
            format_paragraph_result,
            parse_command,
        )

        # Check if vocabulary feature is enabled
        if not self.channels_config or not getattr(self.channels_config, "vocabulary", None):
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="❌ 词汇学习功能未启用。请在配置中设置 channels.vocabulary.enabled = true",
            )

        vocab_config = self.channels_config.vocabulary
        if not vocab_config.enabled:
            return OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="❌ 词汇学习功能未启用。请在配置中设置 channels.vocabulary.enabled = true",
            )

        # Initialize handler
        store = VocabularyStore(db_path=vocab_config.db_path)
        generator = VocabGenerator(provider=self.provider, model=self.model)
        handler = VocabHandler(store=store, generator=generator)

        command, args = parse_command(msg.content)

        try:
            if command == "/vocab":
                if not args:
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="❌ 请提供要存储的单词，例如：/vocab apple banana cherry",
                    )

                result = await handler.handle_vocab(args, user_id=msg.sender_id)
                if result["success"]:
                    text = f"✅ 已存储 {result['stored_count']} 个单词"
                    if result["duplicate_count"] > 0:
                        text += f"（跳过 {result['duplicate_count']} 个重复）"
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=text,
                    )
                else:
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"❌ {result['error']}",
                    )

            elif command == "/word":
                if not args:
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="❌ 请提供要查询的单词，例如：/word resilience",
                    )

                word = args[0]
                result = await handler.handle_word(word, user_id=msg.sender_id)

                if result["success"]:
                    content = format_word_result(result["word"])
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=content,
                    )
                else:
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"❌ {result['error']}",
                    )

            elif command == "/paragraph":
                result = await handler.handle_paragraph(words=args, level="intermediate")

                if result["success"]:
                    content = format_paragraph_result(result["content"])
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=content,
                    )
                else:
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"❌ {result['error']}",
                    )

            elif command == "/review":
                result = await handler.handle_review(user_id=msg.sender_id)

                if result["success"]:
                    lines = ["📝 **复习单词**\n"]
                    for word_entry in result.get("review_words", []):
                        lines.append(f"**{word_entry['word']}** - {word_entry.get('meaning', 'N/A')}")

                    if result.get("message"):
                        lines.append(f"\n💡 {result['message']}")

                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content="\n".join(lines),
                    )

            elif command == "/stats":
                result = await handler.handle_stats(user_id=msg.sender_id)

                if result["success"]:
                    return OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"📊 **学习统计**\n\n总单词数：{result['total_words']}\n今日学习：{result['today_count']}",
                    )

        finally:
            store.close()

        # Fallback for unknown vocab commands
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="❌ 未知的词汇命令。支持的命令：/vocab, /word, /paragraph, /review, /stats",
        )

    async def _handle_canvas_command(
        self, msg: InboundMessage, session: Session
    ) -> OutboundMessage:
        """Handle /canvas <query> — run LLM with Canvas API tool."""
        from nanobot.agent.tools.canvas_api import CanvasAPITool

        parts = msg.content.strip().split(maxsplit=1)
        query = parts[1].strip() if len(parts) > 1 else ""

        if not query:
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=(
                    "📚 /canvas — 查询 Canvas (oc.sjtu.edu.cn)\n\n"
                    "用法：/canvas <你的问题>\n\n"
                    "示例：\n"
                    "  /canvas 我有哪些课程\n"
                    "  /canvas 大学物理有哪些作业\n"
                    "  /canvas 列出 CS2305 的所有文件"
                ),
            )

        cookie = self._sjtu_config.canvas_session if self._sjtu_config else ""
        if not cookie:
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="❌ 尚未登录 Canvas，请先运行 /config /canvas。",
            )

        canvas_tool = CanvasAPITool(cookie)
        # Temporarily register the tool
        self.tools.register(canvas_tool)
        try:
            history = session.get_history(max_messages=0)
            canvas_context = (
                "[Canvas 模式] 你可以使用 canvas_api 工具访问 SJTU Canvas LMS (oc.sjtu.edu.cn/api/v1/)。"
                "根据用户问题调用合适的 Canvas API 获取数据，再用中文给出清晰回答。\n\n"
                f"用户问题：{query}"
            )
            initial_messages = self.context.build_messages(
                history=history,
                current_message=canvas_context,
                channel=msg.channel,
                chat_id=msg.chat_id,
            )

            async def _progress(content: str, *, tool_hint: bool = False) -> None:
                meta = dict(msg.metadata or {})
                meta["_progress"] = True
                meta["_tool_hint"] = tool_hint
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
                ))

            final_content, _, all_msgs = await self._run_agent_loop(
                initial_messages, on_progress=_progress,
            )
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
        finally:
            self.tools.unregister("canvas_api")

        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content=final_content or "❌ 未能获取 Canvas 数据，请重试。",
        )

    async def _handle_config_canvas_start(
        self, msg: InboundMessage, session: Session
    ) -> OutboundMessage:
        """Step 1: Generate JAccount authorize URL and ask user to paste callback."""
        from nanobot.agent.canvas import get_authorize_url
        try:
            url = await get_authorize_url()
        except Exception as exc:
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=f"❌ 获取授权链接失败：{exc}",
            )
        session.metadata["_config_step"] = "canvas_callback"
        self.sessions.save(session)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=(
                "🎓 Canvas (oc.sjtu.edu.cn) JAccount 登录\n\n"
                "**第一步**：在浏览器打开以下链接，用 JAccount 完成登录：\n\n"
                f"`{url}`\n\n"
                "**第二步**：登录成功后，浏览器跳转到 Canvas 主页。"
                "打开浏览器开发者工具（F12）→ Application → Cookies → `https://oc.sjtu.edu.cn`，"
                "找到名为 `_normandy_session` 的 cookie，复制其 Value 粘贴回来。"
            ),
        )

    async def _handle_config_step(
        self, msg: InboundMessage, session: Session
    ) -> OutboundMessage:
        """Handle subsequent messages in any /config flow."""
        from nanobot.config.loader import get_config_path, load_config, save_config

        step = session.metadata.get("_config_step")
        text = msg.content.strip()

        # --- canvas session cookie ---
        if step == "canvas_callback":
            session.metadata.pop("_config_step", None)
            self.sessions.save(session)

            cookie = text.strip()
            if not cookie:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="❌ Cookie 为空，请重试 /config /canvas。",
                )
            try:
                config = load_config(get_config_path())
                config.sjtu.canvas_session = cookie
                save_config(config, get_config_path())
                if self._sjtu_config is not None:
                    self._sjtu_config.canvas_session = cookie
            except Exception as exc:
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"❌ 保存失败：{exc}",
                )
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="✅ Canvas session 已保存，可以使用 Canvas 功能了。",
            )

        # --- xuanke username/password ---
        if step == "xuanke_username":
            session.metadata["_config_step"] = "xuanke_password"
            session.metadata["_config_pending_username"] = text
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content="请输入你的 course.sjtu.plus 密码：",
            )

        if step == "xuanke_password":
            username = session.metadata.pop("_config_pending_username", "")
            session.metadata.pop("_config_step", None)
            try:
                config = load_config(get_config_path())
                config.sjtu.jaccount_username = username
                config.sjtu.jaccount_password = text
                save_config(config, get_config_path())
                if self._sjtu_config is not None:
                    self._sjtu_config.jaccount_username = username
                    self._sjtu_config.jaccount_password = text
            except Exception as exc:
                self.sessions.save(session)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"❌ 保存失败：{exc}",
                )
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=f"✅ 选课社区账号已保存（用户名：{username}）\n\n现在可以使用 /xuanke 查询课程评价了。",
            )

        # Unknown step — clear state
        session.metadata.pop("_config_step", None)
        session.metadata.pop("_config_pending_username", None)
        self.sessions.save(session)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content="❌ 设置流程出错，已重置。",
        )

    # ── Voice config ──────────────────────────────────────────────────────────

    _VOICE_OPTIONS: list[tuple[str, str]] = [
        ("zh-CN-XiaoxiaoNeural", "晓晓（女，温柔）"),
        ("zh-CN-XiaoyiNeural",   "晓伊（女，活泼）"),
        ("zh-CN-YunxiNeural",    "云希（男，自然）"),
        ("zh-CN-YunjianNeural",  "云健（男，沉稳）"),
        ("zh-CN-YunyangNeural",  "云扬（男，新闻）"),
        ("zh-CN-ManboNeural",    "曼波·小马（Manbo TTS）"),
    ]

    _MANBO_TTS_API = "https://api.h473fd122.nyat.app:41939/tts"
    _MANBO_TTS_KEY = "dayun-web-test"

    async def _handle_voice_config_start(
        self, msg: InboundMessage, session: Session
    ) -> OutboundMessage:
        """Step 1 of /voice: show available voices and ask user to pick."""
        vc = self._voice_config
        current_voice = vc.voice_name if vc else "zh-CN-XiaoxiaoNeural"
        current_status = "开启 ✅" if (vc and vc.activate) else "关闭 ❌"

        lines = ["🔊 语音播报设置", "", f"当前状态：{current_status}    当前音色：{current_voice}", ""]
        lines.append("请选择音色（输入序号）：")
        for i, (name, desc) in enumerate(self._VOICE_OPTIONS, 1):
            marker = " ◀ 当前" if name == current_voice else ""
            lines.append(f"  {i}. {desc}  ({name}){marker}")

        session.metadata["_voice_config_step"] = "pick_voice"
        self.sessions.save(session)
        return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id, content="\n".join(lines))

    async def _handle_voice_config_step(
        self, msg: InboundMessage, session: Session
    ) -> OutboundMessage:
        """Handle subsequent messages in the /voice config flow."""
        from nanobot.config.loader import get_config_path, load_config, save_config

        step = session.metadata.get("_voice_config_step")
        text = msg.content.strip()

        if step == "pick_voice":
            try:
                idx = int(text) - 1
                if not (0 <= idx < len(self._VOICE_OPTIONS)):
                    raise ValueError
            except ValueError:
                session.metadata.pop("_voice_config_step", None)
                self.sessions.save(session)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="❌ 无效选项，已退出设置。请重新输入 /voice。",
                )
            chosen_name, chosen_desc = self._VOICE_OPTIONS[idx]
            session.metadata["_voice_pending_name"] = chosen_name
            session.metadata["_voice_config_step"] = "pick_activate"
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=(
                    f"✅ 已选择音色：{chosen_desc}\n\n"
                    "是否开启语音播报？\n"
                    "  1. 开启\n"
                    "  2. 关闭"
                ),
            )

        if step == "pick_activate":
            chosen_name = session.metadata.pop("_voice_pending_name", "zh-CN-XiaoxiaoNeural")
            session.metadata.pop("_voice_config_step", None)

            if text in ("1", "开启", "on", "true", "yes"):
                activate = True
            elif text in ("2", "关闭", "off", "false", "no"):
                activate = False
            else:
                self.sessions.save(session)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="❌ 无效选项，已退出设置。请重新输入 /voice。",
                )

            try:
                config = load_config(get_config_path())
                config.voice.voice_name = chosen_name
                config.voice.activate = activate
                save_config(config, get_config_path())
                if self._voice_config is not None:
                    self._voice_config.voice_name = chosen_name
                    self._voice_config.activate = activate
            except Exception as exc:
                self.sessions.save(session)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=f"❌ 保存失败：{exc}",
                )

            status_str = "开启 ✅" if activate else "关闭 ❌"
            _, desc = next((v for v in self._VOICE_OPTIONS if v[0] == chosen_name), (chosen_name, chosen_name))
            self.sessions.save(session)
            return OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id,
                content=f"✅ 语音设置已保存\n\n音色：{desc}\n状态：{status_str}",
            )

        # Unknown step
        session.metadata.pop("_voice_config_step", None)
        session.metadata.pop("_voice_pending_name", None)
        self.sessions.save(session)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id,
            content="❌ 语音设置流程出错，已重置。",
        )

    async def _speak(self, text: str) -> None:
        """Pipeline: LLM summarize → edge-tts → play mp3 (background, overwrites each time)."""
        import sys

        tts_path = Path.home() / ".nanobot" / "tts_output.mp3"
        tts_path.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: LLM → 30-char brief
        # Use a minimal, topic-neutral prompt so Claude doesn't editorialize
        try:
            resp = await self.provider.chat_with_retry(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "提取以下内容的核心信息，用不超过30个汉字输出一句话。"
                            "只输出那句话本身，不要标点以外的任何前缀、后缀或解释。"
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=80,
                temperature=0.1,
            )
            brief = (resp.content or "").strip()
            # Clamp to 30 Chinese chars as hard limit
            brief = brief[:30]
        except Exception as exc:
            logger.warning("TTS summarization failed: {}; falling back to truncation", exc)
            brief = text[:30]

        if not brief:
            return

        voice_name = self._voice_config.voice_name if self._voice_config else "zh-CN-XiaoxiaoNeural"
        logger.info("TTS brief: {} | voice: {}", brief, voice_name)

        # Step 2: generate mp3 — Manbo API or edge-tts
        if voice_name == "zh-CN-ManboNeural":
            tts_path = tts_path.with_suffix(".wav")

            def _fetch_manbo(text: str, out_path: Path) -> str | None:
                """POST text → receive WAV bytes directly. Runs in thread executor."""
                import requests as _req
                try:
                    resp = _req.post(
                        self._MANBO_TTS_API,
                        headers={"Content-Type": "application/json", "X-API-Key": self._MANBO_TTS_KEY},
                        json={"text": text},
                        timeout=15,
                    )
                    if resp.status_code != 200:
                        return f"HTTP {resp.status_code}: {resp.text[:200]}"
                    out_path.write_bytes(resp.content)
                    return None
                except Exception as exc:
                    return str(exc)

            loop = asyncio.get_event_loop()
            err = await loop.run_in_executor(None, _fetch_manbo, brief, tts_path)
            if err:
                logger.warning("Manbo TTS failed: {}", err)
                return
        else:
            # edge-tts
            try:
                proc = await asyncio.create_subprocess_exec(
                    "edge-tts",
                    "--text", brief,
                    "--voice", voice_name,
                    "--write-media", str(tts_path),
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                )
                _, stderr_data = await proc.communicate()
                if proc.returncode != 0:
                    logger.warning("edge-tts error (code {}): {}", proc.returncode,
                                   stderr_data.decode(errors="replace").strip())
                    return
            except FileNotFoundError:
                logger.warning("edge-tts not found — run: pip install edge-tts")
                return
            except Exception as exc:
                logger.warning("edge-tts failed: {}", exc)
                return

        # Step 3: play mp3
        player_cmd = ["afplay", str(tts_path)] if sys.platform == "darwin" else ["mpg123", "-q", str(tts_path)]
        try:
            play_proc = await asyncio.create_subprocess_exec(
                *player_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await play_proc.wait()
        except FileNotFoundError:
            logger.warning("Audio player not found: {}", player_cmd[0])
        except Exception as exc:
            logger.warning("TTS playback failed: {}", exc)

    # ── /config /xuanke ───────────────────────────────────────────────────────

    async def _handle_config_xuanke_start(
        self, msg: InboundMessage, session: Session
    ) -> OutboundMessage:
        """Start the /config /xuanke credential collection flow."""
        session.metadata["_config_step"] = "xuanke_username"
        self.sessions.save(session)
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content="🔐 选课社区账号设置\n\n请输入你的 course.sjtu.plus 邮箱：",
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
    
    def _get_oc_base_url(self) -> str:
        """Return the course community base URL.

        Uses the official course.sjtu.plus when JAccount credentials are
        configured, otherwise falls back to the community mirror.
        """
        if (
            self._sjtu_config
            and self._sjtu_config.jaccount_username
            and self._sjtu_config.jaccount_password
        ):
            return "https://course.sjtu.plus"
        return "http://nas.oct0pus.top:41735"

    async def _make_course_session(self) -> tuple[Any, str]:
        """Return (session_or_None, base_url). Session is already logged in if credentials exist."""
        from nanobot.agent.jaccount import CourseSJTUSession

        base_url = self._get_oc_base_url()
        if not base_url.startswith("https://course.sjtu.plus"):
            return None, base_url

        session = CourseSJTUSession(
            self._sjtu_config.jaccount_username,  # type: ignore[union-attr]
            self._sjtu_config.jaccount_password,  # type: ignore[union-attr]
        )
        ok = await session.login()
        if not ok:
            logger.warning("course.sjtu.plus login failed; falling back to mirror API")
            return None, "http://nas.oct0pus.top:41735"
        return session, base_url

    async def _search_courses_batch(self, query: str, max_pages: int = 100) -> list[dict]:
        """Search courses using the course-in-review search API (server-side filtering)."""
        session, base_url = await self._make_course_session()
        use_official = base_url.startswith("https://course.sjtu.plus")

        try:
            if use_official and session:
                # Official API: use server-side search endpoint
                resp = await session.get(
                    f"{base_url}/api/course-in-review/",
                    params={"q": query},
                )
                if resp.status_code == 200:
                    return resp.json().get("results", [])
                return []

            # Mirror fallback: client-side filtering across all pages
            query_lower = query.lower()
            all_courses: list[dict] = []

            async def _fetch_mirror(page: int) -> list[dict]:
                try:
                    async with httpx.AsyncClient(timeout=15.0) as c:
                        r = await c.get(
                            f"{base_url}/api/course/",
                            params={"page": page, "page_size": 20},
                        )
                        if r.status_code == 200:
                            return r.json().get("results", [])
                except Exception:
                    pass
                return []

            first = await _fetch_mirror(1)
            # get total
            async with httpx.AsyncClient(timeout=15.0) as c:
                r0 = await c.get(f"{base_url}/api/course/", params={"page": 1, "page_size": 20})
            total = r0.json().get("count", 0) if r0.status_code == 200 else 0
            total_pages = min(max_pages, (total + 19) // 20)

            all_courses.extend(first)
            tasks = [_fetch_mirror(p) for p in range(2, total_pages + 1)]
            for batch in [tasks[i:i+50] for i in range(0, len(tasks), 50)]:
                for results in await asyncio.gather(*batch):
                    all_courses.extend(results)

            return [
                c for c in all_courses
                if query_lower in c.get("name", "").lower()
                or query_lower in c.get("code", "").lower()
                or query_lower in c.get("teacher", "").lower()
            ]
        finally:
            if session:
                await session.__aexit__(None, None, None)

    async def _get_course_reviews(self, course: dict) -> dict:
        """Get reviews for a specific course."""
        session, base_url = await self._make_course_session()
        course_id = course["id"]
        reviews: list[dict] = []

        try:
            params: dict = {"size": 10} if base_url.startswith("https://course.sjtu.plus") else {"page_size": 10}
            if session:
                async with session:
                    resp = await session.get(f"{base_url}/api/course/{course_id}/review/", params=params)
            else:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(f"{base_url}/api/course/{course_id}/review/", params=params)

            if resp.status_code == 200:
                reviews = resp.json().get("results", [])
        except Exception as exc:
            logger.warning("Failed to fetch reviews for course {}: {}", course_id, exc)

        return {"course": course, "reviews": reviews}
    
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
