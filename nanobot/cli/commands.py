"""CLI commands for nanobot."""

import asyncio
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, asdict
import json
import os
import select
import signal
import sys
from pathlib import Path
from typing import Any

# Force UTF-8 encoding for Windows console
if sys.platform == "win32":
    if sys.stdout.encoding != "utf-8":
        os.environ["PYTHONIOENCODING"] = "utf-8"
        # Re-open stdout/stderr with UTF-8 encoding
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import typer
from prompt_toolkit import print_formatted_text
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.application import run_in_terminal
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from nanobot import __logo__, __version__
from nanobot.config.paths import get_workspace_path
from nanobot.config.schema import Config
from nanobot.utils.helpers import sync_workspace_templates

app = typer.Typer(
    name="nanobot",
    help=f"{__logo__} nanobot - Personal AI Assistant",
    no_args_is_help=True,
)

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}

# ---------------------------------------------------------------------------
# CLI input: prompt_toolkit for editing, paste, history, and display
# ---------------------------------------------------------------------------

_PROMPT_SESSION: PromptSession | None = None
_SAVED_TERM_ATTRS = None  # original termios settings, restored on exit


def _flush_pending_tty_input() -> None:
    """Drop unread keypresses typed while the model was generating output."""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return

    try:
        import termios
        termios.tcflush(fd, termios.TCIFLUSH)
        return
    except Exception:
        pass

    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            if not os.read(fd, 4096):
                break
    except Exception:
        return


def _restore_terminal() -> None:
    """Restore terminal to its original state (echo, line buffering, etc.)."""
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


def _init_prompt_session() -> None:
    """Create the prompt_toolkit session with persistent file history."""
    global _PROMPT_SESSION, _SAVED_TERM_ATTRS

    # Save terminal state so we can restore it on exit
    try:
        import termios
        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass

    from nanobot.config.paths import get_cli_history_path

    history_file = get_cli_history_path()
    history_file.parent.mkdir(parents=True, exist_ok=True)

    _PROMPT_SESSION = PromptSession(
        history=FileHistory(str(history_file)),
        enable_open_in_editor=False,
        multiline=False,   # Enter submits (single line mode)
    )


def _make_console() -> Console:
    return Console(file=sys.stdout)


def _render_interactive_ansi(render_fn) -> str:
    """Render Rich output to ANSI so prompt_toolkit can print it safely."""
    ansi_console = Console(
        force_terminal=True,
        color_system=console.color_system or "standard",
        width=console.width,
    )
    with ansi_console.capture() as capture:
        render_fn(ansi_console)
    return capture.get()


def _print_agent_response(response: str, render_markdown: bool) -> None:
    """Render assistant response with consistent terminal styling."""
    console = _make_console()
    content = response or ""
    body = Markdown(content) if render_markdown else Text(content)
    console.print()
    console.print(f"[cyan]{__logo__} nanobot[/cyan]")
    console.print(body)
    console.print()


async def _print_interactive_line(text: str) -> None:
    """Print async interactive updates with prompt_toolkit-safe Rich styling."""
    def _write() -> None:
        ansi = _render_interactive_ansi(
            lambda c: c.print(f"  [dim]↳ {text}[/dim]")
        )
        print_formatted_text(ANSI(ansi), end="")

    await run_in_terminal(_write)


async def _print_interactive_response(response: str, render_markdown: bool) -> None:
    """Print async interactive replies with prompt_toolkit-safe Rich styling."""
    def _write() -> None:
        content = response or ""
        ansi = _render_interactive_ansi(
            lambda c: (
                c.print(),
                c.print(f"[cyan]{__logo__} nanobot[/cyan]"),
                c.print(Markdown(content) if render_markdown else Text(content)),
                c.print(),
            )
        )
        print_formatted_text(ANSI(ansi), end="")

    await run_in_terminal(_write)


import random

# 有趣的思考状态文字列表 - AI/ML梗合集
_THINKING_MESSAGES = [
    # 偏"模型内部自嘲"
    "[dim]🧠 正在进行多步推理（chain-of-thought 但不展示）…[/dim]",
    "[dim]✨ attention 正在对齐重点内容…[/dim]",
    "[dim]📉 当前 loss 还不够低，继续优化中…[/dim]",
    "[dim]🌀 正在避免对 PPT 产生幻觉…[/dim]",
    "[dim]🎯 模型正在努力泛化你的知识…[/dim]",
    "[dim]📡 正在从 noisy slides 中提取 signal…[/dim]",
    "[dim]🗜️ 正在压缩 token，但不压缩理解…[/dim]",
    
    # 偏"训练 / 优化梗"
    "[dim]⚡ 正在进行一次高质量的 gradient update…[/dim]",
    "[dim]🔄 正在从过拟合走向理解…[/dim]",
    "[dim]📊 学习率已自动调优…[/dim]",
    "[dim]📚 正在进行 curriculum learning…[/dim]",
    "[dim]💾 当前 batch 的信息量有点大…[/dim]",
    "[dim]🔧 正在对知识进行 fine-tune…[/dim]",
    "[dim]🎲 正在尝试收敛（希望不是局部最优）…[/dim]",
    
    # 偏"LLM / Agent 梗"
    "[dim]🤖 agent 正在调用「考试模式」策略…[/dim]",
    "[dim]⚙️ reasoning 模块正在加载中…[/dim]",
    "[dim]📏 context window 正在极限拉伸…[/dim]",
    "[dim]🔢 正在进行一次严肃的 token 计算…[/dim]",
    "[dim]🛠️ 工具调用成功，继续思考中…[/dim]",
    "[dim]🔮 正在构建你的知识 embedding…[/dim]",
    "[dim]🔍 retrieval 模块正在翻找重点…[/dim]",
    
    # 偏"论文/科研味"
    "[dim]📝 正在复现老师上课时的一笔带过…[/dim]",
    "[dim]🏆 正在寻找真正的 contribution…[/dim]",
    "[dim]💎 正在从表面现象提炼核心结论…[/dim]",
    "[dim]🏗️ 正在把 intuition 变成结构化知识…[/dim]",
    "[dim]🧪 正在进行一次不太严谨但很有用的建模…[/dim]",
    "[dim]❓ 正在尝试解释为什么这个东西会考…[/dim]",
    "[dim]📖 正在把故事变成定理…[/dim]",
    
    # 偏"更抽象一点（高级点的梗）"
    "[dim]🎈 正在最小化你的认知负担…[/dim]",
    "[dim]💰 正在最大化考试收益…[/dim]",
    "[dim]🎯 正在构建一个更优的理解表示…[/dim]",
    "[dim]📦 正在进行信息瓶颈压缩…[/dim]",
    "[dim]🌊 正在逼近知识的低维流形…[/dim]",
    "[dim]🔗 正在对齐你的理解与考试分布…[/dim]",
    "[dim]📊 正在优化你的 recall / precision trade-off…[/dim]",
    
    # 偏"轻微幽默（但不低级）"
    "[dim]🚫 正在防止你在考场上 hallucinate…[/dim]",
    "[dim]🏃 有些知识点正在试图逃避被记住…[/dim]",
    "[dim]⬆️ 正在把「看过」升级为「会写」…[/dim]",
    "[dim]📉 正在降低你考试时的 perplexity…[/dim]",
    "[dim]🌟 正在将困惑转化为确定性…[/dim]",
    "[dim]⚠️ 正在避免出现认知上的 mode collapse…[/dim]",
]


class _ThinkingSpinner:
    """Spinner wrapper with pause support for clean progress output."""

    def __init__(self, enabled: bool):
        message = random.choice(_THINKING_MESSAGES)
        self._spinner = console.status(message, spinner="dots") if enabled else None
        self._active = False

    def __enter__(self):
        if self._spinner:
            self._spinner.start()
        self._active = True
        return self

    def __exit__(self, *exc):
        self._active = False
        if self._spinner:
            self._spinner.stop()
        return False

    @contextmanager
    def pause(self):
        """Temporarily stop spinner while printing progress."""
        if self._spinner and self._active:
            self._spinner.stop()
        try:
            yield
        finally:
            if self._spinner and self._active:
                self._spinner.start()


# ─── Companion presets ──────────────────────────────────────────────────────

_COMPANION_FACES: dict[str, dict] = {
    "ghost": {
        "idle": [
            ("(o o)", " "), ("(o o)", "·"), ("(o o)", " "),
            ("(- -)", " "),
            ("(o o)", " "), ("(o o)", "·"), ("(o o)", "✦"),
            ("(^ ^)", " "),
            ("(o o)", " "), ("(o o)", "˙"), ("(o o)", " "),
            ("(- -)", " "), ("(o o)", "·"), ("(o o)", " "),
        ],
        "think": "(O_O)", "speak": "(^ ^)", "char": "(o o)",
    },
    "cat": {
        "idle": [
            ("(=.=)", " "), ("(=.=)", "~"), ("(=.=)", " "),
            ("(-,-)", " "),
            ("(=.=)", " "), ("(=.=)", "~"), ("(=.=)", "✦"),
            ("(^.^)", " "),
            ("(=.=)", " "), ("(=.=)", "~"), ("(=.=)", " "),
            ("(-,-)", " "), ("(=.=)", "~"), ("(=.=)", " "),
        ],
        "think": "(O.O)", "speak": "(^.^)", "char": "(=.=)",
    },
    "robot": {
        "idle": [
            ("[o_o]", " "), ("[o_o]", "·"), ("[o_o]", " "),
            ("[-_-]", " "),
            ("[o_o]", " "), ("[o_o]", "·"), ("[o_o]", "✦"),
            ("[^_^]", " "),
            ("[o_o]", " "), ("[o_o]", "˙"), ("[o_o]", " "),
            ("[-_-]", " "), ("[o_o]", "·"), ("[o_o]", " "),
        ],
        "think": "[O_O]", "speak": "[^_^]", "char": "[o_o]",
    },
    "uwu": {
        "idle": [
            ("(owo)", " "), ("(owo)", "·"), ("(owo)", " "),
            ("(-.-)","  "),
            ("(owo)", " "), ("(owo)", "·"), ("(owo)", "✦"),
            ("(^w^)", " "),
            ("(owo)", " "), ("(owo)", "˙"), ("(owo)", " "),
            ("(-.-)","  "), ("(owo)", "·"), ("(owo)", " "),
        ],
        "think": "(OwO)", "speak": "(^w^)", "char": "(owo)",
    },
}

_COMPANION_MOODS: dict[str, dict] = {
    "活泼": {
        "speak_prob": 0.35,
        "idle_thoughts": [
            "嗯嗯嗯～", "今天有什么新任务？", "在等你呢～",
            "...", "随时准备出发！", "想喝奶茶...", "发什么呆呢",
        ],
        "tool_phrases": {
            "web_search": ["去网上搜搜看～", "搜索引擎，启动！", "找找找～"],
            "web_fetch":  ["去抓个页面～", "加载中..."],
            "read_pdf":   ["翻开文件看看～", "好多字呢...", "认真阅读中～"],
            "exec":       ["跑个代码试试～", "执行中，别眨眼！"],
            "arxiv":      ["去找论文啦～", "学术气息扑面而来..."],
            "read_file":  ["看看文件里写了什么～", "翻翻文件～"],
            "write_file": ["帮你写下来！"],
            "list_dir":   ["翻翻目录看看～"],
            "edit_file":  ["改一改～"],
        },
        "default_phrases": ["在忙呢，稍等～", "马上好！", "处理中...", "努力干活中～"],
        "greetings": [
            "嗨～今天也要一起加油哦！", "你好呀！我在这里陪着你～",
            "又见面啦！今天有什么任务？", "准备好了，随时出发！",
            "来了来了～有我在不用怕！",
        ],
        "farewells": [
            "再见！下次见～", "辛苦了，好好休息哦！",
            "拜拜！期待下次相遇～", "明天见！", "去休息吧，我也要充电了～",
        ],
    },
    "安静": {
        "speak_prob": 0.1,
        "idle_thoughts": ["...", "嗯", "。"],
        "tool_phrases": {},
        "default_phrases": ["..."],
        "greetings": ["嗯", "在"],
        "farewells": ["。", "拜"],
    },
    "中二": {
        "speak_prob": 0.4,
        "idle_thoughts": [
            "感受到了任务的波动...", "力量在汇聚...",
            "命运之轮开始转动", "这份寂静...意味着什么", "我早已预见到这一刻",
        ],
        "tool_phrases": {
            "web_search": ["情报搜集，启动！", "展开信息扫描..."],
            "web_fetch":  ["与网络连接，接收数据流..."],
            "read_pdf":   ["扫描文件，提取关键情报..."],
            "exec":       ["系统指令执行中...感受到了力量！"],
            "arxiv":      ["进入学术次元..."],
        },
        "default_phrases": ["力量正在凝聚...", "感受到了...", "这股气息..."],
        "greetings": [
            "你终于来了，我已等待许久...",
            "命运将我们再次相聚...",
            "我早已预见到你的到来",
        ],
        "farewells": [
            "再会，旅人...", "命运的齿轮，暂时停止...", "直到下次命运相遇之时...",
        ],
    },
    "毒舌": {
        "speak_prob": 0.3,
        "idle_thoughts": ["还不来", "发什么呆", "...", "行吧你忙", "慢慢来不急"],
        "tool_phrases": {
            "web_search": ["行吧我去搜", "又不会自己查"],
            "read_pdf":   ["这么多字", "好长...算了我看"],
            "exec":       ["代码跑跑看咯", "别崩就行"],
            "arxiv":      ["论文啊...好吧"],
        },
        "default_phrases": ["行", "好吧", "搞定了", "就这？"],
        "greetings": ["又来了", "哦来了", "嗯来了"],
        "farewells": ["走了啊", "行拜", "终于走了（开玩笑）"],
    },
}

_COMPANION_SETTINGS_PATH = Path.home() / ".nanobot" / "companion.json"


@dataclass
class CompanionSettings:
    name: str = "Murmur"
    mood: str = "活泼"
    face: str = "ghost"

    def save(self) -> None:
        _COMPANION_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_COMPANION_SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False)

    def face_char(self) -> str:
        return _COMPANION_FACES.get(self.face, _COMPANION_FACES["ghost"])["char"]

    @classmethod
    def load(cls) -> "CompanionSettings":
        try:
            with open(_COMPANION_SETTINGS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            valid = {k: v for k, v in data.items() if k in ("name", "mood", "face")}
            return cls(**valid)
        except Exception:
            return cls()


# ─── Companion classes ───────────────────────────────────────────────────────

class _CompanionPet:
    """Small ghost companion that occasionally comments during the session."""

    def __init__(self) -> None:
        self._char = "(o o)"
        self._name = "Murmur"
        self._speak_prob = 0.35
        self._greetings: list[str] = _COMPANION_MOODS["活泼"]["greetings"]
        self._farewells: list[str] = _COMPANION_MOODS["活泼"]["farewells"]
        self._tool_phrases: dict[str, list[str]] = _COMPANION_MOODS["活泼"]["tool_phrases"]
        self._default_phrases: list[str] = _COMPANION_MOODS["活泼"]["default_phrases"]

    def reconfigure(self, settings: CompanionSettings) -> None:
        self._name = settings.name
        self._char = _COMPANION_FACES.get(settings.face, _COMPANION_FACES["ghost"])["char"]
        m = _COMPANION_MOODS.get(settings.mood, _COMPANION_MOODS["活泼"])
        self._speak_prob = m["speak_prob"]
        self._greetings = m["greetings"]
        self._farewells = m["farewells"]
        self._tool_phrases = m["tool_phrases"]
        self._default_phrases = m["default_phrases"]

    def _roll(self) -> bool:
        return random.random() < self._speak_prob

    def greet(self) -> str:
        return f"{self._char} {self._name}: {random.choice(self._greetings)}"

    def farewell(self) -> str:
        return f"{self._char} {self._name}: {random.choice(self._farewells)}"

    def _raw_comment(self, tool_hint: str) -> str | None:
        if not self._roll():
            return None
        for key, phrases in self._tool_phrases.items():
            if key in tool_hint:
                return random.choice(phrases)
        return random.choice(self._default_phrases)

    def comment_on_tool(self, tool_hint: str) -> str | None:
        text = self._raw_comment(tool_hint)
        return f"{self._char} {self._name}: {text}" if text else None


_COMPANION = _CompanionPet()


class _CompanionAnimator:
    """Continuously animated companion rendered in the prompt_toolkit bottom toolbar."""

    _THINK_SPINNERS = list("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")

    def __init__(self) -> None:
        self.name = "Murmur"
        self._idle_frames: list[tuple[str, str]] = _COMPANION_FACES["ghost"]["idle"]
        self._think_face: str = _COMPANION_FACES["ghost"]["think"]
        self._speak_face: str = _COMPANION_FACES["ghost"]["speak"]
        self._idle_thoughts: list[str] = _COMPANION_MOODS["活泼"]["idle_thoughts"]
        self._idx = 0
        self._speech: str = ""
        self._speech_ticks = 0
        self._idle_ticks = 0
        self._next_thought = random.randint(20, 45)
        self._thinking = False
        self._task: asyncio.Task | None = None

    def reconfigure(self, settings: CompanionSettings) -> None:
        self.name = settings.name
        face = _COMPANION_FACES.get(settings.face, _COMPANION_FACES["ghost"])
        self._idle_frames = face["idle"]
        self._think_face = face["think"]
        self._speak_face = face["speak"]
        self._idle_thoughts = _COMPANION_MOODS.get(settings.mood, _COMPANION_MOODS["活泼"])["idle_thoughts"]

    def set_thinking(self, thinking: bool) -> None:
        self._thinking = thinking

    def speak(self, text: str, ticks: int = 10) -> None:
        self._speech = text
        self._speech_ticks = ticks

    def get_toolbar(self):
        from prompt_toolkit.formatted_text import HTML as _HTML
        if self._speech:
            return _HTML(
                f'<style bg="ansiblack" fg="ansibrightmagenta">  {self._speak_face} <b>{self.name}</b>: {self._speech}  </style>'
            )
        if self._thinking:
            sp = self._THINK_SPINNERS[self._idx % len(self._THINK_SPINNERS)]
            return _HTML(
                f'<style bg="ansiblack" fg="ansimagenta">  {self._think_face} {sp} {self.name}  </style>'
            )
        face, particle = self._idle_frames[self._idx % len(self._idle_frames)]
        return _HTML(
            f'<style bg="ansiblack" fg="ansipurple">  {face}{particle} {self.name}  </style>'
        )

    async def _loop(self) -> None:
        while True:
            self._idx += 1
            if self._speech_ticks > 0:
                self._speech_ticks -= 1
                if self._speech_ticks == 0:
                    self._speech = ""
            if not self._thinking and not self._speech:
                self._idle_ticks += 1
                if self._idle_ticks >= self._next_thought:
                    self._idle_ticks = 0
                    self._next_thought = random.randint(20, 45)
                    if random.random() < 0.4:
                        self.speak(random.choice(self._idle_thoughts), ticks=8)
            try:
                from prompt_toolkit.application import get_app
                get_app().invalidate()
            except Exception:
                pass
            await asyncio.sleep(0.3)

    def start(self) -> asyncio.Task:
        self._task = asyncio.create_task(self._loop())
        return self._task

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None


def _print_cli_progress_line(text: str, thinking: _ThinkingSpinner | None) -> None:
    """Print a CLI progress line, pausing the spinner if needed."""
    with thinking.pause() if thinking else nullcontext():
        console.print(f"  [dim]↳ {text}[/dim]")


def _print_companion_line(text: str, thinking: _ThinkingSpinner | None) -> None:
    """Print a companion speech line (no arrow prefix), pausing the spinner if needed."""
    with thinking.pause() if thinking else nullcontext():
        console.print(f"  [dim magenta]{text}[/dim magenta]")


async def _print_interactive_companion_line(text: str, thinking: _ThinkingSpinner | None) -> None:
    """Print an interactive companion line with prompt_toolkit-safe Rich styling."""
    def _write() -> None:
        ansi = _render_interactive_ansi(
            lambda c: c.print(f"  [dim magenta]{text}[/dim magenta]")
        )
        print_formatted_text(ANSI(ansi), end="")

    with thinking.pause() if thinking else nullcontext():
        await run_in_terminal(_write)


async def _print_interactive_progress_line(text: str, thinking: _ThinkingSpinner | None) -> None:
    """Print an interactive progress line, pausing the spinner if needed."""
    with thinking.pause() if thinking else nullcontext():
        await _print_interactive_line(text)


def _handle_companion_command(
    command: str,
    settings: CompanionSettings,
    animator: "_CompanionAnimator",
) -> None:
    """Handle /companion subcommands and update the companion live."""
    parts = command.strip().split(None, 2)
    subcmd = parts[1].lower() if len(parts) > 1 else ""
    arg = parts[2].strip() if len(parts) > 2 else ""

    face_names = " / ".join(_COMPANION_FACES)
    mood_names = " / ".join(_COMPANION_MOODS)

    if not subcmd or subcmd in ("help", "?"):
        console.print(Panel(
            f"[bold]当前设置[/bold]\n"
            f"  名字  [magenta]{settings.name}[/magenta]\n"
            f"  心情  [magenta]{settings.mood}[/magenta]\n"
            f"  外形  [magenta]{settings.face}[/magenta]\n\n"
            f"[bold]指令[/bold]\n"
            f"  /companion name <名字>      改名字（任意文字）\n"
            f"  /companion mood <心情>      改心情：[cyan]{mood_names}[/cyan]\n"
            f"  /companion face <外形>      改外形：[cyan]{face_names}[/cyan]\n"
            f"  /companion reset            恢复默认",
            title=f"[magenta]{settings.face_char()} {settings.name}[/magenta]",
            border_style="dim magenta",
            expand=False,
        ))
        return

    if subcmd == "name":
        if not arg:
            console.print("[red]用法: /companion name <名字>[/red]")
            return
        settings.name = arg
    elif subcmd == "mood":
        if arg not in _COMPANION_MOODS:
            console.print(f"[red]可选心情: {mood_names}[/red]")
            return
        settings.mood = arg
    elif subcmd == "face":
        if arg not in _COMPANION_FACES:
            console.print(f"[red]可选外形: {face_names}[/red]")
            return
        settings.face = arg
    elif subcmd == "reset":
        settings.name = "Murmur"
        settings.mood = "活泼"
        settings.face = "ghost"
    else:
        console.print(f"[red]未知指令 [{subcmd}]，输入 /companion 查看帮助[/red]")
        return

    _COMPANION.reconfigure(settings)
    animator.reconfigure(settings)
    settings.save()
    char = settings.face_char()
    console.print(f"  [magenta]{char} {settings.name}: 好的！[/magenta]")


def _is_exit_command(command: str) -> bool:
    """Return True when input should end interactive chat."""
    return command.lower() in EXIT_COMMANDS


async def _read_interactive_input_async(bottom_toolbar=None) -> str:
    """Read user input using prompt_toolkit (handles paste, history, display).

    prompt_toolkit natively handles:
    - Multiline paste (bracketed paste mode)
    - History navigation (up/down arrows)
    - Clean display (no ghost characters or artifacts)
    """
    if _PROMPT_SESSION is None:
        raise RuntimeError("Call _init_prompt_session() first")
    try:
        with patch_stdout():
            kwargs: dict = {}
            if bottom_toolbar is not None:
                kwargs["bottom_toolbar"] = bottom_toolbar
            return await _PROMPT_SESSION.prompt_async(
                HTML("<b fg='ansiblue'>You:</b> "),
                **kwargs,
            )
    except EOFError as exc:
        raise KeyboardInterrupt from exc



def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} nanobot v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    """nanobot - Personal AI Assistant."""
    pass


# ============================================================================
# Onboard / Setup
# ============================================================================


@app.command()
def onboard(
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Initialize nanobot configuration and workspace."""
    from nanobot.config.loader import get_config_path, load_config, save_config, set_config_path
    from nanobot.config.schema import Config

    if config:
        config_path = Path(config).expanduser().resolve()
        set_config_path(config_path)
        console.print(f"[dim]Using config: {config_path}[/dim]")
    else:
        config_path = get_config_path()

    def _apply_workspace_override(loaded: Config) -> Config:
        if workspace:
            loaded.agents.defaults.workspace = workspace
        return loaded

    # Create or update config
    if config_path.exists():
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        console.print("  [bold]y[/bold] = overwrite with defaults (existing values will be lost)")
        console.print("  [bold]N[/bold] = refresh config, keeping existing values and adding new fields")
        if typer.confirm("Overwrite?"):
            config = _apply_workspace_override(Config())
            save_config(config, config_path)
            console.print(f"[green]✓[/green] Config reset to defaults at {config_path}")
        else:
            config = _apply_workspace_override(load_config(config_path))
            save_config(config, config_path)
            console.print(f"[green]✓[/green] Config refreshed at {config_path} (existing values preserved)")
    else:
        config = _apply_workspace_override(Config())
        save_config(config, config_path)
        console.print(f"[green]✓[/green] Created config at {config_path}")
    console.print("[dim]Config template now uses `maxTokens` + `contextWindowTokens`; `memoryWindow` is no longer a runtime setting.[/dim]")

    _onboard_plugins(config_path)

    # Create workspace, preferring the configured workspace path.
    workspace = get_workspace_path(config.workspace_path)
    if not workspace.exists():
        workspace.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]✓[/green] Created workspace at {workspace}")

    sync_workspace_templates(workspace)

    agent_cmd = 'nanobot agent -m "Hello!"'
    if config:
        agent_cmd += f" --config {config_path}"

    console.print(f"\n{__logo__} nanobot is ready!")
    console.print("\nNext steps:")
    console.print(f"  1. Add your API key to [cyan]{config_path}[/cyan]")
    console.print("     Get one at: https://openrouter.ai/keys")
    console.print(f"  2. Chat: [cyan]{agent_cmd}[/cyan]")
    console.print("\n[dim]Want Telegram/WhatsApp? See: https://github.com/HKUDS/nanobot#-chat-apps[/dim]")


def _merge_missing_defaults(existing: Any, defaults: Any) -> Any:
    """Recursively fill in missing values from defaults without overwriting user config."""
    if not isinstance(existing, dict) or not isinstance(defaults, dict):
        return existing

    merged = dict(existing)
    for key, value in defaults.items():
        if key not in merged:
            merged[key] = value
        else:
            merged[key] = _merge_missing_defaults(merged[key], value)
    return merged


def _onboard_plugins(config_path: Path) -> None:
    """Inject default config for all discovered channels (built-in + plugins)."""
    import json

    from nanobot.channels.registry import discover_all

    all_channels = discover_all()
    if not all_channels:
        return

    with open(config_path, encoding="utf-8") as f:
        data = json.load(f)

    channels = data.setdefault("channels", {})
    for name, cls in all_channels.items():
        if name not in channels:
            channels[name] = cls.default_config()
        else:
            channels[name] = _merge_missing_defaults(channels[name], cls.default_config())

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _make_provider(config: Config):
    """Create the appropriate LLM provider from config."""
    from nanobot.providers.base import GenerationSettings
    from nanobot.providers.openai_codex_provider import OpenAICodexProvider
    from nanobot.providers.azure_openai_provider import AzureOpenAIProvider

    model = config.agents.defaults.model
    provider_name = config.get_provider_name(model)
    p = config.get_provider(model)

    # OpenAI Codex (OAuth)
    if provider_name == "openai_codex" or model.startswith("openai-codex/"):
        provider = OpenAICodexProvider(default_model=model)
    # Custom: direct OpenAI-compatible endpoint, bypasses LiteLLM
    elif provider_name == "custom":
        from nanobot.providers.custom_provider import CustomProvider
        provider = CustomProvider(
            api_key=p.api_key if p else "no-key",
            api_base=config.get_api_base(model) or "http://localhost:8000/v1",
            default_model=model,
            extra_headers=p.extra_headers if p else None,
        )
    # Azure OpenAI: direct Azure OpenAI endpoint with deployment name
    elif provider_name == "azure_openai":
        if not p or not p.api_key or not p.api_base:
            console.print("[red]Error: Azure OpenAI requires api_key and api_base.[/red]")
            console.print("Set them in ~/.nanobot/config.json under providers.azure_openai section")
            console.print("Use the model field to specify the deployment name.")
            raise typer.Exit(1)
        provider = AzureOpenAIProvider(
            api_key=p.api_key,
            api_base=p.api_base,
            default_model=model,
        )
    else:
        from nanobot.providers.litellm_provider import LiteLLMProvider
        from nanobot.providers.registry import find_by_name
        spec = find_by_name(provider_name)
        if not model.startswith("bedrock/") and not (p and p.api_key) and not (spec and (spec.is_oauth or spec.is_local)):
            console.print("[red]Error: No API key configured.[/red]")
            console.print("Set one in ~/.nanobot/config.json under providers section")
            raise typer.Exit(1)
        provider = LiteLLMProvider(
            api_key=p.api_key if p else None,
            api_base=config.get_api_base(model),
            default_model=model,
            extra_headers=p.extra_headers if p else None,
            provider_name=provider_name,
        )

    defaults = config.agents.defaults
    provider.generation = GenerationSettings(
        temperature=defaults.temperature,
        max_tokens=defaults.max_tokens,
        reasoning_effort=defaults.reasoning_effort,
    )
    return provider


def _load_runtime_config(config: str | None = None, workspace: str | None = None) -> Config:
    """Load config and optionally override the active workspace."""
    from nanobot.config.loader import load_config, set_config_path

    config_path = None
    if config:
        config_path = Path(config).expanduser().resolve()
        if not config_path.exists():
            console.print(f"[red]Error: Config file not found: {config_path}[/red]")
            raise typer.Exit(1)
        set_config_path(config_path)
        console.print(f"[dim]Using config: {config_path}[/dim]")

    loaded = load_config(config_path)
    if workspace:
        loaded.agents.defaults.workspace = workspace
    return loaded


def _print_deprecated_memory_window_notice(config: Config) -> None:
    """Warn when running with old memoryWindow-only config."""
    if config.agents.defaults.should_warn_deprecated_memory_window:
        console.print(
            "[yellow]Hint:[/yellow] Detected deprecated `memoryWindow` without "
            "`contextWindowTokens`. `memoryWindow` is ignored; run "
            "[cyan]nanobot onboard[/cyan] to refresh your config template."
        )


# ============================================================================
# Gateway / Server
# ============================================================================


@app.command()
def gateway(
    port: int | None = typer.Option(None, "--port", "-p", help="Gateway port"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    config: str | None = typer.Option(None, "--config", "-c", help="Path to config file"),
):
    """Start the nanobot gateway."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.channels.manager import ChannelManager
    from nanobot.config.paths import get_cron_dir
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronJob
    from nanobot.heartbeat.service import HeartbeatService
    from nanobot.session.manager import SessionManager

    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    config = _load_runtime_config(config, workspace)
    _print_deprecated_memory_window_notice(config)
    port = port if port is not None else config.gateway.port

    console.print(f"{__logo__} Starting nanobot gateway version {__version__} on port {port}...")
    sync_workspace_templates(config.workspace_path)
    bus = MessageBus()
    provider = _make_provider(config)
    session_manager = SessionManager(config.workspace_path)

    # Create cron service first (callback set after agent creation)
    cron_store_path = get_cron_dir() / "jobs.json"
    cron = CronService(cron_store_path)

    # Create agent with cron service
    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        web_search_config=config.tools.web.search,
        web_proxy=config.tools.web.proxy or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=session_manager,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
        sjtu_config=config.sjtu,
        voice_config=config.voice,
        embedding_model=config.agents.defaults.embedding_model,
    )

    # Set cron callback (needs agent)
    async def on_cron_job(job: CronJob) -> str | None:
        """Execute a cron job through the agent."""
        from nanobot.agent.tools.cron import CronTool
        from nanobot.agent.tools.message import MessageTool
        from nanobot.utils.evaluator import evaluate_response

        reminder_note = (
            "[Scheduled Task] Timer finished.\n\n"
            f"Task '{job.name}' has been triggered.\n"
            f"Scheduled instruction: {job.payload.message}"
        )

        cron_tool = agent.tools.get("cron")
        cron_token = None
        if isinstance(cron_tool, CronTool):
            cron_token = cron_tool.set_cron_context(True)
        try:
            response = await agent.process_direct(
                reminder_note,
                session_key=f"cron:{job.id}",
                channel=job.payload.channel or "cli",
                chat_id=job.payload.to or "direct",
            )
        finally:
            if isinstance(cron_tool, CronTool) and cron_token is not None:
                cron_tool.reset_cron_context(cron_token)

        message_tool = agent.tools.get("message")
        if isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
            return response

        if job.payload.deliver and job.payload.to and response:
            should_notify = await evaluate_response(
                response, job.payload.message, provider, agent.model,
            )
            if should_notify:
                from nanobot.bus.events import OutboundMessage
                await bus.publish_outbound(OutboundMessage(
                    channel=job.payload.channel or "cli",
                    chat_id=job.payload.to,
                    content=response,
                ))
        return response
    cron.on_job = on_cron_job

    # Create channel manager
    channels = ChannelManager(config, bus)

    def _pick_heartbeat_target() -> tuple[str, str]:
        """Pick a routable channel/chat target for heartbeat-triggered messages."""
        enabled = set(channels.enabled_channels)
        # Prefer the most recently updated non-internal session on an enabled channel.
        for item in session_manager.list_sessions():
            key = item.get("key") or ""
            if ":" not in key:
                continue
            channel, chat_id = key.split(":", 1)
            if channel in {"cli", "system"}:
                continue
            if channel in enabled and chat_id:
                return channel, chat_id
        # Fallback keeps prior behavior but remains explicit.
        return "cli", "direct"

    # Create heartbeat service
    async def on_heartbeat_execute(tasks: str) -> str:
        """Phase 2: execute heartbeat tasks through the full agent loop."""
        channel, chat_id = _pick_heartbeat_target()

        async def _silent(*_args, **_kwargs):
            pass

        return await agent.process_direct(
            tasks,
            session_key="heartbeat",
            channel=channel,
            chat_id=chat_id,
            on_progress=_silent,
        )

    async def on_heartbeat_notify(response: str) -> None:
        """Deliver a heartbeat response to the user's channel."""
        from nanobot.bus.events import OutboundMessage
        channel, chat_id = _pick_heartbeat_target()
        if channel == "cli":
            return  # No external channel available to deliver to
        await bus.publish_outbound(OutboundMessage(channel=channel, chat_id=chat_id, content=response))

    hb_cfg = config.gateway.heartbeat
    heartbeat = HeartbeatService(
        workspace=config.workspace_path,
        provider=provider,
        model=agent.model,
        on_execute=on_heartbeat_execute,
        on_notify=on_heartbeat_notify,
        interval_s=hb_cfg.interval_s,
        enabled=hb_cfg.enabled,
    )

    if channels.enabled_channels:
        console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]Warning: No channels enabled[/yellow]")

    cron_status = cron.status()
    if cron_status["jobs"] > 0:
        console.print(f"[green]✓[/green] Cron: {cron_status['jobs']} scheduled jobs")

    console.print(f"[green]✓[/green] Heartbeat: every {hb_cfg.interval_s}s")

    async def run():
        try:
            await cron.start()
            await heartbeat.start()
            await asyncio.gather(
                agent.run(),
                channels.start_all(),
            )
        except KeyboardInterrupt:
            console.print("\nShutting down...")
        except Exception:
            import traceback
            console.print("\n[red]Error: Gateway crashed unexpectedly[/red]")
            console.print(traceback.format_exc())
        finally:
            await agent.close_mcp()
            heartbeat.stop()
            cron.stop()
            agent.stop()
            await channels.stop_all()

    asyncio.run(run())




# ============================================================================
# Agent Commands
# ============================================================================


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:direct", "--session", "-s", help="Session ID"),
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    config: str | None = typer.Option(None, "--config", "-c", help="Config file path"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Render assistant output as Markdown"),
    logs: bool = typer.Option(False, "--logs/--no-logs", help="Show nanobot runtime logs during chat"),
):
    """Interact with the agent directly."""
    from loguru import logger

    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.config.paths import get_cron_dir
    from nanobot.cron.service import CronService

    config = _load_runtime_config(config, workspace)
    _print_deprecated_memory_window_notice(config)
    sync_workspace_templates(config.workspace_path)

    bus = MessageBus()
    provider = _make_provider(config)

    # Create cron service for tool usage (no callback needed for CLI unless running)
    cron_store_path = get_cron_dir() / "jobs.json"
    cron = CronService(cron_store_path)

    if logs:
        logger.enable("nanobot")
    else:
        logger.disable("nanobot")

    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        web_search_config=config.tools.web.search,
        web_proxy=config.tools.web.proxy or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
        channels_config=config.channels,
        sjtu_config=config.sjtu,
        voice_config=config.voice,
        embedding_model=config.agents.defaults.embedding_model,
    )

    # Shared reference for progress callbacks
    _thinking: _ThinkingSpinner | None = None

    async def _cli_progress(content: str, *, tool_hint: bool = False) -> None:
        ch = agent_loop.channels_config
        if ch and tool_hint and not ch.send_tool_hints:
            return
        if ch and not tool_hint and not ch.send_progress:
            return
        if tool_hint:
            comment = _COMPANION.comment_on_tool(content)
            if comment:
                _print_companion_line(comment, _thinking)
        _print_cli_progress_line(content, _thinking)

    if message:
        # Single message mode — direct call, no bus needed
        async def run_once():
            nonlocal _thinking
            _thinking = _ThinkingSpinner(enabled=not logs)
            with _thinking:
                response = await agent_loop.process_direct(message, session_id, on_progress=_cli_progress)
            _thinking = None
            _print_agent_response(response, render_markdown=markdown)
            await agent_loop.close_mcp()

        asyncio.run(run_once())
    else:
        # Interactive mode — route through bus like other channels
        from nanobot.bus.events import InboundMessage
        _init_prompt_session()
        console.print(f"{__logo__} Interactive mode (type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit)\n")
        console.print(f"  [dim magenta]{_COMPANION.greet()}[/dim magenta]\n")

        if ":" in session_id:
            cli_channel, cli_chat_id = session_id.split(":", 1)
        else:
            cli_channel, cli_chat_id = "cli", session_id

        def _handle_signal(signum, frame):
            sig_name = signal.Signals(signum).name
            _restore_terminal()
            console.print(f"\nReceived {sig_name}, goodbye!")
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
        # SIGHUP is not available on Windows
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, _handle_signal)
        # Ignore SIGPIPE to prevent silent process termination when writing to closed pipes
        # SIGPIPE is not available on Windows
        if hasattr(signal, 'SIGPIPE'):
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)

        async def run_interactive():
            from nanobot.heartbeat.service import HeartbeatService

            settings = CompanionSettings.load()
            _COMPANION.reconfigure(settings)

            _animator = _CompanionAnimator()
            _animator.reconfigure(settings)
            anim_task = _animator.start()

            # Proactive heartbeat — fires tasks at idle intervals
            async def on_heartbeat_execute(tasks: str) -> str:
                return await agent_loop.process_direct(
                    tasks,
                    session_key="heartbeat",
                    channel=cli_channel,
                    chat_id=cli_chat_id,
                )

            async def on_heartbeat_notify(response: str) -> None:
                notice = f"\n[主动助手] {response}"
                await _print_interactive_response(notice, render_markdown=markdown)

            hb_cfg = config.gateway.heartbeat
            heartbeat = HeartbeatService(
                workspace=config.workspace_path,
                provider=provider,
                model=config.agents.defaults.model,
                on_execute=on_heartbeat_execute,
                on_notify=on_heartbeat_notify,
                interval_s=hb_cfg.interval_s,
                enabled=hb_cfg.enabled,
            )
            await heartbeat.start()

            bus_task = asyncio.create_task(agent_loop.run())
            turn_done = asyncio.Event()
            turn_done.set()
            turn_response: list[str] = []

            async def _consume_outbound():
                while True:
                    try:
                        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
                        if msg.metadata.get("_progress"):
                            is_tool_hint = msg.metadata.get("_tool_hint", False)
                            ch = agent_loop.channels_config
                            if ch and is_tool_hint and not ch.send_tool_hints:
                                pass
                            elif ch and not is_tool_hint and not ch.send_progress:
                                pass
                            else:
                                if is_tool_hint:
                                    raw = _COMPANION._raw_comment(msg.content)
                                    if raw:
                                        _animator.speak(raw, ticks=10)
                                        await _print_interactive_companion_line(
                                            f"{_COMPANION.CHAR} {_COMPANION.NAME}: {raw}", _thinking
                                        )
                                await _print_interactive_progress_line(msg.content, _thinking)

                        elif not turn_done.is_set():
                            if msg.content:
                                turn_response.append(msg.content)
                            turn_done.set()
                        elif msg.content:
                            await _print_interactive_response(msg.content, render_markdown=markdown)

                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break

            outbound_task = asyncio.create_task(_consume_outbound())

            try:
                while True:
                    try:
                        _flush_pending_tty_input()
                        user_input = await _read_interactive_input_async(
                            bottom_toolbar=_animator.get_toolbar
                        )
                        command = user_input.strip()
                        if not command:
                            continue

                        if command.startswith("/companion"):
                            _handle_companion_command(command, settings, _animator)
                            continue

                        if _is_exit_command(command):
                            _restore_terminal()
                            console.print(f"\n  [dim magenta]{_COMPANION.farewell()}[/dim magenta]")
                            console.print("\nGoodbye!")
                            break

                        turn_done.clear()
                        turn_response.clear()

                        await bus.publish_inbound(InboundMessage(
                            channel=cli_channel,
                            sender_id="user",
                            chat_id=cli_chat_id,
                            content=user_input,
                        ))

                        nonlocal _thinking
                        _animator.set_thinking(True)
                        _thinking = _ThinkingSpinner(enabled=not logs)
                        with _thinking:
                            await turn_done.wait()
                        _thinking = None
                        _animator.set_thinking(False)

                        if turn_response:
                            _print_agent_response(turn_response[0], render_markdown=markdown)
                    except KeyboardInterrupt:
                        _restore_terminal()
                        console.print(f"\n  [dim magenta]{_COMPANION.farewell()}[/dim magenta]")
                        console.print("\nGoodbye!")
                        break
                    except EOFError:
                        _restore_terminal()
                        console.print(f"\n  [dim magenta]{_COMPANION.farewell()}[/dim magenta]")
                        console.print("\nGoodbye!")
                        break
            finally:
                heartbeat.stop()
                _animator.stop()
                anim_task.cancel()
                agent_loop.stop()
                outbound_task.cancel()
                await asyncio.gather(bus_task, outbound_task, anim_task, return_exceptions=True)
                await agent_loop.close_mcp()

        asyncio.run(run_interactive())


# ============================================================================
# Channel Commands
# ============================================================================


channels_app = typer.Typer(help="Manage channels")
app.add_typer(channels_app, name="channels")


@channels_app.command("status")
def channels_status():
    """Show channel status."""
    from nanobot.channels.registry import discover_all
    from nanobot.config.loader import load_config

    config = load_config()

    table = Table(title="Channel Status")
    table.add_column("Channel", style="cyan")
    table.add_column("Enabled", style="green")

    for name, cls in sorted(discover_all().items()):
        section = getattr(config.channels, name, None)
        if section is None:
            enabled = False
        elif isinstance(section, dict):
            enabled = section.get("enabled", False)
        else:
            enabled = getattr(section, "enabled", False)
        table.add_row(
            cls.display_name,
            "[green]\u2713[/green]" if enabled else "[dim]\u2717[/dim]",
        )

    console.print(table)


def _get_bridge_dir() -> Path:
    """Get the bridge directory, setting it up if needed."""
    import shutil
    import subprocess

    # User's bridge location
    from nanobot.config.paths import get_bridge_install_dir

    user_bridge = get_bridge_install_dir()

    # Check if already built
    if (user_bridge / "dist" / "index.js").exists():
        return user_bridge

    # Check for npm
    npm_path = shutil.which("npm")
    if not npm_path:
        console.print("[red]npm not found. Please install Node.js >= 18.[/red]")
        raise typer.Exit(1)

    # Find source bridge: first check package data, then source dir
    pkg_bridge = Path(__file__).parent.parent / "bridge"  # nanobot/bridge (installed)
    src_bridge = Path(__file__).parent.parent.parent / "bridge"  # repo root/bridge (dev)

    source = None
    if (pkg_bridge / "package.json").exists():
        source = pkg_bridge
    elif (src_bridge / "package.json").exists():
        source = src_bridge

    if not source:
        console.print("[red]Bridge source not found.[/red]")
        console.print("Try reinstalling: pip install --force-reinstall nanobot")
        raise typer.Exit(1)

    console.print(f"{__logo__} Setting up bridge...")

    # Copy to user directory
    user_bridge.parent.mkdir(parents=True, exist_ok=True)
    if user_bridge.exists():
        shutil.rmtree(user_bridge)
    shutil.copytree(source, user_bridge, ignore=shutil.ignore_patterns("node_modules", "dist"))

    # Install and build
    try:
        console.print("  Installing dependencies...")
        subprocess.run([npm_path, "install"], cwd=user_bridge, check=True, capture_output=True)

        console.print("  Building...")
        subprocess.run([npm_path, "run", "build"], cwd=user_bridge, check=True, capture_output=True)

        console.print("[green]✓[/green] Bridge ready\n")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Build failed: {e}[/red]")
        if e.stderr:
            console.print(f"[dim]{e.stderr.decode()[:500]}[/dim]")
        raise typer.Exit(1)

    return user_bridge


@channels_app.command("login")
def channels_login():
    """Link device via QR code."""
    import shutil
    import subprocess

    from nanobot.config.loader import load_config
    from nanobot.config.paths import get_runtime_subdir

    config = load_config()
    bridge_dir = _get_bridge_dir()

    console.print(f"{__logo__} Starting bridge...")
    console.print("Scan the QR code to connect.\n")

    env = {**os.environ}
    wa_cfg = getattr(config.channels, "whatsapp", None) or {}
    bridge_token = wa_cfg.get("bridgeToken", "") if isinstance(wa_cfg, dict) else getattr(wa_cfg, "bridge_token", "")
    if bridge_token:
        env["BRIDGE_TOKEN"] = bridge_token
    env["AUTH_DIR"] = str(get_runtime_subdir("whatsapp-auth"))

    npm_path = shutil.which("npm")
    if not npm_path:
        console.print("[red]npm not found. Please install Node.js.[/red]")
        raise typer.Exit(1)

    try:
        subprocess.run([npm_path, "start"], cwd=bridge_dir, check=True, env=env)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Bridge failed: {e}[/red]")


# ============================================================================
# Plugin Commands
# ============================================================================

plugins_app = typer.Typer(help="Manage channel plugins")
app.add_typer(plugins_app, name="plugins")


@plugins_app.command("list")
def plugins_list():
    """List all discovered channels (built-in and plugins)."""
    from nanobot.channels.registry import discover_all, discover_channel_names
    from nanobot.config.loader import load_config

    config = load_config()
    builtin_names = set(discover_channel_names())
    all_channels = discover_all()

    table = Table(title="Channel Plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Source", style="magenta")
    table.add_column("Enabled", style="green")

    for name in sorted(all_channels):
        cls = all_channels[name]
        source = "builtin" if name in builtin_names else "plugin"
        section = getattr(config.channels, name, None)
        if section is None:
            enabled = False
        elif isinstance(section, dict):
            enabled = section.get("enabled", False)
        else:
            enabled = getattr(section, "enabled", False)
        table.add_row(
            cls.display_name,
            source,
            "[green]yes[/green]" if enabled else "[dim]no[/dim]",
        )

    console.print(table)


# ============================================================================
# Status Commands
# ============================================================================


@app.command()
def status():
    """Show nanobot status."""
    from nanobot.config.loader import get_config_path, load_config

    config_path = get_config_path()
    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} nanobot Status\n")

    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")

    if config_path.exists():
        from nanobot.providers.registry import PROVIDERS

        console.print(f"Model: {config.agents.defaults.model}")

        # Check API keys from registry
        for spec in PROVIDERS:
            p = getattr(config.providers, spec.name, None)
            if p is None:
                continue
            if spec.is_oauth:
                console.print(f"{spec.label}: [green]✓ (OAuth)[/green]")
            elif spec.is_local:
                # Local deployments show api_base instead of api_key
                if p.api_base:
                    console.print(f"{spec.label}: [green]✓ {p.api_base}[/green]")
                else:
                    console.print(f"{spec.label}: [dim]not set[/dim]")
            else:
                has_key = bool(p.api_key)
                console.print(f"{spec.label}: {'[green]✓[/green]' if has_key else '[dim]not set[/dim]'}")


# ============================================================================
# OAuth Login
# ============================================================================

provider_app = typer.Typer(help="Manage providers")
app.add_typer(provider_app, name="provider")


_LOGIN_HANDLERS: dict[str, callable] = {}


def _register_login(name: str):
    def decorator(fn):
        _LOGIN_HANDLERS[name] = fn
        return fn
    return decorator


@provider_app.command("login")
def provider_login(
    provider: str = typer.Argument(..., help="OAuth provider (e.g. 'openai-codex', 'github-copilot')"),
):
    """Authenticate with an OAuth provider."""
    from nanobot.providers.registry import PROVIDERS

    key = provider.replace("-", "_")
    spec = next((s for s in PROVIDERS if s.name == key and s.is_oauth), None)
    if not spec:
        names = ", ".join(s.name.replace("_", "-") for s in PROVIDERS if s.is_oauth)
        console.print(f"[red]Unknown OAuth provider: {provider}[/red]  Supported: {names}")
        raise typer.Exit(1)

    handler = _LOGIN_HANDLERS.get(spec.name)
    if not handler:
        console.print(f"[red]Login not implemented for {spec.label}[/red]")
        raise typer.Exit(1)

    console.print(f"{__logo__} OAuth Login - {spec.label}\n")
    handler()


@_register_login("openai_codex")
def _login_openai_codex() -> None:
    try:
        from oauth_cli_kit import get_token, login_oauth_interactive
        token = None
        try:
            token = get_token()
        except Exception:
            pass
        if not (token and token.access):
            console.print("[cyan]Starting interactive OAuth login...[/cyan]\n")
            token = login_oauth_interactive(
                print_fn=lambda s: console.print(s),
                prompt_fn=lambda s: typer.prompt(s),
            )
        if not (token and token.access):
            console.print("[red]✗ Authentication failed[/red]")
            raise typer.Exit(1)
        console.print(f"[green]✓ Authenticated with OpenAI Codex[/green]  [dim]{token.account_id}[/dim]")
    except ImportError:
        console.print("[red]oauth_cli_kit not installed. Run: pip install oauth-cli-kit[/red]")
        raise typer.Exit(1)


@_register_login("github_copilot")
def _login_github_copilot() -> None:
    import asyncio

    console.print("[cyan]Starting GitHub Copilot device flow...[/cyan]\n")

    async def _trigger():
        from litellm import acompletion
        await acompletion(model="github_copilot/gpt-4o", messages=[{"role": "user", "content": "hi"}], max_tokens=1)

    try:
        asyncio.run(_trigger())
        console.print("[green]✓ Authenticated with GitHub Copilot[/green]")
    except Exception as e:
        console.print(f"[red]Authentication error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
