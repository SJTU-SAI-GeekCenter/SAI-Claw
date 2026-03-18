"""PDF Summarization Tool for Exam Review - 期末考试复习助手"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.filesystem import _resolve_path


class SummaryPDFFileTool(Tool):
    """
    期末考试复习助手 - 智能总结PDF讲义/PPT
    
    读取PDF并按照学习科学原理进行结构化总结，帮助学生高效复习。
    """

    @property
    def name(self) -> str:
        return "summary_pdf_file"

    @property
    def description(self) -> str:
        return (
            "【期末考试复习助手】深度总结PDF讲义/PPT，生成结构化复习材料。\n"
            "功能：1)核心要点提取 2)重点标记 3)易混淆概念辨析 4)模拟题生成 "
            "5)知识图谱 6)记忆技巧 7)时间规划建议\n"
            "适用：课程讲义、PPT、lecture notes等复习资料"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "PDF文件的绝对路径或相对路径",
                },
                "exam_date": {
                    "type": "string",
                    "description": "考试日期（可选，用于制定复习计划），格式：YYYY-MM-DD",
                },
                "focus_areas": {
                    "type": "string",
                    "description": "重点关注的主题（可选，如'第三章的微积分'）",
                },
            },
            "required": ["file_path"],
        }

    def __init__(
        self,
        workspace: Path | None = None,
        allowed_dir: Path | None = None,
        extra_allowed_dirs: list[Path] | None = None,
    ):
        self._workspace = workspace
        self._allowed_dir = allowed_dir
        self._extra_allowed_dirs = extra_allowed_dirs

    async def execute(
        self,
        file_path: str,
        exam_date: str | None = None,
        focus_areas: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        执行PDF总结 - 返回一个结构化总结指令，由LLM处理
        """
        try:
            import pdfplumber
        except ImportError:
            return (
                "Error: pdfplumber not installed. "
                "Install with: pip install pdfplumber"
            )

        try:
            fp = _resolve_path(
                file_path, self._workspace, self._allowed_dir, self._extra_allowed_dirs
            )

            if not fp.exists():
                return f"❌ 错误：文件不存在: {file_path}"
            if not fp.is_file():
                return f"❌ 错误：这不是一个文件: {file_path}"
            if not file_path.lower().endswith(".pdf"):
                return f"❌ 错误：请提供PDF文件: {file_path}"

            # 提取PDF内容
            full_text = []
            with pdfplumber.open(fp) as pdf:
                total_pages = len(pdf.pages)
                
                for i, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    if text.strip():
                        full_text.append(f"=== 第{i}页 ===\n{text.strip()}")

            content = "\n\n".join(full_text)
            
            # 内容过长时截断提示
            content_display = content
            if len(content) > 50000:
                content_display = content[:50000] + "\n\n... (内容过长已截断，但总结会基于全部内容)"

            # 构建考试复习助手的系统提示
            exam_context = f"""
📚 **文件信息**
- 文件名：{fp.name}
- 总页数：{total_pages}页
- 考试日期：{exam_date if exam_date else '未指定'}
- 重点关注：{focus_areas if focus_areas else '全部内容'}

---

📖 **PDF完整内容如下**

{content_display}

---

🎯 **请按照以下结构生成期末考试复习材料**

## 1️⃣ 核心要点速览 (Key Takeaways)
用 bullet points 列出本章/本讲的 5-10 个最核心的概念和结论
- 每个要点一句话概括
- 标记 ⭐ 表示高频考点

## 2️⃣ 重点复习区域 (Priority Review)
按重要性排序的内容模块：
- 🔴 必考点（几乎必考）
- 🟡 高频点（经常考到）  
- 🟢 了解点（可能考到）

## 3️⃣ 易混淆概念辨析 (Confusion Alert)
列出本章节容易混淆的概念对，用对比表格或清单形式：
- 概念A vs 概念B：区别是什么？
- 常见错误：学生常把什么搞混？
- 记忆口诀：如何区分？

## 4️⃣ 公式与定理速记 (Formula Sheet)
提取所有重要公式和定理：
- 公式名称
- 数学表达式
- 适用条件/假设
- 简单记忆技巧

## 5️⃣ 模拟练习题 (Practice Questions)
根据内容出 5-8 道模拟题：
- 选择题 2-3 道（覆盖基础概念）
- 简答题/计算题 2-3 道（覆盖核心方法）
- 综合应用题 1-2 道（覆盖多个知识点）

每道题后标注：
- 考点对应（对应上面的第几部分）
- 难度评级（⭐/⭐⭐/⭐⭐⭐）
- 参考答案要点（简要）

## 6️⃣ 知识图谱 (Knowledge Map)
用层级结构或思维导图形式展示：
- 核心概念位于中心
- 分支列出相关概念和公式
- 标注概念间的关联

## 7️⃣ 记忆技巧与口诀 (Memory Hacks)
- 首字母缩写法
- 联想记忆法
- 编个口诀或故事
- 图示化建议

## 8️⃣ 常见考试陷阱 (Exam Traps)
- 审题易错点
- 计算常见错误
- 边界条件注意
- 格式/单位易错点

## 9️⃣ 复习时间规划建议 (Study Plan)
{'' if not exam_date else f'距离考试还有一段时间，建议按以下节奏复习：\n- 第1遍：通读 + 理解（建议时长）\n- 第2遍：重点记忆 + 做题（建议时长）\n- 第3遍：查漏补缺 + 模拟（建议时长）'}

## 🔟 延伸阅读与资源 (Extra Resources)
- 如果还有不懂，应该去看哪本书/哪个视频？
- 推荐的相关练习题来源

---

⚠️ **生成要求**：
1. 基于PDF内容，不要编造未提及的知识点
2. 语言风格适合学生复习，简洁明了
3. 多用 emoji、表格、列表增加可读性
4. 重点突出，层次分明
"""

            return exam_context

        except PermissionError as e:
            return f"❌ 错误：无权限访问 - {e}"
        except Exception as e:
            logger.error("Failed to summarize PDF {}: {}", file_path, e)
            return f"❌ 读取PDF失败: {e}"


class SummaryFileCommandTool(Tool):
    """
    处理 /summaryfile 命令的工具
    解析用户输入的命令格式：/summaryfile /path/to/file.pdf
    """

    @property
    def name(self) -> str:
        return "summaryfile_command"

    @property
    def description(self) -> str:
        return "Internal tool to handle /summaryfile command"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
            },
            "required": ["command"],
        }

    async def execute(self, command: str, **kwargs: Any) -> str:
        """解析 /summaryfile 命令"""
        # 解析命令格式: /summaryfile /path/to/file.pdf --exam-date 2024-01-15
        parts = command.split()
        
        if len(parts) < 2:
            return (
                "❌ 用法错误\n"
                "正确格式：\n"
                "  /summaryfile /path/to/file.pdf\n"
                "  /summaryfile ~/Downloads/lecture.pdf --exam-date 2024-01-20\n"
                "  /summaryfile ./notes.pdf --focus '第三章积分'"
            )
        
        file_path = parts[1]
        exam_date = None
        focus_areas = None
        
        # 解析可选参数
        for i, part in enumerate(parts[2:], 2):
            if part == "--exam-date" and i + 1 < len(parts):
                exam_date = parts[i + 1]
            elif part == "--focus" and i + 1 < len(parts):
                focus_areas = parts[i + 1]
        
        # 调用总结工具（通过返回特殊标记，让agent loop处理）
        return f"[SUMMARY_FILE_REQUEST] path={file_path} date={exam_date} focus={focus_areas}"
