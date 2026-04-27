# SAI-Claw
<img width="579.2" height="434.4" alt="9b88e66cc628856026a250320a425f78" src="https://github.com/user-attachments/assets/6b81e838-dd3b-452e-b511-d27a3c1d2215" />

## Introduction

本仓库 fork 自 [NanoBot](https://github.com/HKUDS/nanobot)，一个由 Python 编写的超轻量 OpenClaw。SAI-Claw 将在此基础上进行二次开发。

## Usage

```bash
uv tool install nanobot-ai
```

然后编辑 `~/.nanobot/config.json` 配置个性化信息。

## 开发说明

### 本地修改后重新安装

`nanobot` 命令使用 uv 独立管理的 venv（`nanobot-ai`），**不会**自动读取本地源码改动。每次修改代码后需要重新安装才能生效：

```bash
uv tool install . --reinstall
```

> **原因**：`uv tool install nanobot-ai` 从 PyPI 安装的是发布版，与本地源码是两份独立的代码。直接用 `python` 运行时因为当前目录在 `sys.path` 里所以会读到本地版，但 `nanobot` 命令不会。重装后本地源码会替换 venv 里的包。

## Companion 宠物

进入交互模式（`nanobot agent`，不加 `-m`）后，输入框正下方会出现一个持续动画的小伙伴 Murmur：

```
  (o o)· Murmur        ← 空闲，粒子浮动 + 偶尔眨眼/微笑
  (O_O)⠙ Murmur        ← Agent 思考中，spinner 旋转
  (^ ^)  Murmur: 搜索引擎，启动！   ← 工具调用时冒出气泡
```

**自动行为：** 启动打招呼、工具调用时随机评论（35% 概率）、长时间闲置自动说闲话、退出道别。

### 自定义命令

```bash
/companion                  # 查看当前设置
/companion name <名字>      # 改名，任意文字
/companion mood <心情>      # 改心情
/companion face <外形>      # 改外形
/companion reset            # 恢复默认
```

**Moods:**

| Mood | Style | Frequency |
|------|-------|-----------|
| 活泼 | Default, upbeat | 35% |
| 安静 | Nearly silent, only「...」 | 10% |
| 中二 | "The wheel of fate turns..." dramatic mode | 40% |
| 毒舌 | Sarcastic — "fine, I'll look it up" | 30% |

**Faces:**

| Face | Idle | Thinking | Speaking |
|------|------|----------|----------|
| ghost | `(o o)` | `(O_O)` | `(^ ^)` |
| cat | `(=.=)` | `(O.O)` | `(^.^)` |
| robot | `[o_o]` | `[O_O]` | `[^_^]` |
| uwu | `(owo)` | `(OwO)` | `(^w^)` |

设置自动保存到 `~/.nanobot/companion.json`，下次启动自动恢复。

## feature

### /summaryfile - 期末考试复习助手 帮你快速总结PDF讲义，生成结构化复习材料                                                                

`/summaryfile /path/to/file.pdf /summaryfile ~/Downloads/lecture.pdf --exam-date 2024-01-20 /summaryfile ./notes.pdf --focus '第三章积分’`                                                                                                        

### /xuanke 选课社区

首先 `/config /xuanke`来输入邮箱前缀（不包含@sjtu.edu.cn）和选课社区密码。然后使用 `/xuanke 老师名`,`/xuanke 课程名` 来搜索。

### /canvas Canvas 功能

首先 `/config /canvas`，然后按照教程打开已经登陆的 canvas，拿到 `_normandy_session` 码复制给 bot，然后使用 `/canvas 你的需求` 即可使用！
例如：`/canvas 查看我一共有哪些课程`，`/canvas 帮我下载大学物理三的最新一次 ppt`

### /vocab - 词汇学习

智能英语词汇学习系统，支持单词存储、查询和语境生成。

**命令：**
- `/vocab <words>` - 存储单词到词库，例如：`/vocab apple resilience serendipity`
- `/word <word>` - 查询单词详情（释义、音标、例句、词源、同反义词），例如：`/word resilience`
- `/paragraph <words>` - 生成包含指定单词的语境短文（带中文翻译），例如：`/paragraph resilience community`
- `/review` - 复习已学单词
- `/stats` - 查看学习统计（总单词数、今日学习数）

**特性：**
- 硬拦截命令，0 延迟响应
- SQLite 本地存储，用户隔离
- LLM 生成例句、词源、语境短文
- 支持自然语言查询（如"我昨天背了什么单词"）

**配置：** 在 `~/.nanobot/config.json` 中启用：
```json
{
  "channels": {
    "vocabulary": {
      "enabled": true,
      "db_path": "~/.nanobot/data/vocabulary.db"
    }
  }
}
```

### /profile 个人画像

输入 `/profile` 查看助手对你的长期记忆摘要——包括你的偏好、习惯、常用工具、历史话题等。

画像由每次对话后的自动记忆整合构建，越聊越准。

### 主动助手（Proactive Heartbeat）

在工作区创建 `HEARTBEAT.md`，写入你希望助手定期自动执行的任务：

```markdown
- 检查今天的 Canvas 作业截止情况
- 提醒我复习昨天学过的知识点
- 如果工作区有新 PDF 就自动摘要
```

每隔约 30 分钟，助手会静默检查并执行这些任务，有结果时主动告诉你。

输入 `/proactive` 查看当前任务列表。
