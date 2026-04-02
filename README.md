# SAI-Claw

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

## feature

### /summaryfile - 期末考试复习助手 帮你快速总结PDF讲义，生成结构化复习材料                                                                

`/summaryfile /path/to/file.pdf /summaryfile ~/Downloads/lecture.pdf --exam-date 2024-01-20 /summaryfile ./notes.pdf --focus '第三章积分’`                                                                                                        

### /xuanke 选课社区

首先 `/config /xuanke`来输入邮箱前缀（不包含@sjtu.edu.cn）和选课社区密码。然后使用 `/xuanke 老师名`,`/xuanke 课程名` 来搜索。

### /canvas Canvas 功能

首先 `/config /canvas`，然后按照教程打开已经登陆的 canvas，拿到 `_normandy_session` 码复制给 bot，然后使用 `/canvas 你的需求` 即可使用！
例如：`/canvas 查看我一共有哪些课程`，`/canvas 帮我下载大学物理三的最新一次 ppt`