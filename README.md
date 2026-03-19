# SAI-Claw

## Introduction

本仓库 fork 自 [NanoBot](https://github.com/HKUDS/nanobot)，一个由 Python 编写的超轻量 OpenClaw。SAI-Claw 将在此基础上进行二次开发。

## Usage

```bash
uv tool install nanobot-ai
```

然后编辑 `~/.nanobot/config.json` 配置个性化信息。

## feature

### /summaryfile - 期末考试复习助手 帮你快速总结PDF讲义，生成结构化复习材料                                                                

`/summaryfile /path/to/file.pdf /summaryfile ~/Downloads/lecture.pdf --exam-date 2024-01-20 /summaryfile ./notes.pdf --focus '第三章积分’`                                                                                                        

### /xuanke 选课社区
首先 `/config /xuanke`来输入邮箱前缀（不包含@sjtu.edu.cn）和选课社区密码。然后使用 `/xuanke 老师名`,`/xuanke 课程名` 来搜索。