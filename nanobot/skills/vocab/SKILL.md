---
name: vocab
description: Vocabulary learning and management. Store, query, and practice English vocabulary with LLM-generated examples, etymology, and contextual paragraphs.
version: "1.0"
author: SAI-Claw
tags: [education, vocabulary, learning, english]
---

# Vocabulary Learning Skill

## Overview

This skill helps you learn and manage English vocabulary. You can store words, query their meanings with examples, get etymology information, generate contextual paragraphs, and track your learning progress.

## Available Actions

### Store Words

Store vocabulary words for later review.

**User phrases indicating this action:**
- "记住单词 resilience"
- "添加单词 apple, banana, cherry"
- "保存这些单词"
- "store the word: serendipity"

**What to do:**
1. Extract the words from the user's message
2. Call the `store_vocabulary` tool with the words
3. Report the result (stored count, duplicates)

### Query Word Details

Get detailed information about a specific word including meaning, phonetic, examples, etymology, synonyms, and antonyms.

**User phrases indicating this action:**
- "resilience 是什么意思"
- "查单词 resilience"
- "explain the word: ephemeral"
- "what does serendipity mean"

**What to do:**
1. Extract the target word
2. Call the `query_word` tool
3. Present the formatted result to the user

### Generate Contextual Paragraph

Create a natural English paragraph using specified vocabulary words.

**User phrases indicating this action:**
- "用 resilience 和 community 写一段话"
- "generate a paragraph with: serendipity, joy"
- "写个短文包含这些单词"

**What to do:**
1. Extract the target words
2. Call the `generate_paragraph` tool
3. Present the paragraph with translation

### Review Vocabulary

Review previously learned words.

**User phrases indicating this action:**
- "复习单词"
- "review my vocabulary"
- "show me words to review"

**What to do:**
1. Call the `get_review_queue` tool
2. Present the review words to the user

### Get Learning Statistics

View vocabulary learning progress.

**User phrases indicating this action:**
- "我学了多少单词"
- "show my vocabulary stats"
- "学习统计"

**What to do:**
1. Call the `get_learning_stats` tool
2. Present the statistics to the user

## Guidelines

- Always extract words accurately from user messages
- For Chinese-English mixed input, focus on the English words
- If a word is not found in the database, offer to add it with user-provided meaning
- Present information in a clear, organized format with appropriate emoji
- Be encouraging about the user's learning progress

## Data Privacy

Vocabulary data is stored per-user and never shared between users.
