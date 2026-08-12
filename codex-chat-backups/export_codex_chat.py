#!/usr/bin/env python3
"""把 Codex 本地会话 JSONL 导出为可读的 Markdown 备份。

用法:
    python3 export_codex_chat.py <session.jsonl> [-o output.md]

不带 -o 时，输出到当前目录下 <session-id>.md。
会话文件默认在 ~/.codex/sessions/ 和 ~/.codex/archived_sessions/ 下，
每个文件对应一次会话（rollout-*.jsonl）。
"""

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


ENV_CONTEXT_RE = re.compile(
    r"<environment_context>.*?</environment_context>", re.DOTALL
)


def extract_text(content) -> str:
    """从 response_item 的 content 数组中取出可见文本。"""
    parts = []
    if not isinstance(content, list):
        return ""
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") in ("input_text", "output_text", "text"):
            parts.append(block.get("text", ""))
    return "\n".join(parts)


def clean_user_text(text: str) -> str:
    """去掉 Codex 自动注入的环境上下文，只保留用户真正输入的内容。"""
    cleaned = ENV_CONTEXT_RE.sub("", text)
    return cleaned.strip()


def fmt_ts(iso: str) -> str:
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    except (ValueError, TypeError):
        return iso or ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Codex 会话 JSONL -> Markdown")
    parser.add_argument("jsonl", help="rollout-*.jsonl 会话文件")
    parser.add_argument("-o", "--output", help="输出 Markdown 路径")
    args = parser.parse_args()

    src = Path(args.jsonl)
    if not src.is_file():
        print(f"找不到会话文件: {src}", file=sys.stderr)
        return 1

    meta = {}
    messages = []  # (timestamp, role, text)
    ts_by_id = {}

    with src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            ts = event.get("timestamp", "")
            etype = event.get("type")
            payload = event.get("payload", {})
            if etype == "session_meta":
                meta = payload
                continue
            if etype == "response_item" and payload.get("type") == "message":
                ts_by_id[payload.get("id")] = ts

    # 按时间顺序收集消息
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event.get("type") != "response_item":
                continue
            payload = event.get("payload", {})
            if payload.get("type") != "message" or payload.get("role") not in (
                "user",
                "assistant",
            ):
                continue
            role = payload["role"]
            text = extract_text(payload.get("content"))
            if role == "user":
                text = clean_user_text(text)
            if not text:
                continue
            ts = ts_by_id.get(payload.get("id"), event.get("timestamp", ""))
            messages.append((ts, role, text, payload.get("phase", "")))

    messages.sort(key=lambda m: m[0])

    session_id = meta.get("session_id") or src.stem
    out = Path(args.output) if args.output else Path.cwd() / f"{session_id}.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"# Codex 对话备份")
    lines.append("")
    lines.append(f"- 会话 ID：`{session_id}`")
    lines.append(f"- 开始时间：{fmt_ts(meta.get('timestamp', ''))}")
    lines.append(f"- 工作目录：`{meta.get('cwd', '')}`")
    lines.append(f"- 来源：{meta.get('originator', '')} / {meta.get('cli_version', '')}")
    lines.append(f"- 模型提供方：{meta.get('model_provider', '')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for ts, role, text, phase in messages:
        ts_s = fmt_ts(ts)
        if role == "user":
            lines.append(f"## 👤 用户 {('· ' + ts_s) if ts_s else ''}")
        else:
            tag = "（过程更新）" if phase == "commentary" else ""
            lines.append(f"## 🤖 Codex {tag}{(' · ' + ts_s) if ts_s else ''}")
        lines.append("")
        lines.append(text)
        lines.append("")
        lines.append("---")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"已导出 {len(messages)} 条消息 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
