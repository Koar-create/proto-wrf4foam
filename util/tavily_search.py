#!/usr/bin/env python3
"""Tavily Search API 命令行封装（不依赖 MCP）。

密钥只从以下来源读取，脚本内不保存、不打印 API key：
  1. 环境变量 TAVILY_API_KEY
  2. 仓库根目录 local.env（已在 .gitignore）
  3. $HOME/.cache/wrf-of-coupling/tavily.env（建议 chmod 600）

示例:
  export TAVILY_API_KEY='tvly-...'
  python3 util/tavily_search.py "WRF urban canopy model" --max-results 3
  python3 util/tavily_search.py "..." --format text
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.tavily.com/search"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_ENV = REPO_ROOT / "local.env"
USER_ENV = Path.home() / ".cache" / "wrf-of-coupling" / "tavily.env"


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key:
            out[key] = val
    return out


def load_api_key() -> str:
    key = os.environ.get("TAVILY_API_KEY", "").strip()
    if key:
        return key
    for path in (DEFAULT_LOCAL_ENV, USER_ENV):
        key = _parse_env_file(path).get("TAVILY_API_KEY", "").strip()
        if key:
            return key
    return ""


def _key_help() -> str:
    return (
        "未找到 Tavily API key。请任选其一：\n"
        "  export TAVILY_API_KEY='tvly-...'\n"
        f"  在 {DEFAULT_LOCAL_ENV} 或 {USER_ENV} 中设置 TAVILY_API_KEY=...\n"
        f"  （建议: mkdir -p {USER_ENV.parent} && chmod 700 {USER_ENV.parent})"
    )


def tavily_search(
    api_key: str,
    query: str,
    max_results: int,
    search_depth: str,
    include_answer: bool,
) -> dict:
    body = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
    }
    if include_answer:
        body["include_answer"] = True

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Tavily HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Tavily 网络错误: {e.reason}") from e


def format_text(payload: dict) -> str:
    lines: list[str] = []
    if payload.get("answer"):
        lines.append("=== answer ===")
        lines.append(str(payload["answer"]))
        lines.append("")
    lines.append(f"query: {payload.get('query', '')}")
    lines.append(f"response_time: {payload.get('response_time')}s  request_id: {payload.get('request_id')}")
    lines.append("")
    for i, hit in enumerate(payload.get("results") or [], start=1):
        lines.append(f"[{i}] {hit.get('title', '(no title)')}")
        lines.append(f"    {hit.get('url', '')}")
        score = hit.get("score")
        if score is not None:
            lines.append(f"    score: {score}")
        content = (hit.get("content") or "").replace("\n", " ")
        if len(content) > 500:
            content = content[:497] + "..."
        if content:
            lines.append(f"    {content}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Tavily Search API（密钥仅来自环境或 local.env）")
    parser.add_argument("query", help="搜索查询")
    parser.add_argument("--max-results", type=int, default=5, metavar="N")
    parser.add_argument(
        "--search-depth",
        choices=("basic", "advanced"),
        default="basic",
    )
    parser.add_argument("--include-answer", action="store_true")
    parser.add_argument(
        "--format",
        choices=("json", "text"),
        default="json",
        help="输出格式（默认 json）",
    )
    args = parser.parse_args()

    api_key = load_api_key()
    if not api_key:
        print(_key_help(), file=sys.stderr)
        return 3

    if args.max_results < 1 or args.max_results > 20:
        print("错误: --max-results 须在 1–20 之间", file=sys.stderr)
        return 2

    try:
        payload = tavily_search(
            api_key,
            args.query,
            args.max_results,
            args.search_depth,
            args.include_answer,
        )
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 4

    if args.format == "text":
        sys.stdout.write(format_text(payload))
    else:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
