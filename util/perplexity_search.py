#!/usr/bin/env python3
"""Perplexity Agent API 命令行封装（不依赖 MCP）。

密钥只从以下来源读取，脚本内不保存、不打印 API key：
  1. 环境变量 PERPLEXITY_API_KEY
  2. 仓库根目录 local.env（已在 .gitignore）
  3. $HOME/.cache/wrf-of-coupling/perplexity.env（建议 chmod 600）

示例:
  python3 util/perplexity_search.py "WRF urban canopy model validation"
  python3 util/perplexity_search.py "..." --preset medium --format text
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.perplexity.ai/v1/agent"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_ENV = REPO_ROOT / "local.env"
USER_ENV = Path.home() / ".cache" / "wrf-of-coupling" / "perplexity.env"
KEY_NAME = "PERPLEXITY_API_KEY"
PLACEHOLDERS = frozenset(
    {
        "",
        "YOUR_PERPLEXITY_API_KEY_HERE",
        "REPLACE_ME",
        "CHANGEME",
        "PLACEHOLDER",
    }
)
PRESETS = ("fast", "low", "medium", "high", "xhigh")
DEFAULT_TIMEOUT = {
    "fast": 120,
    "low": 180,
    "medium": 600,
    "high": 900,
    "xhigh": 1200,
}


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
    key = os.environ.get(KEY_NAME, "").strip()
    if key and key not in PLACEHOLDERS:
        return key
    for path in (DEFAULT_LOCAL_ENV, USER_ENV):
        key = _parse_env_file(path).get(KEY_NAME, "").strip()
        if key and key not in PLACEHOLDERS:
            return key
    return ""


def _key_help() -> str:
    return (
        f"未找到有效的 {KEY_NAME}（占位符不算）。请任选其一：\n"
        f"  把 {DEFAULT_LOCAL_ENV} 或 {USER_ENV} 里的占位符换成真实 key\n"
        f"  export {KEY_NAME}='pplx-...'\n"
        f"  （建议: mkdir -p {USER_ENV.parent} && chmod 700 {USER_ENV.parent}）"
    )


def extract_output_text(payload: dict) -> str:
    text = payload.get("output_text")
    if isinstance(text, str) and text.strip():
        return text
    chunks: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, str) and part:
                chunks.append(part)
            elif isinstance(part, dict) and part.get("type") == "output_text":
                value = part.get("text") or ""
                if value:
                    chunks.append(str(value))
    return "\n".join(chunks)


def collect_search_results(payload: dict) -> list[dict]:
    seen: set[str] = set()
    results: list[dict] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "search_results":
            continue
        for hit in item.get("results") or []:
            if not isinstance(hit, dict):
                continue
            ident = hit.get("id")
            url = hit.get("url") or ""
            key = f"{ident}:{url}" if ident is not None else url
            if not key or key in seen:
                continue
            seen.add(key)
            results.append(hit)
    return results


def perplexity_search(
    api_key: str,
    query: str,
    preset: str,
    timeout: int,
    instructions: str,
    domains: list[str],
) -> dict:
    body: dict = {
        "preset": preset,
        "input": query,
    }
    if instructions:
        body["instructions"] = instructions
    if domains:
        body["tools"] = [
            {
                "type": "web_search",
                "filters": {"search_domain_filter": domains},
            }
        ]

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Perplexity HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Perplexity 网络错误: {e.reason}") from e


def format_text(payload: dict) -> str:
    lines: list[str] = []
    status = payload.get("status", "")
    model = payload.get("model", "")
    usage = payload.get("usage") or {}
    lines.append(f"status: {status}  model: {model}  id: {payload.get('id', '')}")
    if usage:
        cost = (usage.get("cost") or {}).get("total_cost")
        lines.append(
            "tokens: "
            f"in={usage.get('input_tokens')}  out={usage.get('output_tokens')}  "
            f"total={usage.get('total_tokens')}  cost_usd={cost}"
        )
    lines.append("")
    answer = extract_output_text(payload)
    if answer:
        lines.append("=== answer ===")
        lines.append(answer.rstrip())
        lines.append("")
    hits = collect_search_results(payload)
    if hits:
        lines.append("=== sources ===")
        for hit in hits:
            ident = hit.get("id")
            prefix = f"[{ident}] " if ident is not None else ""
            lines.append(f"{prefix}{hit.get('title', '(no title)')}")
            lines.append(f"    {hit.get('url', '')}")
            snippet = (hit.get("snippet") or "").replace("\n", " ")
            if len(snippet) > 500:
                snippet = snippet[:497] + "..."
            if snippet:
                lines.append(f"    {snippet}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Perplexity Agent API（密钥仅来自环境或 local.env）"
    )
    parser.add_argument("query", help="检索 / 研究问题")
    parser.add_argument(
        "--preset",
        choices=PRESETS,
        default="low",
        help="fast=速查, low=日常研究(默认), medium=深度, high/xhigh=更重",
    )
    parser.add_argument("--instructions", default="", help="可选系统指令")
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        metavar="HOST",
        help="限制搜索域名，可重复，如 --domain arxiv.org",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        metavar="SEC",
        help="HTTP 超时秒数（默认随 preset 变化）",
    )
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

    timeout = args.timeout if args.timeout > 0 else DEFAULT_TIMEOUT[args.preset]
    try:
        payload = perplexity_search(
            api_key,
            args.query,
            args.preset,
            timeout,
            args.instructions,
            args.domain,
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
