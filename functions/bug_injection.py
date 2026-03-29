"""
bug_injector_node.py  –  ErrorX Bug Injection Agent
=====================================================
LangGraph node that reads a BugReport from the ErrorX state,
iterates over each BugSurface file, injects exactly one subtle bug
per file, threads cross-file context for coherence, and writes a
running context log to  <repo_path>/context/abcd.txt.

Usage (inside your LangGraph graph):
    builder.add_node("inject_bugs", bug_injector_node)
"""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models  (shared with analysis agent)
# ─────────────────────────────────────────────────────────────────────────────

class BugSurface(BaseModel):
    file: str = Field(...)
    entry_points: List[str] = Field(...)
    bug_types: List[str] = Field(...)

class LayerSurface(BaseModel):
    items: List[BugSurface] = Field(...)

class BugReport(BaseModel):
    backend: LayerSurface = Field(...)
    frontend: LayerSurface = Field(...)
    database: LayerSurface = Field(...)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

SEP  = "=" * 72
DASH = "-" * 72


def _read_file(path: Path) -> str:
    """Return file contents, or an error string."""
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"# READ_ERROR: {exc}"


def _write_file(path: Path, content: str) -> bool:
    """Write content to path, creating parent dirs as needed."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return True
    except Exception as exc:
        print(f"  [WRITE ERROR] {exc}")
        return False


def _flatten_report(report: BugReport) -> List[Dict[str, Any]]:
    """
    Flatten BugReport into an ordered list of task dicts.
    Order: backend → frontend → database (natural dependency flow).
    """
    tasks: List[Dict[str, Any]] = []
    for layer_name, layer in [
        ("backend",  report.backend),
        ("frontend", report.frontend),
        ("database", report.database),
    ]:
        for surface in layer.items:
            tasks.append({
                "layer":        layer_name,
                "file":         surface.file,
                "entry_points": surface.entry_points,
                "bug_types":    surface.bug_types,
            })
    return tasks


def _build_context_block(injection_context: List[Dict]) -> str:
    """Format already-injected bugs for the LLM cross-file context."""
    if not injection_context:
        return "No prior injections yet."
    lines = ["ALREADY INJECTED BUGS IN OTHER FILES (keep the system coherent):"]
    for c in injection_context:
        lines.append(f"  • [{c['layer']}] {c['file']}: {c['bug_summary']}")
    return "\n".join(lines)


def _extract_summary(raw: str) -> tuple[str, str]:
    """
    Split LLM response into (modified_code, bug_summary).
    The model appends:  # BUG_SUMMARY: <sentence>
    """
    bug_summary = ""
    clean_lines = []
    for line in raw.splitlines():
        if line.strip().startswith("# BUG_SUMMARY:"):
            bug_summary = line.strip().replace("# BUG_SUMMARY:", "").strip()
        else:
            clean_lines.append(line)
    return "\n".join(clean_lines), bug_summary


# ─────────────────────────────────────────────────────────────────────────────
# Context log writer  (appends incrementally so you can watch it live)
# ─────────────────────────────────────────────────────────────────────────────

def _append_context_log(log_path: Path, entry: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(entry + "\n")


def _init_context_log(log_path: Path, repo_path: str, total: int) -> None:
    header = "\n".join([
        SEP,
        "  ErrorX – Bug Injection Context Log",
        f"  Started   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Repo      : {repo_path}",
        f"  Files     : {total}",
        SEP, "",
    ])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(header, encoding="utf-8")


def _finalize_context_log(log_path: Path, succeeded: int, total: int) -> None:
    footer = "\n".join([
        "", SEP,
        f"  Finished  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Result    : {succeeded}/{total} succeeded",
        SEP,
    ])
    _append_context_log(log_path, footer)


# ─────────────────────────────────────────────────────────────────────────────
# Core per-file injection logic
# ─────────────────────────────────────────────────────────────────────────────

async def _inject_one_file(
    task: Dict[str, Any],
    repo_root: Path,
    injection_context: List[Dict],
    llm: ChatOpenAI,
) -> Dict[str, Any]:
    """
    Read one file, call the LLM to inject a bug, write back.
    Returns a result dict enriched with status / bug_summary.
    """
    filepath     = task["file"]
    entry_points = task["entry_points"]
    bug_types    = task["bug_types"]
    layer        = task["layer"]
    abs_path     = repo_root / filepath

    print(f"\n  Layer     : {layer}")
    print(f"  File      : {filepath}")
    print(f"  Entries   : {entry_points}")
    print(f"  Bug types : {bug_types}")

    # ── read source ──────────────────────────────────────────────────────────
    original_code = _read_file(abs_path)
    if original_code.startswith("# READ_ERROR"):
        print(f"  [SKIP] {original_code}")
        return {
            **task,
            "status":        "skipped",
            "reason":        original_code,
            "original_code": "",
            "modified_code": "",
            "bug_summary":   "",
        }

    # ── prompt ───────────────────────────────────────────────────────────────
    context_block = _build_context_block(injection_context)

    prompt = f"""You are a precise code sabotage agent for a bug-injection benchmark.

TASK:
- File         : {filepath}
- Layer        : {layer}
- Entry points : {entry_points}
- Bug categories (choose one): {bug_types}

CROSS-FILE CONTEXT (bugs already planted — keep the system coherent):
{context_block}

RULES:
1. Inject EXACTLY ONE subtle bug.
2. Place the bug at or near one of the listed entry points.
3. Pick the bug category that creates the most coherent cross-system failure
   when combined with already-planted bugs.
4. The bug must look like a plausible human mistake:
   off-by-one, wrong key, swapped condition, missing check, wrong operator, etc.
   NOT an obvious syntax error. NOT a comment change.
5. Return ONLY the complete modified source file.
   No markdown fences, no backticks, no preamble, no explanation.
6. After the file content append ONE comment line in EXACTLY this format:
   # BUG_SUMMARY: <one sentence describing what was injected and why>

ORIGINAL CODE:
{original_code}
"""

    try:
        response = await llm.ainvoke(prompt)
        raw = response.content.strip()
    except Exception as exc:
        print(f"  [LLM ERROR] {exc}")
        return {
            **task,
            "status":        "failed",
            "reason":        str(exc),
            "original_code": original_code,
            "modified_code": "",
            "bug_summary":   "",
        }

    # ── parse response ────────────────────────────────────────────────────────
    modified_code, bug_summary = _extract_summary(raw)

    # ── write modified file back ──────────────────────────────────────────────
    success = _write_file(abs_path, modified_code)
    status  = "success" if success else "failed"
    print(f"  [{status.upper()}] {bug_summary or '(no summary extracted)'}")

    return {
        **task,
        "status":        status,
        "original_code": original_code,
        "modified_code": modified_code,
        "bug_summary":   bug_summary or "(unknown)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph node  –  drop this into your StateGraph
# ─────────────────────────────────────────────────────────────────────────────

async def bug_injector_node(state: dict) -> dict:
    """
    ErrorX LangGraph node: inject bugs into every file listed in bug_report.

    Reads from state:
        state["bug_report"]   – dict matching BugReport schema
        state["repo_path"]    – absolute path to the cloned repo
        state["logs"]         – existing log list (optional)
        state["errors"]       – existing error list (optional)

    Writes back:
        state["injection_results"]  – list of per-file result dicts
        state["logs"]               – appended log lines
        state["errors"]             – appended error lines
        <repo_path>/context/abcd.txt – incremental context log on disk
    """

    # ── unpack state ──────────────────────────────────────────────────────────
    raw_report: dict   = state["bug_report"]
    repo_path:  str    = state["repo_path"]
    logs:       list   = list(state.get("logs",   []))
    errors:     list   = list(state.get("errors", []))

    repo_root = Path(repo_path)
    # Write context log to the project-level context/ directory
    repo_name = repo_root.name
    project_root = Path(__file__).resolve().parent.parent
    log_path  = project_root / "context" / f"{repo_name}.txt"

    # ── validate & parse report ───────────────────────────────────────────────
    try:
        report = BugReport(**raw_report)
    except Exception as exc:
        msg = f"[bug_injector_node] Invalid bug_report schema: {exc}"
        errors.append(msg)
        print(msg)
        return {**state, "errors": errors}

    # ── set up LLM ────────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENAI_KEY")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key,
        temperature=0.3,
    )

    # ── flatten tasks ─────────────────────────────────────────────────────────
    tasks = _flatten_report(report)
    total = len(tasks)

    print(SEP)
    print("  ErrorX – Bug Injection Agent")
    print(f"  Repo: {repo_path}")
    print(f"  Files to process: {total}")
    print(SEP)

    _init_context_log(log_path, repo_path, total)

    # ── fire ALL injections concurrently ──────────────────────────────────────
    import asyncio

    injection_context: List[Dict] = []   # empty — all tasks run in parallel

    async def _run_task(idx: int, task: Dict) -> Dict:
        print(DASH)
        print(f"  Task {idx}/{total}")
        return await _inject_one_file(task, repo_root, injection_context, llm)

    all_results = await asyncio.gather(
        *[_run_task(idx, task) for idx, task in enumerate(tasks, 1)]
    )

    # ── write context log & state logs AFTER all tasks complete ────────────────
    for idx, result in enumerate(all_results, 1):
        log_entry_lines = [
            DASH,
            f"[{idx}/{total}]  [{result['status'].upper()}]  {result['file']}  ({result['layer']})",
            f"  Entry points : {result.get('entry_points', [])}",
            f"  Bug types    : {result.get('bug_types', [])}",
            f"  Bug planted  : {result.get('bug_summary', 'n/a')}",
        ]

        if result["status"] == "failed":
            log_entry_lines.append(f"  Reason       : {result.get('reason', '')}")

        if result.get("original_code"):
            log_entry_lines += [
                "",
                "  ── ORIGINAL " + "─" * 56,
                result["original_code"],
            ]

        if result.get("modified_code"):
            log_entry_lines += [
                "",
                "  ── MODIFIED " + "─" * 56,
                result["modified_code"],
            ]

        log_entry_lines.append("")
        _append_context_log(log_path, "\n".join(log_entry_lines))

        status_tag = {"success": "✓", "failed": "✗", "skipped": "–"}.get(
            result["status"], "?"
        )
        logs.append(
            f"[bug_injector] {status_tag} {result['file']} – {result.get('bug_summary', result.get('reason', ''))}"
        )
        if result["status"] == "failed":
            errors.append(f"[bug_injector] FAIL {result['file']}: {result.get('reason', '')}")

    # ── finalize log ──────────────────────────────────────────────────────────
    succeeded = sum(1 for r in all_results if r["status"] == "success")
    _finalize_context_log(log_path, succeeded, total)

    print(f"\n{SEP}")
    print(f"  Done. {succeeded}/{total} succeeded.")
    print(f"  Context log → {log_path}")
    print(SEP)

    return {
        **state,
        "injection_results": all_results,
        "logs":              logs,
        "errors":            errors,
    }