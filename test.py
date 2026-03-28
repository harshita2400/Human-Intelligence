import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph

# ── encoding fix for Windows terminals ───────────────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Schema  (mirrors the analysis agent output)
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
# 2. Config
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(r"C:\Users\saumy\OneDrive\Desktop\Hackmol 7.0\repo\flask")

SEP  = "=" * 72
DASH = "-" * 72

# One shared LLM instance
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key= "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",  
    temperature=0.3,
)


# ─────────────────────────────────────────────────────────────────────────────
# 3. File helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve(relative: str) -> Path:
    """Turn a repo-relative path into an absolute Path."""
    return REPO_ROOT / relative


def read_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"# READ ERROR: {e}"


def write_file(path: Path, content: str) -> bool:
    try:
        path.write_text(content, encoding="utf-8")
        return True
    except Exception as e:
        print(f"  [WRITE ERROR] {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 4. Task flattening
# ─────────────────────────────────────────────────────────────────────────────

def flatten_report(report: BugReport) -> List[dict]:
    """
    Returns a flat list of task dicts, one per BugSurface, with layer info.
    Order: backend → frontend → database  (so context flows naturally)
    """
    tasks = []
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


# ─────────────────────────────────────────────────────────────────────────────
# 5. LangGraph node
# ─────────────────────────────────────────────────────────────────────────────

def inject_bug_node(state: dict) -> dict:
    """
    Single LangGraph node.

    Reads:  state["task"]             – current BugSurface task dict
            state["injection_context"] – list of already-injected bug summaries
    Writes: updated state with result appended to injection_context
    """
    task             = state["task"]
    injection_context: list = state.get("injection_context", [])

    filepath    = task["file"]
    entry_points = task["entry_points"]
    bug_types    = task["bug_types"]
    layer        = task["layer"]

    abs_path = resolve(filepath)

    print(f"\n  Layer     : {layer}")
    print(f"  File      : {filepath}")
    print(f"  Entries   : {entry_points}")
    print(f"  Bug types : {bug_types}")

    # ── read source ──────────────────────────────────────────────────────────
    original_code = read_file(abs_path)
    if original_code.startswith("# READ ERROR"):
        print(f"  [SKIP] {original_code}")
        result = {
            **task,
            "status":        "skipped",
            "reason":        original_code,
            "original_code": "",
            "modified_code": "",
            "bug_summary":   "",
        }
        return {**state, "last_result": result}

    # ── build cross-file context block ───────────────────────────────────────
    if injection_context:
        ctx_lines = ["ALREADY INJECTED BUGS IN OTHER FILES (for coherence):"]
        for c in injection_context:
            ctx_lines.append(
                f"  • [{c['layer']}] {c['file']}: {c['bug_summary']}"
            )
        context_block = "\n".join(ctx_lines)
    else:
        context_block = "No prior injections yet."

    # ── prompt ───────────────────────────────────────────────────────────────
    prompt = f"""You are a precise code sabotage agent for a bug-injection benchmark.

TASK:
- File     : {filepath}
- Layer    : {layer}
- Entry points where the bug must be planted: {entry_points}
- Allowed bug categories: {bug_types}

CROSS-FILE CONTEXT (bugs already planted elsewhere — keep the system coherent):
{context_block}

RULES:
1. Inject EXACTLY ONE subtle bug.
2. The bug MUST be placed at or near one of the listed entry points.
3. Choose the bug category from the allowed list that creates the most coherent
   cross-system failure when combined with already-planted bugs.
4. The bug must look like a plausible human mistake (off-by-one, wrong key,
   swapped condition, missing await, etc.).  NOT an obvious syntax error.
5. Return ONLY the complete modified source file.  No markdown, no backticks,
   no explanations.
6. After the file content, append a single comment line in this exact format:
   # BUG_SUMMARY: <one sentence describing what was injected and why>

ORIGINAL CODE:
{original_code}
"""

    try:
        response = llm.invoke(prompt)
        raw = response.content.strip()
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        result = {
            **task,
            "status":        "failed",
            "reason":        str(e),
            "original_code": original_code,
            "modified_code": "",
            "bug_summary":   "",
        }
        return {**state, "last_result": result}

    # ── extract bug summary comment ───────────────────────────────────────────
    bug_summary = ""
    lines = raw.splitlines()
    clean_lines = []
    for line in lines:
        if line.strip().startswith("# BUG_SUMMARY:"):
            bug_summary = line.strip().replace("# BUG_SUMMARY:", "").strip()
        else:
            clean_lines.append(line)
    modified_code = "\n".join(clean_lines)

    # ── write back ────────────────────────────────────────────────────────────
    success = write_file(abs_path, modified_code)
    status  = "success" if success else "failed"
    print(f"  [{status.upper()}] {bug_summary or '(no summary extracted)'}")

    result = {
        **task,
        "status":        status,
        "original_code": original_code,
        "modified_code": modified_code,
        "bug_summary":   bug_summary,
    }

    # ── update shared context ─────────────────────────────────────────────────
    new_context = injection_context + [{
        "layer":       layer,
        "file":        filepath,
        "bug_summary": bug_summary or "(unknown)",
    }]

    return {
        **state,
        "last_result":       result,
        "injection_context": new_context,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Build graph (single node — iterating externally for clean state threading)
# ─────────────────────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(dict)
    builder.add_node("inject", inject_bug_node)
    builder.set_entry_point("inject")
    return builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Report writer
# ─────────────────────────────────────────────────────────────────────────────

def write_report(all_results: list, report_path: Path):
    lines = [
        SEP,
        "  BUG INJECTION REPORT  –  Context-Aware Multi-Layer Agent",
        f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Repo root : {REPO_ROOT}",
        f"  Files processed : {len(all_results)}",
        SEP,
    ]

    succeeded = [r for r in all_results if r.get("status") == "success"]
    failed    = [r for r in all_results if r.get("status") == "failed"]
    skipped   = [r for r in all_results if r.get("status") == "skipped"]

    lines += [
        "",
        f"  Succeeded : {len(succeeded)}",
        f"  Failed    : {len(failed)}",
        f"  Skipped   : {len(skipped)}",
        "",
    ]

    # ── cross-file relationship summary ──────────────────────────────────────
    lines += [
        SEP,
        "  CROSS-FILE BUG RELATIONSHIP MAP",
        SEP,
    ]
    for r in all_results:
        if r.get("bug_summary"):
            lines.append(
                f"  [{r['layer'].upper():<8}] {r['file']:<55} | {r['bug_summary']}"
            )

    # ── per-file detail ───────────────────────────────────────────────────────
    lines += ["", SEP, "  DETAILED INJECTION LOG", SEP]

    for i, r in enumerate(all_results, 1):
        tag = {"success": "[OK]", "failed": "[FAIL]", "skipped": "[SKIP]"}.get(
            r.get("status", ""), "[???]"
        )
        lines += [
            "",
            DASH,
            f"[{i}] {tag}  {r['file']}  ({r['layer']})",
            f"     Status      : {r.get('status', '?').upper()}",
            f"     Entry pts   : {r.get('entry_points', [])}",
            f"     Bug types   : {r.get('bug_types', [])}",
            f"     Bug planted : {r.get('bug_summary', 'n/a')}",
        ]
        if r.get("status") == "failed":
            lines.append(f"     Reason      : {r.get('reason', '')}")

        if r.get("original_code"):
            lines += ["", "  -- ORIGINAL " + "-" * 58, r["original_code"]]
        if r.get("modified_code"):
            lines += ["", "  -- MODIFIED " + "-" * 58, r["modified_code"]]

    lines += ["", SEP, "  End of report", SEP]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  Report saved → {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run(report: BugReport, execute: bool = True):
    """
    Iterates over every BugSurface in the report, injecting bugs one by one
    while threading the accumulated injection_context through each step.
    """
    print(SEP)
    print("  CONTEXT-AWARE BUG INJECTION AGENT  –  LangGraph")
    print(SEP)

    graph  = build_graph()
    tasks  = flatten_report(report)
    print(f"  Total files to process : {len(tasks)}\n")

    # Shared mutable context threaded through graph invocations
    injection_context: list = []
    all_results: list       = []

    for i, task in enumerate(tasks, 1):
        print(DASH)
        print(f"  Task {i}/{len(tasks)}")

        if not execute:
            print(f"  [PREVIEW] {task['file']}  →  {task['bug_types']}")
            all_results.append({**task, "status": "preview",
                                 "original_code": "", "modified_code": "",
                                 "bug_summary": ""})
            continue

        state = {
            "task":              task,
            "injection_context": injection_context,
        }

        output            = graph.invoke(state)
        result            = output["last_result"]
        injection_context = output.get("injection_context", injection_context)

        all_results.append(result)

    # ── final report ─────────────────────────────────────────────────────────
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPO_ROOT / f"injection_report_{ts}.txt"
    write_report(all_results, report_path)

    succeeded = sum(1 for r in all_results if r.get("status") == "success")
    print(f"\n{SEP}")
    print(f"  Done. {succeeded}/{len(all_results)} succeeded.")
    print(SEP)

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# 9. Entry point  –  paste your BugReport here
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_report = BugReport(
        backend=LayerSurface(items=[
            BugSurface(
                file="src/flask/__init__.py",
                entry_points=["API routes", "request handlers"],
                bug_types=["api_contract", "auth", "logic", "validation"],
            ),
            BugSurface(
                file="src/flask/views.py",
                entry_points=["view functions"],
                bug_types=["api_contract", "logic", "validation"],
            ),
        ]),
        frontend=LayerSurface(items=[
            BugSurface(
                file="src/flask/static/js/app.js",
                entry_points=["user interactions", "API calls"],
                bug_types=["async", "state", "event"],
            ),
        ]),
        database=LayerSurface(items=[
            BugSurface(
                file="src/flask/models.py",
                entry_points=["queries", "schema operations"],
                bug_types=["injection", "transaction", "query"],
            ),
        ]),
    )

    run(sample_report, execute=True)