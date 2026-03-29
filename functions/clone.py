import os
import json
import uuid
import subprocess
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from models import ErrorX

load_dotenv()

# -------------------- MCP CONFIG --------------------
SERVERS = {
    "github": {
        "transport": "stdio",
        "command": "docker",
        "args": [
            "run",
            "-i",
            "--rm",
            "-e",
            "GITHUB_PERSONAL_ACCESS_TOKEN",
            "ghcr.io/github/github-mcp-server"
        ],
        "env": {
            "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_KEY")
        }
    }
}


# -------------------- NODE 1: SEARCH --------------------
async def github_search_node(state: ErrorX) -> Dict[str, Any]:

    tech_stack = state.get("tech_stack", [])

    client = MultiServerMCPClient(SERVERS)
    tools = await client.get_tools()

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_KEY")
    )

    llm_with_tools = llm.bind_tools(tools)

    query = f"Search GitHub for repositories using: {', '.join(tech_stack)}"

    response = await llm_with_tools.ainvoke(query)

    tools_by_name = {tool.name: tool for tool in tools}
    tool_outputs = []

    for call in response.tool_calls:
        tool = tools_by_name[call["name"]]
        result = await tool.ainvoke(call["args"])
        tool_outputs.append(result)

    # -------------------- PARSE OUTPUT --------------------
    parsed_items = []

    for r in tool_outputs:
        if isinstance(r, list):
            for item in r:
                if "text" in item:
                    try:
                        data = json.loads(item["text"])
                        parsed_items.extend(data.get("items", []))
                    except Exception:
                        continue

    if not parsed_items:
        return {
            **state,
            "errors": state.get("errors", []) + ["No repositories found"],
            "logs": state.get("logs", []) + ["MCP returned empty results"]
        }

    # -------------------- RANK --------------------
    parsed_items = sorted(
        parsed_items,
        key=lambda x: x.get("stargazers_count", 0),
        reverse=True
    )

    top_repos = parsed_items[:3]
    selected = top_repos[0]

    return {
        **state,
        "candidate_repos": top_repos,
        "selected_repo": selected,
        "github_link": selected.get("html_url", ""),
        "logs": state.get("logs", []) + [f"Selected repo: {selected.get('html_url', '')}"]
    }


# -------------------- NODE 2: CLONE --------------------
def clone_repo_node(state: ErrorX) -> Dict[str, Any]:

    github_link = state.get("github_link")

    if not github_link:
        return {
            **state,
            "errors": state.get("errors", []) + ["No GitHub link found"]
        }

    # 🔥 NEW STRUCTURE
    workspace_id = uuid.uuid4().hex[:8]
    base_workspace = os.path.join("workspaces", workspace_id)

    os.makedirs(base_workspace, exist_ok=True)

    repo_name = github_link.rstrip("/").split("/")[-1]
    repo_path = os.path.join(base_workspace, repo_name)

    try:
        subprocess.run(
            ["git", "clone", github_link, repo_path],
            check=True
        )
    except Exception as e:
        return {
            **state,
            "errors": state.get("errors", []) + [str(e)]
        }

    return {
        **state,
        "workspace_id": workspace_id,                 # 🔥 useful for tracking
        "workspace_path": base_workspace,             # 🔥 updated path
        "repo_path": repo_path,
        "cloned": True,
        "logs": state.get("logs", []) + [
            f"Cloned repo to {repo_path}"
        ]
    }