import os
import json
import asyncio

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from models import ErrorX
from functions.clone import github_search_node, clone_repo_node
from functions.analyze_surface import analyze_surface_node
from functions.build_tree_node import build_tree_node
from functions.bug_injection import bug_injector_node
from functions.coco_index import indexer_node


load_dotenv()


# -------------------- GRAPH --------------------
def build_graph():
    graph = StateGraph(ErrorX)

    graph.add_node("github_search", github_search_node)
    graph.add_node("clone_repo", clone_repo_node)
    graph.add_node("build_tree", build_tree_node)
    graph.add_node("analyze_surface", analyze_surface_node)
    graph.add_node("inject_bugs", bug_injector_node)
    graph.add_node("coco_index", indexer_node)

    graph.add_edge(START, "github_search")
    graph.add_edge("github_search", "clone_repo")
    graph.add_edge("clone_repo", "build_tree")
    graph.add_edge("build_tree",    "coco_index")
    graph.add_edge("coco_index",    "analyze_surface")
    graph.add_edge("analyze_surface", "inject_bugs")
    graph.add_edge("inject_bugs", END)

    return graph.compile()


# -------------------- MAIN --------------------
async def main():
    graph = build_graph()

    result = await graph.ainvoke({
        "tech_stack": ["flask"],
        "logs": [],
        "errors": []
    })

    print("\n===== FINAL OUTPUT =====\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())