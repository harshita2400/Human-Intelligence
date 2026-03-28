import os
import json
import asyncio

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from models import ErrorX
from functions.clone import github_search_node, clone_repo_node

load_dotenv()


# -------------------- GRAPH --------------------
def build_graph():

    graph = StateGraph(ErrorX)

    graph.add_node("github_search", github_search_node)
    graph.add_node("clone_repo", clone_repo_node)

    graph.add_edge(START, "github_search")
    graph.add_edge("github_search", "clone_repo")
    graph.add_edge("clone_repo", END)

    return graph.compile()


# -------------------- MAIN --------------------
async def main():

    graph = build_graph()

    result = await graph.ainvoke({
        "tech_stack": ["Flask", "MongoDB"],
        "logs": [],
        "errors": []
    })

    print("\n===== FINAL OUTPUT =====\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())