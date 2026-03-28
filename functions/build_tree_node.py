import os
from models import ErrorX


def build_tree_string(path, prefix=""):
    tree_str = ""
    try:
        items = sorted(os.listdir(path))
    except PermissionError:
        return ""

    for i, item in enumerate(items):
        full_path = os.path.join(path, item)
        connector = "└── " if i == len(items) - 1 else "├── "
        tree_str += prefix + connector + item + "\n"

        if os.path.isdir(full_path):
            extension = "    " if i == len(items) - 1 else "│   "
            tree_str += build_tree_string(full_path, prefix + extension)

    return tree_str


# -------------------- NODE --------------------

def build_tree_node(state: ErrorX) -> dict:
    repo_path = state.get("repo_path", "")
    logs = state.get("logs", [])
    errors = state.get("errors", [])

    if not repo_path or not os.path.exists(repo_path):
        return {
            "errors": errors + ["build_tree: repo_path is missing or does not exist"],
        }

    tree_structure = build_tree_string(repo_path)

    if not tree_structure:
        return {
            "errors": errors + ["build_tree: generated tree is empty"],
        }

    return {
        "tree_structure": tree_structure,
        "logs": logs + [f"build_tree: tree built from {repo_path}"],
    }