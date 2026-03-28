from typing import TypedDict, List, Dict, Any

class ErrorX(TypedDict, total=False):
    tech_stack: List[str]

    candidate_repos: List[Dict[str, Any]]
    selected_repo: Dict[str, Any]
    github_link: str
    cloned: bool

    workspace_path: str
    repo_path: str

    logs: List[str]
    errors: List[str]