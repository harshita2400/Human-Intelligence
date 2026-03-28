from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field

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

class ErrorX(TypedDict, total=False):
    tech_stack: List[str]

    candidate_repos: List[Dict[str, Any]]
    selected_repo: Dict[str, Any]
    github_link: str
    cloned: bool

    workspace_path: str
    repo_path: str

    tree_structure: str          # raw tree string from build_tree node
    bug_report: Dict[str, Any]   # serialized BugReport

    logs: List[str]
    errors: List[str]