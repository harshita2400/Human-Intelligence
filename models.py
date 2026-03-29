import operator
from typing import Annotated, TypedDict, List, Dict, Any, Optional
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

    tree_structure: str
    bug_report: Dict[str, Any]
    injection_results: List[Dict[str, Any]]

    index_path: str
    index_meta_path: str

    # operator.add tells LangGraph to merge (append) these lists
    # across nodes instead of treating concurrent writes as a conflict
    logs:   Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]