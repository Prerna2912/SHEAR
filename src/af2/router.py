"""FastAPI router for AF2 User Projects Library."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from .models import get_engine, get_session_factory, Project
from .crud import save_project, list_projects, get_project, delete_project, load_project_fields

router = APIRouter(prefix="/api/v1/projects", tags=["projects"])

# ---- dependency ----
_factory = None


def _get_session():
    global _factory
    if _factory is None:
        _factory = get_session_factory(get_engine())
    db = _factory()
    try:
        yield db
    finally:
        db.close()


# ---- schemas ----

class SaveProjectRequest(BaseModel):
    name:          str = Field(..., min_length=1, max_length=120)
    tags:          List[str] = []
    geometry_type: str
    params:        Dict[str, Any]
    mean_tau:      List[List[float]]
    variance_tau:  List[List[float]]
    positions:     List[List[float]]
    grad_u:        List[List[float]]
    metrics:       Dict[str, Any] = {}
    thumbnail_b64: Optional[str]  = None


class ProjectSummary(BaseModel):
    id:            int
    name:          str
    tags:          List[str]
    geometry_type: str
    params:        Dict[str, Any]
    metrics:       Dict[str, Any]
    thumbnail_b64: Optional[str]
    created_at:    Optional[str]


class ProjectDetail(ProjectSummary):
    mean_tau:     Optional[List[Any]]
    variance_tau: Optional[List[Any]]
    positions:    Optional[List[Any]]
    grad_u:       Optional[List[Any]]


# ---- endpoints ----

@router.post("", response_model=ProjectSummary, status_code=201)
def create_project(req: SaveProjectRequest, db=Depends(_get_session)):
    p = save_project(
        session       = db,
        name          = req.name,
        geometry_type = req.geometry_type,
        params        = req.params,
        mean_tau      = req.mean_tau,
        variance_tau  = req.variance_tau,
        positions     = req.positions,
        grad_u        = req.grad_u,
        metrics       = req.metrics,
        tags          = req.tags,
        thumbnail_b64 = req.thumbnail_b64,
    )
    return ProjectSummary(**p.to_dict())


@router.get("", response_model=List[ProjectSummary])
def get_projects(tag: Optional[str] = None, db=Depends(_get_session)):
    projects = list_projects(db, tag=tag)
    return [ProjectSummary(**p.to_dict()) for p in projects]


@router.get("/{project_id}", response_model=ProjectDetail)
def get_project_detail(project_id: int, db=Depends(_get_session)):
    p = get_project(db, project_id)
    if not p:
        raise HTTPException(status_code=404, detail="Project not found")
    fields = load_project_fields(p)
    return ProjectDetail(**p.to_dict(), **fields)


@router.delete("/{project_id}", status_code=204)
def remove_project(project_id: int, db=Depends(_get_session)):
    if not delete_project(db, project_id):
        raise HTTPException(status_code=404, detail="Project not found")
