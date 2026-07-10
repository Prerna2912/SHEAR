"""CRUD operations for AF2 project library."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from .models import Project, npy_dir


def save_project(
    session,
    name:          str,
    geometry_type: str,
    params:        dict,
    mean_tau:      list | np.ndarray,
    variance_tau:  list | np.ndarray,
    positions:     list | np.ndarray,
    grad_u:        list | np.ndarray,
    metrics:       dict,
    tags:          list[str] | None = None,
    thumbnail_b64: str | None = None,
) -> Project:
    uid  = uuid.uuid4().hex[:12]
    base = npy_dir() / uid

    def _save(arr, suffix):
        p = Path(f"{base}_{suffix}.npy")
        np.save(p, np.array(arr, dtype=np.float32))
        return str(p)

    p = Project(
        name          = name,
        tags          = ",".join(tags or []),
        geometry_type = geometry_type,
        params_json   = json.dumps(params),
        mean_tau_path = _save(mean_tau,    "mean_tau"),
        variance_path = _save(variance_tau, "variance"),
        positions_path= _save(positions,   "positions"),
        grad_u_path   = _save(grad_u,      "grad_u"),
        metrics_json  = json.dumps(metrics),
        thumbnail_b64 = thumbnail_b64,
    )
    session.add(p)
    session.commit()
    session.refresh(p)
    return p


def load_project_fields(project: Project) -> dict:
    """Load numpy field arrays for a saved project."""
    def _load(path):
        if path and Path(path).exists():
            return np.load(path).tolist()
        return None

    return {
        "mean_tau":     _load(project.mean_tau_path),
        "variance_tau": _load(project.variance_path),
        "positions":    _load(project.positions_path),
        "grad_u":       _load(project.grad_u_path),
    }


def list_projects(session, tag: str | None = None) -> list[Project]:
    q = session.query(Project).order_by(Project.created_at.desc())
    if tag:
        q = q.filter(Project.tags.contains(tag))
    return q.all()


def get_project(session, project_id: int) -> Optional[Project]:
    return session.query(Project).filter(Project.id == project_id).first()


def delete_project(session, project_id: int) -> bool:
    p = get_project(session, project_id)
    if not p:
        return False
    for path in [p.mean_tau_path, p.variance_path, p.positions_path, p.grad_u_path]:
        if path:
            fp = Path(path)
            if fp.exists():
                fp.unlink(missing_ok=True)
    session.delete(p)
    session.commit()
    return True
