"""SQLAlchemy ORM models for AF2 User Projects Library."""
import json
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

_DB_PATH  = Path(__file__).parents[2] / "data" / "shear_projects.db"
_NPY_DIR  = Path(__file__).parents[2] / "data" / "project_fields"


def get_engine(db_path: Path = _DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})


def get_session_factory(engine=None):
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


def npy_dir() -> Path:
    _NPY_DIR.mkdir(parents=True, exist_ok=True)
    return _NPY_DIR


class Project(Base):
    __tablename__ = "projects"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    name          = Column(String(120), nullable=False)
    tags          = Column(String(256), default="")   # comma-separated
    geometry_type = Column(String(64),  nullable=False)
    params_json   = Column(Text,        nullable=False)   # JSON string
    mean_tau_path = Column(String(512), nullable=True)    # path to .npy
    variance_path = Column(String(512), nullable=True)
    positions_path= Column(String(512), nullable=True)
    grad_u_path   = Column(String(512), nullable=True)
    metrics_json  = Column(Text,        default="{}")     # peak_tau, backscatter_frac, etc.
    thumbnail_b64 = Column(Text,        nullable=True)    # base64 PNG
    created_at    = Column(DateTime,    default=datetime.utcnow)

    # ---- helpers ----
    def params(self) -> dict:
        return json.loads(self.params_json)

    def metrics(self) -> dict:
        return json.loads(self.metrics_json or "{}")

    def tag_list(self) -> list[str]:
        return [t.strip() for t in self.tags.split(",") if t.strip()]

    def to_dict(self) -> dict:
        return {
            "id":            self.id,
            "name":          self.name,
            "tags":          self.tag_list(),
            "geometry_type": self.geometry_type,
            "params":        self.params(),
            "metrics":       self.metrics(),
            "thumbnail_b64": self.thumbnail_b64,
            "created_at":    self.created_at.isoformat() if self.created_at else None,
        }
