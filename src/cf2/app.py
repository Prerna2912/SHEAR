"""
CF2 — FastAPI serving application.

Endpoints
---------
POST   /api/v1/infer                 Submit inference request → job_id
GET    /api/v1/jobs/{job_id}         Poll job status / get result
GET    /api/v1/jobs/{job_id}/result  Same as above (alias for CF5 explorer)
DELETE /api/v1/cache/{geometry_hash} Manually invalidate a cache entry
GET    /api/v1/cache/stats           Cache occupancy
WS     /ws/progress/{job_id}         Stream sample-by-sample progress

Startup / configuration
-----------------------
The application expects two environment variables (or explicit paths):
    V1_CHECKPOINT  path to runs/.../v1/best.pt  (or last.pt)
    V1_STATS       path to runs/.../stats.pt

Defaults to runs/quick/v1/last.pt and runs/quick/stats.pt if not set.

Run locally:
    uvicorn src.cf2.app:app --host 0.0.0.0 --port 7860 --reload

HuggingFace Spaces (Docker SDK):
    CMD ["uvicorn", "src.cf2.app:app", "--host", "0.0.0.0", "--port", "7860"]
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from .schemas import (
    CF4Request,
    CF4Result,
    ExplorerJobRequest,
    ExplorerResult,
    GeometryRequest,
    InferenceResult,
    JobResponse,
    JobStatus,
    ParameterRanking,
    PerturbationPoint,
    InteractionMap,
    ProgressMessage,
)
from .cache import get_cache
from .checkpoint import init_model
from .queue import get_queue, make_job


# ---------------------------------------------------------------------------
# Startup / shutdown lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load V1 model once at startup, start GPU worker."""
    ckpt = os.environ.get("V1_CHECKPOINT", "runs/quick/v1/last.pt")
    stats = os.environ.get("V1_STATS", "runs/quick/stats.pt")

    print(f"[CF2] Loading V1 model from {ckpt}")
    init_model(checkpoint_path=ckpt, stats_path=stats)

    queue = get_queue()
    worker_task = asyncio.create_task(queue.start_worker())
    print("[CF2] GPU worker started")

    yield

    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="SHEAR CF2 — V1 Inference Engine",
    version="1.0.0",
    description=(
        "Production-grade SE(3)-equivariant SGS stress inference. "
        "Submit a geometry, get back mean τ, variance, and optionally all CFM samples."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# POST /api/v1/infer  — submit inference request
# ---------------------------------------------------------------------------

@app.post("/api/v1/infer", response_model=JobResponse, status_code=202)
async def submit_inference(request: GeometryRequest):
    """
    Submit a geometry for V1 inference.

    Returns immediately with a job_id.  Poll GET /api/v1/jobs/{job_id} or
    subscribe to WS /ws/progress/{job_id} for progress and the final result.
    """
    from .inference import geometry_hash

    ghash = geometry_hash(request.geometry_type, request.params)

    # Cache hit: return a synthetic job that is immediately complete

     cache = get_cache()
    cached = cache.get(ghash)
    if cached is not None:
        import uuid
        job_id = str(uuid.uuid4())
        return JobResponse(
            job_id=job_id,
            status=JobStatus.COMPLETE,
            geometry_hash=ghash,
        )

    job = make_job(
        geometry_type=request.geometry_type,
        params=request.params,
        n_samples=request.n_samples,
        ode_steps=request.ode_steps,
        grid_size=request.grid_size,
        include_samples=request.include_samples,
    )

    queue = get_queue()
    await queue.submit(job)

    return JobResponse(
        job_id=job.job_id,
        status=JobStatus.QUEUED,
        geometry_hash=ghash,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/jobs/{job_id}  — poll job / get result
# ---------------------------------------------------------------------------

@app.get("/api/v1/jobs/{job_id}", response_model=InferenceResult)
async def get_job_result(job_id: str):
    """
    Return the status and result of a previously submitted inference job.

    While running: returns status=running with empty fields.
    On error: returns status=error with error message in .error field.
    On complete: returns the full InferenceResult.
    """
    queue = get_queue()
    job = queue.get_job(job_id)

    # May also be a cache-hit job (result already in cache, no job entry)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found.")

    if job.status == JobStatus.ERROR:
        raise HTTPException(status_code=500, detail=job.error or "Inference failed.")

    if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        raise HTTPException(
            status_code=202,
            detail=f"Job {job_id!r} is {job.status.value}. "
                   "Subscribe to /ws/progress/{job_id} for updates.",
        )

    return job.result


# Alias for CF5 parametric explorer clarity
app.get("/api/v1/jobs/{job_id}/result", response_model=InferenceResult)(get_job_result)


# ---------------------------------------------------------------------------
# GET /api/v1/cache/stats
# ---------------------------------------------------------------------------

@app.get("/api/v1/cache/stats")
async def cache_stats():
    cache = get_cache()
    return {"size": len(cache), "keys": cache.keys}


# ---------------------------------------------------------------------------
# DELETE /api/v1/cache/{geometry_hash}
# ---------------------------------------------------------------------------

@app.delete("/api/v1/cache/{geometry_hash}")
async def invalidate_cache(geometry_hash: str):
    cache = get_cache()
    removed = cache.invalidate(geometry_hash)
    return {"removed": removed, "geometry_hash": geometry_hash}


# ---------------------------------------------------------------------------
# WS /ws/progress/{job_id}  — live sample-by-sample progress
# ---------------------------------------------------------------------------

@app.websocket("/ws/progress/{job_id}")
async def ws_progress(websocket: WebSocket, job_id: str):
    """
    Stream ProgressMessage JSON over WebSocket until the job is done.

    Example client (JavaScript):
        const ws = new WebSocket(`ws://host/ws/progress/${jobId}`);
        ws.onmessage = e => {
            const msg = JSON.parse(e.data);
            updateProgressBar(msg.sample / msg.n_samples);
            if (msg.status === "complete") ws.close();
        };
    """
    await websocket.accept()

    queue = get_queue()
    job = queue.get_job(job_id)

    if job is None:
        await websocket.send_json({"error": f"Job {job_id!r} not found"})
        await websocket.close(code=4004)
        return

    # Already done before WS connected
    if job.status in (JobStatus.COMPLETE, JobStatus.ERROR):
        msg = ProgressMessage(
            job_id=job_id,
            sample=job.n_samples if job.status == JobStatus.COMPLETE else 0,
            n_samples=job.n_samples,
            status=job.status,
            message="Complete" if job.status == JobStatus.COMPLETE else (job.error or "Error"),
        )
        await websocket.send_json(msg.model_dump())
        await websocket.close()
        return

    # Subscribe for live progress
    sub_queue = queue.subscribe_progress(job_id)

    try:
        while True:
            try:
                msg: ProgressMessage = await asyncio.wait_for(
                    sub_queue.get(), timeout=120.0
                )
            except asyncio.TimeoutError:
                # Keep-alive ping
                await websocket.send_json({"ping": True})
                continue

            await websocket.send_json(msg.model_dump())

            if msg.status in (JobStatus.COMPLETE, JobStatus.ERROR):
                break

    except WebSocketDisconnect:
        pass
    finally:
        queue.unsubscribe_progress(job_id, sub_queue)
        try:
            await websocket.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# POST /api/v1/cf4  — compute CF4 diagnostics for a given inference result
# ---------------------------------------------------------------------------

@app.post("/api/v1/cf4", response_model=CF4Result)
async def compute_cf4(request: CF4Request):
    """
    Compute CF4 diagnostics (Q-criterion, uncertainty, SGS dissipation,
    backscatter fraction) for an already-computed inference result.

    Accepts the mean_tau, variance_tau, grad_u fields directly from InferenceResult
    so the frontend never needs to re-run CF1/CF2 for the diagnostics layer.
    """
    import numpy as np
    from src.cf4.compute import analyse

    cf4 = analyse(
        mean_tau=request.mean_tau,
        variance_tau=request.variance_tau,
        grad_u=request.grad_u,
    )

    return CF4Result(
        q_values=cf4.q_values.tolist(),
        regime_labels=cf4.regime_labels.tolist(),
        regime_thresholds=list(cf4.regime_thresholds),
        uncertainty=cf4.uncertainty.tolist(),
        high_unc_mask=cf4.high_unc_mask.tolist(),
        dissipation=cf4.dissipation.tolist(),
        backscatter_mask=cf4.backscatter_mask.tolist(),
        backscatter_frac=cf4.backscatter_frac,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/explorer  — CF5 parametric stress explorer
# ---------------------------------------------------------------------------

@app.post("/api/v1/explorer", response_model=ExplorerResult)
async def run_explorer(request: ExplorerJobRequest):
    """
    Run the CF5 parametric stress explorer.

    The backend auto-computes ±5/10/20% perturbations for each numeric
    parameter, ranks them by |Δ peak τ|, and builds a 2D interaction map
    for the top-2 parameters.

    Wall time: ~10–60s depending on parameter count and cache state.
    Cache is shared with the main inference path — the base geometry call
    is free after the first inference.
    """
    from src.cf5.explorer import run_explorer_sync, ExplorerResult as _ExplorerResult

    loop = asyncio.get_event_loop()

    raw: _ExplorerResult = await loop.run_in_executor(
        None,
        run_explorer_sync,
        request.geometry_type,
        request.params,
        request.n_samples,
        request.grid_size,
    )

    # Convert dataclasses → Pydantic models
    ranking = [
        ParameterRanking(
            param=r.param,
            base_value=r.base_value,
            max_abs_delta=r.max_abs_delta,
            perturbations=[
                PerturbationPoint(
                    level=p.level,
                    param=p.param,
                    new_value=p.new_value,
                    delta_peak_tau=p.delta_peak_tau,
                    peak_tau=p.peak_tau,
                )
                for p in r.perturbations
            ],
        )
        for r in raw.parameter_ranking
    ]

    imap = None
    if raw.interaction_map is not None:
        m = raw.interaction_map
        imap = InteractionMap(
            param_a=m.param_a,
            param_b=m.param_b,
            base_value_a=m.base_value_a,
            base_value_b=m.base_value_b,
            levels=m.levels,
            grid=m.grid,
        )

    return ExplorerResult(
        geometry_type=raw.geometry_type,
        base_params=raw.base_params,
        base_peak_tau=raw.base_peak_tau,
        parameter_ranking=ranking,
        interaction_map=imap,
        n_inference_calls=raw.n_inference_calls,
        total_time_ms=raw.total_time_ms,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/af1  — Flow Regime Diagnostics (Task 3 pipeline)
# ---------------------------------------------------------------------------

@app.post("/api/v1/af1")
async def compute_af1(request: CF4Request):
    """
    AF1: invariant KDE, OOD detection, per-regime Pearson r from JHTDB stats.
    """
    from src.af1.diagnostics import analyse as af1_analyse
    from dataclasses import asdict

    result = af1_analyse(mean_tau=request.mean_tau, grad_u=request.grad_u)
    return asdict(result)


# ---------------------------------------------------------------------------
# AF2  — User Projects Library (router)
# ---------------------------------------------------------------------------

from src.af2.router import router as _af2_router
app.include_router(_af2_router)


# ---------------------------------------------------------------------------
# AF3  — Comparative Geometry Benchmarking
# ---------------------------------------------------------------------------

@app.get("/api/v1/af3/references")
async def list_references():
    from src.af3.references import REFERENCE_CASES
    return {"references": [{"id": r["id"], "label": r["label"], "geometry_type": r["geometry_type"]} for r in REFERENCE_CASES]}


@app.get("/api/v1/af3/references/{ref_id}")
async def get_reference(ref_id: str):
    from src.af3.references import get_reference as _get_ref
    ref = _get_ref(ref_id)
    if ref is None:
        raise HTTPException(status_code=404, detail=f"Reference '{ref_id}' not found.")
    return ref


class _CompareRequest(object):
    pass

from pydantic import BaseModel as _BM
from typing import List as _L

class CompareRequest(_BM):
    mean_tau_engineer:  _L[_L[float]]
    ref_id: str

@app.post("/api/v1/af3/compare")
async def compare_geometry(req: CompareRequest):
    import numpy as np
    from src.af3.references import get_reference as _get_ref, compare as _compare

    ref = _get_ref(req.ref_id)
    if ref is None:
        raise HTTPException(status_code=404, detail=f"Reference '{req.ref_id}' not found.")

    result = _compare(
        mean_tau_engineer  = np.array(req.mean_tau_engineer, dtype=np.float32),
        mean_tau_reference = np.array(ref["mean_tau"], dtype=np.float32),
    )
    from dataclasses import asdict
    return {**asdict(result), "reference_label": ref["label"]}


# ---------------------------------------------------------------------------
# AF4  — PDF Report download
# ---------------------------------------------------------------------------

class ReportRequest(_BM):
    geometry_type: str
    params:        dict
    metrics:       dict
    cf4_data:      dict = {}
    cf5_data:      dict = {}

@app.post("/api/v1/report")
async def download_report(req: ReportRequest):
    """Generate and return a multi-page PDF report for the current analysis."""
    from src.af4.report import generate_pdf

    loop = asyncio.get_event_loop()
    pdf_bytes = await loop.run_in_executor(
        None,
        generate_pdf,
        req.geometry_type,
        req.params,
        req.metrics,
        req.cf4_data or None,
        req.cf5_data or None,
    )
    filename = f"shear_{req.geometry_type}_{req.params}.pdf".replace(" ", "_")[:80]
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    from .checkpoint import _MODEL
    return {
        "status": "ok",
        "model_loaded": _MODEL is not None,
        "cache_size": len(get_cache()),
        "queue_size": get_queue()._queue.qsize(),
    }
