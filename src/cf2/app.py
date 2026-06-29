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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    GeometryRequest,
    InferenceResult,
    JobResponse,
    JobStatus,
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
