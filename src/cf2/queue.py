"""
CF2 — Asyncio GPU job queue.

Serialises concurrent V1 inference requests onto a single GPU worker so
the GPU is never shared mid-ODE-solve.  Each submitted job gets a UUID and
is tracked in an in-process store keyed by job_id.

Architecture
------------
  asyncio.Queue  →  _gpu_worker() coroutine  →  run_inference()
  WebSocket subscribers broadcast per-sample progress via _progress_subscribers.

Usage (from FastAPI startup):
    queue = get_queue()
    asyncio.create_task(queue.start_worker())

Usage (from endpoint):
    job = await queue.submit(request)
    return JobResponse(job_id=job.job_id, ...)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from .schemas import InferenceResult, JobStatus, ProgressMessage


# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------

@dataclass
class Job:
    job_id: str
    geometry_type: str
    params: dict
    geometry_hash: str
    n_samples: int
    ode_steps: int
    grid_size: int
    include_samples: bool
    status: JobStatus = JobStatus.QUEUED
    result: Optional[InferenceResult] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None


# ---------------------------------------------------------------------------
# Queue + worker
# ---------------------------------------------------------------------------

class GPUJobQueue:
    """
    Single-GPU asyncio job queue.

    The worker coroutine processes one job at a time.  Progress events are
    broadcast to any WebSocket connections that have subscribed to a job_id.
    """

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._jobs: Dict[str, Job] = {}
        # job_id → set of asyncio.Queue subscribers for progress
        self._progress_subs: Dict[str, Set[asyncio.Queue]] = {}

    # --- Public API -----------------------------------------------------------

    async def submit(self, job: Job) -> Job:
        """Enqueue a job and return it (with status=QUEUED)."""
        self._jobs[job.job_id] = job
        self._progress_subs[job.job_id] = set()
        await self._queue.put(job)
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def subscribe_progress(self, job_id: str) -> asyncio.Queue:
        """
        Return an asyncio.Queue that will receive ProgressMessage dicts.
        The subscriber should await messages until status==complete|error.
        """
        q: asyncio.Queue = asyncio.Queue()
        if job_id in self._progress_subs:
            self._progress_subs[job_id].add(q)
        return q

    def unsubscribe_progress(self, job_id: str, q: asyncio.Queue) -> None:
        if job_id in self._progress_subs:
            self._progress_subs[job_id].discard(q)

    # --- Worker ---------------------------------------------------------------

    async def start_worker(self):
        """Long-running coroutine.  Create via asyncio.create_task() at startup."""
        while True:
            job: Job = await self._queue.get()
            await self._process_job(job)
            self._queue.task_done()

    async def _process_job(self, job: Job) -> None:
        """Run inference for one job in the executor (non-blocking)."""
        loop = asyncio.get_event_loop()
        job.status = JobStatus.RUNNING
        await self._broadcast_progress(job, 0)

        try:
            # Run blocking GPU work in a thread-pool executor
            result = await loop.run_in_executor(
                None,
                _blocking_inference,
                job,
                self._make_progress_callback(job, loop),
            )
            job.result = result
            job.status = JobStatus.COMPLETE
        except Exception as exc:
            job.error = str(exc)
            job.status = JobStatus.ERROR
        finally:
            job.finished_at = time.time()
            await self._broadcast_progress(job, job.n_samples)
            # Remove subscribers after job finishes
            self._progress_subs.pop(job.job_id, None)

    def _make_progress_callback(self, job: Job, loop: asyncio.AbstractEventLoop):
        def callback(completed: int, total: int):
            asyncio.run_coroutine_threadsafe(
                self._broadcast_progress(job, completed),
                loop,
            )
        return callback

    async def _broadcast_progress(self, job: Job, completed: int) -> None:
        msg = ProgressMessage(
            job_id=job.job_id,
            sample=completed,
            n_samples=job.n_samples,
            status=job.status,
            message=(
                f"Sample {completed} / {job.n_samples} complete"
                if completed < job.n_samples
                else "Inference complete"
            ),
        )
        for q in list(self._progress_subs.get(job.job_id, [])):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass


# ---------------------------------------------------------------------------
# Blocking inference runner (called from executor)
# ---------------------------------------------------------------------------

def _blocking_inference(job: Job, progress_fn: Callable) -> InferenceResult:
    """
    Synchronous inference called from the thread executor.
    Imports are deferred so they only load when actually needed.
    """
    import time
    t0 = time.time()

    from .checkpoint import get_model
    from .inference import run_inference, geometry_hash as _hash
    from .cache import get_cache
    from .schemas import InferenceResult, JobStatus
    from src.cf1 import solve as cf1_solve

    cache = get_cache()

    # Check cache first
    cached = cache.get(job.geometry_hash)
    if cached is not None:
        # Re-use cached result but give it this job's job_id
        return cached.model_copy(update={"job_id": job.job_id})

    # Run CF1 to get the ∇u graph
    cf1_out = cf1_solve(job.geometry_type, job.params, grid_size=job.grid_size)

    # Run V1 inference
    model, stats, device = get_model()
    raw = run_inference(
        graph=cf1_out.graph,
        model=model,
        stats=stats,
        device=device,
        n_samples=job.n_samples,
        ode_steps=job.ode_steps,
        progress_fn=progress_fn,
        include_samples=job.include_samples,
    )

    elapsed_ms = int((time.time() - t0) * 1000)

    # Compute strain rate from CF1 grad_u (S_ij = (∂u_i/∂x_j + ∂u_j/∂u_i)/2)
    import numpy as np
    g = cf1_out.grad_u.reshape(-1, 3, 3)
    strain_rate = 0.5 * (g + g.transpose(0, 2, 1))   # [N, 3, 3]

    result = InferenceResult(
        mean_tau=raw["mean_tau"].tolist(),
        variance_tau=raw["variance_tau"].tolist(),
        samples=(raw["samples"].tolist() if raw["samples"] is not None else None),
        positions=cf1_out.positions.tolist(),
        grad_u=cf1_out.grad_u.reshape(-1, 9).tolist(),
        grid_shape=list(cf1_out.grid_shape),
        geometry_hash=job.geometry_hash,
        geometry_type=job.geometry_type,
        params=job.params,
        n_samples=job.n_samples,
        n_nodes=raw["n_nodes"],
        ode_steps=job.ode_steps,
        inference_time_ms=elapsed_ms,
        job_id=job.job_id,
        status=JobStatus.COMPLETE,
    )

    cache.put(job.geometry_hash, result)
    return result


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_QUEUE: Optional[GPUJobQueue] = None


def get_queue() -> GPUJobQueue:
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = GPUJobQueue()
    return _QUEUE


def make_job(
    geometry_type: str,
    params: dict,
    n_samples: int = 20,
    ode_steps: int = 100,
    grid_size: int = 16,
    include_samples: bool = False,
) -> Job:
    from .inference import geometry_hash
    ghash = geometry_hash(geometry_type, params)
    return Job(
        job_id=str(uuid.uuid4()),
        geometry_type=geometry_type,
        params=params,
        geometry_hash=ghash,
        n_samples=n_samples,
        ode_steps=ode_steps,
        grid_size=grid_size,
        include_samples=include_samples,
    )
