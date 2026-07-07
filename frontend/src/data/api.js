const API_BASE = process.env.REACT_APP_API_URL || '';

export async function submitInference(geometryType, params, options = {}) {
  const { nSamples = 5, odeSteps = 10, gridSize = 8, odeMethod = 'euler' } = options;
  const resp = await fetch(`${API_BASE}/api/v1/infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      geometry_type: geometryType,
      params,
      n_samples: nSamples,
      ode_steps: odeSteps,
      grid_size: gridSize,
      ode_method: odeMethod,
      include_samples: false,
    }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `Submit failed: ${resp.status}`);
  }
  return resp.json();
}

export async function pollJobResult(jobId) {
  const resp = await fetch(`${API_BASE}/api/v1/jobs/${jobId}`);
  if (resp.status === 202) return null;
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `Poll failed: ${resp.status}`);
  }
  return resp.json();
}

export function connectProgress(jobId, onProgress, onComplete, onError) {
  let pollTimer = null;
  let closed = false;

  function startPolling() {
    let dots = 0;
    pollTimer = setInterval(async () => {
      if (closed) { clearInterval(pollTimer); return; }
      dots++;
      onProgress({ status: 'running', message: `Processing${'.'.repeat((dots % 3) + 1)}` });
      try {
        const resp = await fetch(`${API_BASE}/api/v1/jobs/${jobId}`);
        if (resp.status === 202) return; // still running
        clearInterval(pollTimer);
        if (resp.ok) {
          const result = await resp.json();
          onComplete({ status: 'complete', result });
        } else {
          const err = await resp.json().catch(() => ({}));
          onError({ message: err.detail || `Job failed: ${resp.status}` });
        }
      } catch (_) { /* keep polling */ }
    }, 1500);
  }

  async function handleComplete(msg) {
    // WebSocket only carries progress metadata — fetch full result via REST
    try {
      const resp = await fetch(`${API_BASE}/api/v1/jobs/${jobId}`);
      if (resp.ok) {
        const result = await resp.json();
        onComplete({ ...msg, result });
      } else {
        onError({ message: `Result fetch failed: ${resp.status}` });
      }
    } catch (err) {
      onError({ message: err.message });
    }
  }

  const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const wsHost = API_BASE
    ? API_BASE.replace(/^https?/, wsProto)
    : `${wsProto}://${window.location.host}`;

  let ws;
  try {
    ws = new WebSocket(`${wsHost}/ws/progress/${jobId}`);
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      if (msg.status === 'complete') handleComplete(msg);
      else if (msg.status === 'error') onError(msg);
      else onProgress(msg);
    };
    ws.onerror = () => {
      // WebSocket not supported — fall back to REST polling
      ws = null;
      startPolling();
    };
  } catch (_) {
    startPolling();
  }

  return {
    close() {
      closed = true;
      clearInterval(pollTimer);
      if (ws && ws.readyState < 2) ws.close();
    },
  };
}

export async function fetchCF4(inferenceResult) {
  const resp = await fetch(`${API_BASE}/api/v1/cf4`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      mean_tau: inferenceResult.mean_tau,
      variance_tau: inferenceResult.variance_tau,
      grad_u: inferenceResult.grad_u,
    }),
  });
  if (!resp.ok) throw new Error(`CF4 failed: ${resp.status}`);
  return resp.json();
}

export async function runExplorer(geometryType, params, options = {}) {
  const { nSamples = 5, gridSize = 4, odeSteps = 10, odeMethod = 'euler' } = options;
  const resp = await fetch(`${API_BASE}/api/v1/explorer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      geometry_type: geometryType,
      params,
      n_samples: nSamples,
      grid_size: gridSize,
      ode_steps: odeSteps,
      ode_method: odeMethod,
    }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `Explorer failed: ${resp.status}`);
  }
  return resp.json();
}
