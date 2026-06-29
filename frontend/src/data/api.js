const API_BASE = process.env.REACT_APP_API_URL || '';

export async function submitInference(geometryType, params, options = {}) {
  const { nSamples = 20, odeSteps = 100, gridSize = 16 } = options;
  const resp = await fetch(`${API_BASE}/api/v1/infer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      geometry_type: geometryType,
      params,
      n_samples: nSamples,
      ode_steps: odeSteps,
      grid_size: gridSize,
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
  const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const wsHost = API_BASE
    ? API_BASE.replace(/^https?/, wsProto)
    : `${wsProto}://${window.location.host}`;
  const ws = new WebSocket(`${wsHost}/ws/progress/${jobId}`);
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.status === 'complete') onComplete(msg);
    else if (msg.status === 'error') onError(msg);
    else onProgress(msg);
  };
  ws.onerror = () => onError({ message: 'WebSocket connection failed' });
  return ws;
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
  const { nSamples = 10, gridSize = 16 } = options;
  const resp = await fetch(`${API_BASE}/api/v1/explorer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      geometry_type: geometryType,
      params,
      n_samples: nSamples,
      grid_size: gridSize,
    }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `Explorer failed: ${resp.status}`);
  }
  return resp.json();
}
