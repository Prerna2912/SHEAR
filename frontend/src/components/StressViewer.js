import { useEffect, useRef, useState } from 'react';
import { computeCF4 } from '../data/cf4Compute';

const OVERLAY_OPTIONS = [
  { key: 'mean_tau_mag',  label: '|τ| Magnitude'   },
  { key: 'q_criterion',   label: 'Q-criterion'      },
  { key: 'uncertainty',   label: 'Uncertainty'      },
  { key: 'dissipation',   label: 'SGS Dissipation'  },
  { key: 'regime',        label: 'Turbulence Regime' },
];

function buildColorArray(result, cf4, overlay) {
  const N = result.positions.length;
  switch (overlay) {
    case 'mean_tau_mag':
      return result.mean_tau.map(t => Math.sqrt(t.reduce((s, x) => s + x * x, 0)));
    case 'q_criterion':
      return cf4 ? cf4.q_values : new Array(N).fill(0);
    case 'uncertainty':
      return cf4 ? cf4.uncertainty : new Array(N).fill(0);
    case 'dissipation':
      return cf4 ? cf4.dissipation : new Array(N).fill(0);
    case 'regime':
      return cf4 ? cf4.regime_labels : new Array(N).fill(0);
    default:
      return new Array(N).fill(0);
  }
}

export default function StressViewer({ result }) {
  const containerRef = useRef(null);
  const [overlay, setOverlay] = useState('mean_tau_mag');
  const [cf4, setCf4] = useState(null);

  useEffect(() => {
    if (!result) return;
    const computed = computeCF4(result.mean_tau, result.variance_tau, result.grad_u);
    setCf4(computed);
  }, [result]);

  useEffect(() => {
    if (!result || !window.Plotly || !containerRef.current) return;

    const positions = result.positions;
    const xs = positions.map(p => p[0]);
    const ys = positions.map(p => p[1]);
    const zs = positions.map(p => p[2]);
    const colors = buildColorArray(result, cf4, overlay);

    const colorscale = overlay === 'regime'
      ? [[0, '#3b82f6'], [0.5, '#f59e0b'], [1, '#ef4444']]
      : 'Viridis';

    const trace = {
      type: 'scatter3d',
      mode: 'markers',
      x: xs, y: ys, z: zs,
      marker: {
        size: 3,
        color: colors,
        colorscale,
        showscale: true,
        colorbar: {
          thickness: 12,
          len: 0.6,
          title: { text: OVERLAY_OPTIONS.find(o => o.key === overlay)?.label ?? '', font: { color: '#94a3b8', size: 10 } },
          tickfont: { color: '#94a3b8', size: 9 },
        },
        opacity: 0.85,
      },
      hovertemplate:
        'x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>val: %{marker.color:.4f}<extra></extra>',
    };

    const layout = {
      paper_bgcolor: 'transparent',
      plot_bgcolor:  'transparent',
      margin: { l: 0, r: 0, t: 0, b: 0 },
      scene: {
        bgcolor: 'transparent',
        xaxis: { title: 'x (streamwise)', color: '#64748b', gridcolor: '#334155' },
        yaxis: { title: 'y (normal)',      color: '#64748b', gridcolor: '#334155' },
        zaxis: { title: 'z (spanwise)',    color: '#64748b', gridcolor: '#334155' },
        camera: { eye: { x: 1.5, y: 1.0, z: 0.8 } },
      },
      font: { color: '#94a3b8' },
    };

    const config = { responsive: true, displayModeBar: false };

    window.Plotly.react(containerRef.current, [trace], layout, config);
  }, [result, cf4, overlay]);

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-teal-500/15 to-sky-500/15 flex items-center justify-center border border-white/[0.08]">
          <span className="text-3xl">◉</span>
        </div>
        <div className="text-center">
          <div className="text-slate-300 font-semibold font-display mb-1 text-sm">3D Stress Field</div>
          <div className="text-slate-600 text-xs">Submit a geometry to visualise the SGS stress tensor field.</div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Overlay selector */}
      <div className="flex gap-1 p-2 border-b border-white/[0.05] flex-wrap bg-black/10">
        {OVERLAY_OPTIONS.map(opt => (
          <button
            key={opt.key}
            onClick={() => setOverlay(opt.key)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              overlay === opt.key
                ? 'bg-gradient-to-r from-teal-600/40 to-teal-500/30 text-teal-300 border border-teal-500/30'
                : 'btn-ghost'
            }`}
          >
            {opt.label}
          </button>
        ))}
        <span className="ml-auto text-xs text-slate-600 self-center font-mono">
          {result.n_nodes} nodes · {result.geometry_type}
        </span>
      </div>

      {/* 3D scatter */}
      <div ref={containerRef} className="flex-1 min-h-0" />
    </div>
  );
}
