import { useEffect, useRef } from 'react';
import { computeCF4 } from '../data/cf4Compute';

function StatCard({ label, value, sub }) {
  return (
    <div className="stat-card">
      <div className="text-xs text-slate-500 mb-1.5 uppercase tracking-wider font-medium">{label}</div>
      <div className="text-xl font-bold text-slate-100 font-display">{value}</div>
      {sub && <div className="text-xs text-slate-600 mt-0.5">{sub}</div>}
    </div>
  );
}

function fmt(n, digits = 3) {
  if (n === null || n === undefined) return '—';
  return Math.abs(n) < 0.001 ? n.toExponential(2) : n.toFixed(digits);
}

export default function CF4Panel({ result }) {
  const histRef    = useRef(null);
  const regimeRef  = useRef(null);
  const uncRef     = useRef(null);

  useEffect(() => {
    if (!result || !window.Plotly) return;
    const cf4 = computeCF4(result.mean_tau, result.variance_tau, result.grad_u);

    const darkLayout = (title) => ({
      paper_bgcolor: 'transparent',
      plot_bgcolor:  'transparent',
      margin: { l: 40, r: 10, t: 28, b: 36 },
      font:   { color: '#94a3b8', size: 11 },
      title:  { text: title, font: { size: 12, color: '#e2e8f0' }, x: 0 },
      xaxis:  { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
      yaxis:  { gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
    });

    const cfg = { responsive: true, displayModeBar: false };

    // Q-criterion histogram
    if (histRef.current) {
      window.Plotly.react(histRef.current, [{
        type: 'histogram',
        x: cf4.q_values,
        nbinsx: 30,
        marker: { color: '#14b8a6', opacity: 0.85 },
        name: 'Q',
      }], {
        ...darkLayout('Q-criterion distribution'),
        shapes: [
          { type: 'line', x0: cf4.regime_thresholds[0], x1: cf4.regime_thresholds[0],
            y0: 0, y1: 1, yref: 'paper', line: { color: '#f59e0b', dash: 'dot', width: 1 } },
          { type: 'line', x0: cf4.regime_thresholds[1], x1: cf4.regime_thresholds[1],
            y0: 0, y1: 1, yref: 'paper', line: { color: '#ef4444', dash: 'dot', width: 1 } },
        ],
      }, cfg);
    }

    // Regime bar chart
    if (regimeRef.current) {
      const counts = [0, 1, 2].map(r => cf4.regime_labels.filter(l => l === r).length);
      window.Plotly.react(regimeRef.current, [{
        type: 'bar',
        x: ['Straining', 'Mixed', 'Vortical'],
        y: counts,
        marker: { color: ['#3b82f6', '#f59e0b', '#ef4444'] },
      }], darkLayout('Turbulence regime breakdown'), cfg);
    }

    // Uncertainty histogram
    if (uncRef.current) {
      const highFrac = cf4.high_unc_mask.filter(Boolean).length / cf4.uncertainty.length;
      window.Plotly.react(uncRef.current, [{
        type: 'histogram',
        x: cf4.uncertainty,
        nbinsx: 25,
        marker: { color: '#818cf8', opacity: 0.85 },
        name: 'unc',
      }], {
        ...darkLayout(`Uncertainty (${(highFrac * 100).toFixed(1)}% high)`),
      }, cfg);
    }
  }, [result]);

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500/20 to-teal-500/20 flex items-center justify-center border border-white/[0.07]">
          <span className="text-xl">⚡</span>
        </div>
        <div className="text-slate-500 text-xs">Run V1 analysis to see flow diagnostics.</div>
      </div>
    );
  }

  const cf4 = computeCF4(result.mean_tau, result.variance_tau, result.grad_u);
  const meanDiss      = cf4.dissipation.reduce((s, x) => s + x, 0) / cf4.dissipation.length;
  const highUncFrac   = cf4.high_unc_mask.filter(Boolean).length / cf4.high_unc_mask.length;
  const meanQ         = cf4.q_values.reduce((s, x) => s + x, 0) / cf4.q_values.length;

  return (
    <div className="flex flex-col gap-4 p-4 overflow-y-auto h-full">
      {/* Stats row */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard label="Mean Q-criterion"   value={fmt(meanQ)}                  sub="vortex dominance" />
        <StatCard label="Mean SGS Dissip."   value={fmt(meanDiss)}               sub="Π = −τ:S approx" />
        <StatCard label="Backscatter"        value={`${(cf4.backscatter_frac * 100).toFixed(1)}%`} sub="nodes with Π < 0" />
        <StatCard label="High Uncertainty"   value={`${(highUncFrac * 100).toFixed(1)}%`}          sub="top-25% variance" />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3 flex-1 min-h-0">
        <div className="glass-card p-1 min-h-48"><div ref={histRef}   className="h-full" /></div>
        <div className="glass-card p-1 min-h-48"><div ref={regimeRef} className="h-full" /></div>
        <div className="glass-card p-1 min-h-48"><div ref={uncRef}    className="h-full" /></div>
      </div>
    </div>
  );
}
