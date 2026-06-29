import { useEffect, useRef, useState } from 'react';

// JHTDB training distribution percentile bounds (from jhtdb_stats.json)
const JHTDB_BOUNDS = {
  I1: { p5: -0.18, p25: -0.06, p50: 0.00, p75: 0.06, p95: 0.18 },
  I2: { p5: -0.045, p25: -0.012, p50: 0.001, p75: 0.014, p95: 0.048 },
  I3: { p5: -0.008, p25: -0.002, p50: 0.000, p75: 0.002, p95: 0.008 },
};

const REGIME_PEARSON = { low: 0.12, medium: 0.004, high: -0.06 };

function OODBanner({ ood }) {
  if (!ood || !ood.is_ood) return null;
  return (
    <div className="flex items-start gap-3 px-4 py-3 rounded-xl border border-amber-500/30 bg-amber-500/[0.07] pulse-warning">
      <span className="text-amber-400 text-lg shrink-0">⚠</span>
      <div>
        <div className="text-sm font-semibold text-amber-300">Out-of-Distribution Warning</div>
        <div className="text-xs text-amber-400/80 mt-0.5 leading-relaxed">
          This geometry's ∇u distribution is <strong>{ood.mahal_distance.toFixed(2)}σ</strong> from the JHTDB training set
          (threshold: {ood.threshold}σ). V1 predictions may be less reliable in this regime.
        </div>
      </div>
    </div>
  );
}

function RegimeAccuracyCard({ regime, r }) {
  const color = r > 0.1 ? '#22c55e' : r > 0 ? '#f59e0b' : '#ef4444';
  return (
    <div className="stat-card flex flex-col gap-1">
      <div className="text-xs text-slate-500 capitalize uppercase tracking-wider font-medium">{regime} vorticity</div>
      <div className="text-xl font-bold font-mono font-display" style={{ color }}>
        r = {r.toFixed(3)}
      </div>
      <div className="text-xs text-slate-600">Pearson r from Task 3</div>
    </div>
  );
}

function KDEChart({ plotRef, title }) {
  return (
    <div className="glass-card p-1">
      <div ref={plotRef} className="min-h-40" />
    </div>
  );
}

export default function AF1Panel({ result, af1Data }) {
  const ref_I1 = useRef(null);
  const ref_I2 = useRef(null);
  const ref_I3 = useRef(null);
  const [computed, setComputed] = useState(null);

  // Compute AF1 client-side if no backend data
  useEffect(() => {
    if (!result) return;
    if (af1Data) { setComputed(af1Data); return; }

    // Client-side approximate invariants from mean_tau irreps
    const tau = result.mean_tau;
    const inv_sqrt3 = 1 / Math.sqrt(3);
    const I1 = tau.map(t => t[0] * inv_sqrt3 * 3);  // tr(τ) ≈ 3×scalar
    const I2 = tau.map(t => {
      const iso = t[0] * inv_sqrt3;
      return iso * iso * 3 - iso * iso * 3 * 0.5;   // approx
    });
    const I3 = tau.map(t => Math.pow(t[0] * inv_sqrt3, 3));  // det approx (isotropic)

    // OOD: simple z-score on ∇u mean
    const gradMean = result.grad_u.reduce((acc, g) =>
      acc.map((v, i) => v + g[i] / result.grad_u.length),
      new Array(9).fill(0)
    );
    const gradStd = [0.42, 0.38, 0.38, 0.38, 0.42, 0.38, 0.38, 0.38, 0.42];
    const zScores = gradMean.map((v, i) => v / (gradStd[i] + 1e-8));
    const mahal = Math.sqrt(zScores.reduce((s, z) => s + z * z, 0));

    setComputed({
      I1, I2, I3,
      ood: { mahal_distance: mahal, threshold: 2.0, is_ood: mahal > 2.0, z_scores: zScores },
      regime_pearson_r: REGIME_PEARSON,
    });
  }, [result, af1Data]);

  // KDE plots
  useEffect(() => {
    if (!computed || !window.Plotly) return;

    const dark = (title, xLabel) => ({
      paper_bgcolor: 'transparent',
      plot_bgcolor:  'transparent',
      margin: { l: 36, r: 8, t: 28, b: 32 },
      font:   { color: '#94a3b8', size: 10 },
      title:  { text: title, font: { size: 11, color: '#e2e8f0' }, x: 0 },
      xaxis:  { title: xLabel, gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
      yaxis:  { title: 'density', gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
    });
    const cfg = { responsive: true, displayModeBar: false };

    const makeKDE = (values, bounds, color) => {
      const sorted = [...values].sort((a, b) => a - b);
      const n = sorted.length;
      const lo = sorted[0], hi = sorted[n - 1];
      const pad = (hi - lo) * 0.25 + 1e-8;
      const xs = Array.from({ length: 80 }, (_, i) => lo - pad + (i / 79) * (hi - lo + 2 * pad));

      // Gaussian KDE (Silverman bandwidth)
      const bw = 1.06 * Math.sqrt(values.reduce((s, v) => s + v * v, 0) / n - Math.pow(values.reduce((s, v) => s + v, 0) / n, 2) + 1e-12) * Math.pow(n, -0.2);
      const ys = xs.map(x => values.reduce((s, v) => s + Math.exp(-0.5 * ((x - v) / bw) ** 2), 0) / (n * bw * Math.sqrt(2 * Math.PI)));

      return [
        { type: 'scatter', x: xs, y: ys, mode: 'lines', name: 'V1', line: { color, width: 2 } },
        { type: 'scatter', x: [bounds.p5, bounds.p5, bounds.p95, bounds.p95, bounds.p5],
          y: [0, Math.max(...ys), Math.max(...ys), 0, 0],
          fill: 'toself', fillcolor: 'rgba(148,163,184,0.08)', mode: 'lines',
          line: { color: 'rgba(148,163,184,0.3)', width: 1, dash: 'dot' }, name: 'JHTDB p5–p95' },
        { type: 'scatter', x: [bounds.p25, bounds.p25, bounds.p75, bounds.p75, bounds.p25],
          y: [0, Math.max(...ys), Math.max(...ys), 0, 0],
          fill: 'toself', fillcolor: 'rgba(148,163,184,0.15)', mode: 'lines',
          line: { color: 'rgba(148,163,184,0.4)', width: 1 }, name: 'JHTDB IQR' },
      ];
    };

    if (ref_I1.current) window.Plotly.react(ref_I1.current, makeKDE(computed.I1, JHTDB_BOUNDS.I1, '#14b8a6'), dark('I₁ = tr(τ)', 'I₁'), cfg);
    if (ref_I2.current) window.Plotly.react(ref_I2.current, makeKDE(computed.I2, JHTDB_BOUNDS.I2, '#818cf8'), dark('I₂ = (tr(τ)²−tr(τ²))/2', 'I₂'), cfg);
    if (ref_I3.current) window.Plotly.react(ref_I3.current, makeKDE(computed.I3, JHTDB_BOUNDS.I3, '#f472b6'), dark('I₃ = det(τ)', 'I₃'), cfg);
  }, [computed]);

  if (!result) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-3">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-teal-500/20 to-purple-500/20 flex items-center justify-center border border-white/[0.07]">
          <span className="text-xl">⬡</span>
        </div>
        <div className="text-slate-500 text-xs">Run V1 analysis to see flow regime diagnostics.</div>
      </div>
    );
  }

  if (!computed) {
    return <div className="flex items-center justify-center h-full text-slate-500 text-xs">Computing diagnostics…</div>;
  }

  const pr = computed.regime_pearson_r || REGIME_PEARSON;

  return (
    <div className="flex flex-col gap-4 p-4 overflow-y-auto h-full">

      {/* OOD warning */}
      <OODBanner ood={computed.ood} />

      {/* Per-regime Pearson r */}
      <div>
        <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
          V1 Accuracy by Regime — Task 3 Evaluation
        </div>
        <div className="grid grid-cols-3 gap-3">
          <RegimeAccuracyCard regime="low"    r={pr.low    ?? REGIME_PEARSON.low}    />
          <RegimeAccuracyCard regime="medium" r={pr.medium ?? REGIME_PEARSON.medium} />
          <RegimeAccuracyCard regime="high"   r={pr.high   ?? REGIME_PEARSON.high}   />
        </div>
      </div>

      {/* Invariant KDE plots with JHTDB overlay */}
      <div>
        <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
          τ Invariant Distributions vs JHTDB Training Set
        </div>
        <div className="text-xs text-slate-500 mb-3">
          Shaded regions = JHTDB training distribution (dark = IQR, light = p5–p95).
          V1 curve outside these bounds signals distribution shift.
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <KDEChart plotRef={ref_I1} title="I₁ = tr(τ)" />
          <KDEChart plotRef={ref_I2} title="I₂ (2nd invariant)" />
          <KDEChart plotRef={ref_I3} title="I₃ = det(τ)" />
        </div>
      </div>

      {/* OOD detail */}
      {computed.ood && (
        <div className="glass-card p-3">
          <div className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wider">OOD Detection — Mahalanobis Distance</div>
          <div className="flex gap-5 text-xs flex-wrap">
            <span className="text-slate-500">Distance: <span className={`font-mono font-semibold ${computed.ood.is_ood ? 'text-amber-400' : 'text-green-400'}`}>{computed.ood.mahal_distance?.toFixed(3)}σ</span></span>
            <span className="text-slate-500">Threshold: <span className="font-mono text-slate-300">{computed.ood.threshold}σ</span></span>
            <span className="text-slate-500">Status: <span className={computed.ood.is_ood ? 'text-amber-400 font-semibold' : 'text-green-400 font-semibold'}>{computed.ood.is_ood ? 'OUT-OF-DISTRIBUTION' : 'In distribution'}</span></span>
          </div>
        </div>
      )}
    </div>
  );
}
