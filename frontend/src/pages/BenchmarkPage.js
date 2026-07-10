import { useState, useEffect, useRef } from 'react';

const API_BASE = process.env.REACT_APP_API_URL || '';
const MOCK_MODE = process.env.REACT_APP_MOCK_MODE === 'true' || !process.env.REACT_APP_API_URL;

// Mock reference catalogue
const MOCK_REFS = [
  { id: 'naca0012',       label: 'NACA 0012 (Re=1M, AoA=5°)',      geometry_type: 'aerofoil' },
  { id: 'bluff_body_std', label: 'Standard Bluff Body (Re=5×10⁴)', geometry_type: 'bluff_body' },
  { id: 'cylinder_re1000',label: 'Cylinder (Re=1000)',              geometry_type: 'cylinder' },
  { id: 'turbine_rated',  label: 'Wind Turbine (rated, TSR=7)',     geometry_type: 'turbine_blade' },
  { id: 'flat_plate_aoa10',label: 'Flat Plate (AoA=10°)',          geometry_type: 'flat_plate' },
  { id: 'ship_hull_design',label: 'Ship Hull (design draught)',     geometry_type: 'ship_hull' },
];

function strHash(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619) >>> 0;
  return h;
}

function mockComparison(engineerResult, refId) {
  const h = strHash(refId);
  const r       = 0.12 + (h % 71) / 100;
  const ratio   = 0.65 + (h % 53) / 100;
  const amp     = 0.02 + (h % 37) / 500;
  const freq    = 0.18 + (h % 19) / 50;
  const phase   = (h % 63) / 10;
  const freq2   = 0.05 + (h % 11) / 40;
  const N = 80;
  const diff = Array.from({ length: N }, (_, i) =>
    amp * Math.sin(i * freq + phase) + (amp * 0.4) * Math.sin(i * freq2 + phase * 1.3)
  );
  const refMag = 0.3 + (h % 41) / 100;
  const engMag = refMag * ratio;
  return {
    pearson_r:         parseFloat(r.toFixed(3)),
    peak_tau_ratio:    parseFloat(ratio.toFixed(3)),
    difference:        diff,
    engineer_metrics:  { peak_tau: engMag, mean_tau: engMag * 0.58 },
    reference_metrics: { peak_tau: refMag, mean_tau: refMag * 0.58 },
    reference_label:   MOCK_REFS.find(m => m.id === refId)?.label ?? refId,
  };
}

function MetricRow({ label, eng, ref, isRatio }) {
  const pct = isRatio
    ? `${((eng / (ref + 1e-12) - 1) * 100).toFixed(1)}%`
    : null;
  return (
    <tr className="border-t border-white/[0.05]">
      <td className="py-2.5 px-4 text-xs text-slate-500">{label}</td>
      <td className="py-2.5 px-4 text-xs font-mono text-teal-300">{Number(eng).toFixed(4)}</td>
      <td className="py-2.5 px-4 text-xs font-mono text-slate-300">{Number(ref).toFixed(4)}</td>
      {pct && (
        <td className={`py-2.5 px-4 text-xs font-mono font-semibold ${parseFloat(pct) > 0 ? 'text-red-400' : 'text-green-400'}`}>
          {parseFloat(pct) > 0 ? '+' : ''}{pct}
        </td>
      )}
    </tr>
  );
}

export default function BenchmarkPage({ engineerResult }) {
  const [references,  setReferences]  = useState([]);
  const [selectedRef, setSelectedRef] = useState(null);
  const [comparison,  setComparison]  = useState(null);
  const [loading,     setLoading]     = useState(false);
  const diffRef = useRef(null);

  useEffect(() => {
    if (MOCK_MODE) { setReferences(MOCK_REFS); return; }
    fetch(`${API_BASE}/api/v1/af3/references`)
      .then(r => r.json())
      .then(d => setReferences(d.references || []))
      .catch(console.error);
  }, []);

  useEffect(() => {
    if (!comparison || !window.Plotly || !diffRef.current) return;

    window.Plotly.react(diffRef.current, [{
      type: 'scatter',
      mode: 'lines',
      x: comparison.difference.map((_, i) => i),
      y: comparison.difference,
      line: { color: '#14b8a6', width: 1.5 },
      fill: 'tozeroy',
      fillcolor: 'rgba(20,184,166,0.12)',
    }], {
      paper_bgcolor: 'transparent',
      plot_bgcolor:  'transparent',
      margin: { l: 44, r: 12, t: 32, b: 36 },
      font:   { color: '#94a3b8', size: 10 },
      title:  { text: '|τ| difference (engineer − reference)', font: { size: 12, color: '#e2e8f0' }, x: 0 },
      xaxis:  { title: 'node index', gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
      yaxis:  { title: 'Δ|τ|', gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
    }, { responsive: true, displayModeBar: false });
  }, [comparison]);

  async function handleCompare(refId) {
    setSelectedRef(refId);
    setLoading(true);
    setComparison(null);

    if (MOCK_MODE || !engineerResult) {
      await new Promise(r => setTimeout(r, 400));
      setComparison(mockComparison(engineerResult, refId));
      setLoading(false);
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/api/v1/af3/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          mean_tau_engineer: engineerResult.mean_tau,
          ref_id: refId,
        }),
      });
      setComparison(await res.json());
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-full">
      {/* Left — reference selector */}
      <aside className="w-64 shrink-0 glass-dark flex flex-col">
        <div className="px-4 pt-4 pb-3 border-b border-white/[0.06]">
          <div className="text-sm font-bold font-display text-gradient">Reference Library</div>
          <div className="text-xs text-slate-600 mt-0.5">6 canonical cases</div>
        </div>
        <div className="flex-1 overflow-y-auto p-2 flex flex-col gap-1.5">
          {references.map(ref => (
            <button
              key={ref.id}
              onClick={() => handleCompare(ref.id)}
              disabled={loading}
              className={`w-full text-left px-3 py-2.5 rounded-xl text-xs transition-all ${
                selectedRef === ref.id
                  ? 'glass-card border-teal-500/40 text-teal-200'
                  : 'glass-card text-slate-400 hover:text-slate-200'
              } disabled:opacity-30 disabled:cursor-not-allowed`}
            >
              <div className="font-medium">{ref.label}</div>
              <div className="text-slate-600 mt-0.5 capitalize">{ref.geometry_type.replace(/_/g, ' ')}</div>
            </button>
          ))}
        </div>
        {!engineerResult && (
          <div className="px-3 pb-3 text-xs text-slate-600 border-t border-white/[0.04] pt-2.5">
            Run an analysis to compare against your own geometry.
          </div>
        )}
      </aside>

      {/* Right — comparison result */}
      <div className="flex-1 flex flex-col min-w-0 overflow-y-auto p-6 gap-5">
        {!comparison && !loading && (
          <div className="flex flex-col items-center justify-center h-full gap-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-indigo-500/20 to-teal-500/20 flex items-center justify-center border border-white/[0.07]">
              <span className="text-xl">≈</span>
            </div>
            <div className="text-center">
              <div className="text-slate-300 font-semibold font-display text-sm mb-1">Geometry Benchmarking</div>
              <div className="text-slate-600 text-xs">Select a reference case to compare your geometry's stress field.</div>
            </div>
          </div>
        )}
        {loading && (
          <div className="flex items-center justify-center h-full text-slate-500 text-xs">
            <div className="skeleton w-32 h-4 rounded" />
          </div>
        )}

        {comparison && (
          <>
            <div className="text-sm font-bold font-display text-slate-100">
              Comparing vs. <span className="text-gradient">{comparison.reference_label}</span>
            </div>

            {/* Key metrics table */}
            <div className="glass-card overflow-hidden">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-white/[0.06]">
                    <th className="py-2.5 px-4 text-xs font-semibold text-slate-500 text-left uppercase tracking-wider">Metric</th>
                    <th className="py-2.5 px-4 text-xs font-semibold text-teal-400 text-left">Your geometry</th>
                    <th className="py-2.5 px-4 text-xs font-semibold text-slate-500 text-left">Reference</th>
                    <th className="py-2.5 px-4 text-xs font-semibold text-slate-500 text-left">Δ</th>
                  </tr>
                </thead>
                <tbody>
                  <MetricRow label="Peak τ"  eng={comparison.engineer_metrics.peak_tau}  ref={comparison.reference_metrics.peak_tau}  isRatio />
                  <MetricRow label="Mean τ"  eng={comparison.engineer_metrics.mean_tau}  ref={comparison.reference_metrics.mean_tau}  isRatio />
                  <tr className="border-t border-white/[0.05]">
                    <td className="py-2.5 px-4 text-xs text-slate-500">Pearson r (stress fields)</td>
                    <td className="py-2.5 px-4 text-xs font-mono text-slate-300" colSpan={2}>{comparison.pearson_r}</td>
                    <td />
                  </tr>
                  <tr className="border-t border-white/[0.05]">
                    <td className="py-2.5 px-4 text-xs text-slate-500">Peak τ ratio (yours / ref)</td>
                    <td className="py-2.5 px-4 text-xs font-mono text-slate-300" colSpan={2}>{comparison.peak_tau_ratio}×</td>
                    <td />
                  </tr>
                </tbody>
              </table>
            </div>

            {/* Difference chart */}
            <div className="glass-card p-1">
              <div ref={diffRef} className="min-h-52" />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
