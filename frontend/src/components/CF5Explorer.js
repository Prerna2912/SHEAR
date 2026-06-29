import { useEffect, useRef, useState, useCallback, useMemo } from 'react';

function fmt(n, digits = 3) {
  if (n === null || n === undefined) return '—';
  return Math.abs(n) < 0.001 ? n.toExponential(2) : n.toFixed(digits);
}

// ── Parameter Animator ─────────────────────────────────────────────
function ParameterAnimator({ ranking, activeParam }) {
  const [frame,   setFrame]   = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed,   setSpeed]   = useState(600);
  const chartRef = useRef(null);
  const timerRef = useRef(null);

  const selected = ranking.find(r => r.param === activeParam);
  const levels   = useMemo(() => selected?.perturbations ?? [], [selected]);

  const stopPlay = useCallback(() => {
    clearInterval(timerRef.current);
    setPlaying(false);
  }, []);

  const togglePlay = useCallback(() => {
    if (playing) { stopPlay(); return; }
    if (!selected || levels.length === 0) return;
    setPlaying(true);
    timerRef.current = setInterval(() => {
      setFrame(f => {
        if (f >= levels.length - 1) { clearInterval(timerRef.current); setPlaying(false); return 0; }
        return f + 1;
      });
    }, speed);
  }, [playing, stopPlay, selected, levels, speed]);

  useEffect(() => () => clearInterval(timerRef.current), []);

  // Update chart
  useEffect(() => {
    if (!selected || !window.Plotly || !chartRef.current) return;

    const colors = levels.map((_, i) => i === frame ? '#14b8a6' : 'rgba(20,184,166,0.25)');
    const sizes  = levels.map((_, i) => i === frame ? 10 : 5);

    const xs = levels.map(l => `${(l.level * 100 > 0 ? '+' : '')}${(l.level * 100).toFixed(0)}%`);
    const ys = levels.map(l => l.peak_tau);

    window.Plotly.react(chartRef.current, [
      {
        type: 'scatter', mode: 'lines+markers',
        x: xs, y: ys,
        line: { color: 'rgba(20,184,166,0.3)', width: 1.5 },
        marker: { color: colors, size: sizes, symbol: 'circle' },
        name: 'peak τ',
      },
      {
        type: 'scatter', mode: 'markers',
        x: [xs[frame]], y: [ys[frame]],
        marker: { color: '#14b8a6', size: 14, symbol: 'circle', line: { color: '#5eead4', width: 2 } },
        showlegend: false,
      },
    ], {
      paper_bgcolor: 'transparent',
      plot_bgcolor:  'transparent',
      margin: { l: 44, r: 8, t: 36, b: 36 },
      font:   { color: '#94a3b8', size: 10 },
      title:  { text: `${selected.param} — stress response`, font: { size: 12, color: '#e2e8f0' }, x: 0 },
      xaxis:  { title: 'perturbation', gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
      yaxis:  { title: 'peak τ', gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
      annotations: [{
        x: xs[frame], y: ys[frame],
        text: `${fmt(ys[frame])}`,
        showarrow: true, arrowhead: 2, arrowcolor: '#14b8a6',
        font: { color: '#5eead4', size: 10 },
        bgcolor: 'rgba(20,184,166,0.15)',
        bordercolor: '#14b8a6', borderwidth: 1, borderpad: 3,
        ay: -28,
      }],
    }, { responsive: true, displayModeBar: false });
  }, [selected, frame, levels]);

  if (!selected) {
    return (
      <div className="glass-card p-4 text-center text-slate-500 text-xs">
        Click a parameter below to animate its stress response sweep.
      </div>
    );
  }

  const currLevel = levels[frame];

  return (
    <div className="glass-card overflow-hidden">
      {/* Chart */}
      <div ref={chartRef} className="min-h-48" />

      {/* Controls */}
      <div className="px-4 pb-4 flex items-center gap-3 flex-wrap">
        <button
          onClick={togglePlay}
          className={`px-4 py-1.5 rounded-lg text-xs font-semibold transition-all ${playing ? 'btn-ghost' : 'btn-primary'}`}
        >
          {playing ? '⏸ Pause' : '▶ Play'}
        </button>

        {/* Frame scrubber */}
        <input
          type="range" min={0} max={levels.length - 1} value={frame}
          onChange={e => { stopPlay(); setFrame(Number(e.target.value)); }}
          className="flex-1 min-w-24"
        />

        {/* Speed */}
        <select
          value={speed}
          onChange={e => setSpeed(Number(e.target.value))}
          className="input-glass px-2 py-1 text-xs"
        >
          <option value={1200}>0.5×</option>
          <option value={600}>1×</option>
          <option value={300}>2×</option>
          <option value={150}>4×</option>
        </select>

        {/* Current state */}
        {currLevel && (
          <div className="text-xs font-mono text-slate-400">
            <span className="text-teal-400">{(currLevel.level * 100 > 0 ? '+' : '')}{(currLevel.level * 100).toFixed(0)}%</span>
            {' → '}
            <span className="text-white">{fmt(currLevel.new_value)}</span>
            {' → peak τ '}
            <span className={currLevel.delta_peak_tau > 0 ? 'text-red-400' : 'text-green-400'}>
              {currLevel.delta_peak_tau > 0 ? '+' : ''}{(currLevel.delta_peak_tau * 100).toFixed(1)}%
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

// ── Sensitivity bar ────────────────────────────────────────────────
function SensBar({ r, basePeak, onSelect, isSelected }) {
  const pct = Math.min(100, (r.max_abs_delta / (basePeak || 1)) * 300);
  return (
    <button
      onClick={onSelect}
      className={`w-full text-left px-3 py-2.5 rounded-lg transition-all ${isSelected ? 'glass-card border-teal-500/40' : 'glass-card'}`}
    >
      <div className="flex items-center gap-2 mb-1.5">
        <span className="text-xs text-slate-300 font-mono truncate w-32">{r.param}</span>
        <div className="flex-1 h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
          <div className="h-full bg-gradient-to-r from-teal-500 to-teal-400 rounded-full" style={{ width: `${pct}%` }} />
        </div>
        <span className="text-xs font-mono text-slate-300 shrink-0">±{(r.max_abs_delta * 100).toFixed(1)}%</span>
      </div>
      <div className="text-xs text-slate-500">
        base = <span className="font-mono text-slate-400">{fmt(r.base_value)}</span>
        {' · '}click to animate
      </div>
    </button>
  );
}

// ── Main export ────────────────────────────────────────────────────
export default function CF5Explorer({ explorerResult }) {
  const interactRef = useRef(null);
  const [selectedParam, setSelectedParam] = useState(null);

  useEffect(() => {
    if (!explorerResult || !window.Plotly) return;
    const { parameter_ranking: ranking, interaction_map: imap } = explorerResult;

    if (!selectedParam && ranking.length) setSelectedParam(ranking[0].param);

    if (interactRef.current && imap) {
      const pct = v => `${(v * 100).toFixed(0)}%`;
      window.Plotly.react(interactRef.current, [{
        type: 'heatmap',
        x: imap.levels.map(pct),
        y: imap.levels.map(pct),
        z: imap.grid,
        colorscale: [[0,'#3b82f6'],[0.5,'rgba(255,255,255,0.1)'],[1,'#ef4444']],
        showscale: true,
        colorbar: { thickness: 10, title: { text: 'Δτ', font: { color: '#64748b', size: 9 } }, tickfont: { color: '#64748b', size: 9 } },
      }], {
        paper_bgcolor: 'transparent',
        plot_bgcolor:  'transparent',
        margin: { l: 56, r: 20, t: 36, b: 48 },
        font:   { color: '#94a3b8', size: 10 },
        title:  { text: `${imap.param_a} × ${imap.param_b}`, font: { size: 12, color: '#e2e8f0' }, x: 0 },
        xaxis:  { title: imap.param_a, gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
        yaxis:  { title: imap.param_b, gridcolor: 'rgba(255,255,255,0.06)', zerolinecolor: 'rgba(255,255,255,0.1)' },
      }, { responsive: true, displayModeBar: false });
    }
  }, [explorerResult, selectedParam]);

  if (!explorerResult) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-teal-500/20 to-indigo-500/20 flex items-center justify-center border border-white/[0.08]">
          <span className="text-2xl">⚡</span>
        </div>
        <div className="text-center">
          <div className="text-slate-300 font-semibold mb-1">Parametric Explorer</div>
          <div className="text-slate-500 text-xs">Click "Run Parametric Explorer" to analyse how each parameter affects turbulent stress.</div>
        </div>
      </div>
    );
  }

  const { parameter_ranking: ranking, base_peak_tau, n_inference_calls, total_time_ms, interaction_map: imap } = explorerResult;

  return (
    <div className="flex flex-col gap-4 p-4 overflow-y-auto h-full">
      {/* Meta */}
      <div className="flex gap-4 text-xs text-slate-500">
        <span>Base peak τ: <span className="text-slate-300 font-mono">{fmt(base_peak_tau)}</span></span>
        <span>{n_inference_calls} inference calls · {(total_time_ms / 1000).toFixed(1)}s</span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Left: sensitivity ranking + animator */}
        <div className="flex flex-col gap-3">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Sensitivity Ranking — click to animate
          </div>
          {ranking.map(r => (
            <SensBar
              key={r.param}
              r={r}
              basePeak={base_peak_tau}
              isSelected={selectedParam === r.param}
              onSelect={() => setSelectedParam(r.param)}
            />
          ))}

          <ParameterAnimator
            ranking={ranking}
            activeParam={selectedParam}
          />
        </div>

        {/* Right: interaction heatmap */}
        <div className="flex flex-col gap-3">
          {imap ? (
            <>
              <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                2D Interaction Map
              </div>
              <div className="glass-card p-1">
                <div ref={interactRef} className="min-h-64" />
              </div>
              <div className="glass-card p-3 text-xs text-slate-400 leading-relaxed">
                <span className="text-slate-300 font-semibold">Interaction insight: </span>
                Changing <span className="font-mono text-teal-400">{imap.param_a}</span> and{' '}
                <span className="font-mono text-teal-400">{imap.param_b}</span> jointly.
                Red = stress above baseline · Blue = below baseline.
                Off-diagonal asymmetry indicates parameter interaction — the two parameters do not act independently.
              </div>
            </>
          ) : (
            <div className="glass-card p-6 text-center text-slate-500 text-xs">
              Interaction map requires ≥2 numeric parameters.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
