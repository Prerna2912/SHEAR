import { useState } from 'react';

const GEOMETRY_CONFIGS = {
  aerofoil: {
    label: 'Aerofoil',
    desc: 'Lifting surface — aircraft wings, control surfaces, and hydrofoils.',
    params: [
      {
        key: 'chord', label: 'Chord', unit: 'm', default: 1.0, min: 0.01, max: 20, step: 0.05,
        hint: 'Front-to-back depth of the wing section. Model aircraft ≈0.3m · light GA ≈1–2m · airliner ≈5–8m',
      },
      {
        key: 'span', label: 'Span', unit: 'm', default: 5.0, min: 0.1, max: 100, step: 0.5,
        hint: 'Tip-to-tip wing length. Model plane ≈1–3m · GA aircraft ≈10–15m · airliner ≈60m',
      },
      {
        key: 'alpha_deg', label: 'Angle of Attack', unit: '°', default: 5.0, min: -30, max: 30, step: 0.5,
        hint: 'Nose-up tilt relative to incoming flow. Cruise ≈2–5° · max lift ≈15° · stall begins >18°',
      },
      {
        key: 'U_inf', label: 'Freestream Velocity', unit: 'm/s', default: 50.0, min: 0.1, max: 300, step: 1,
        hint: 'Airspeed. Walking ≈1.5 m/s · highway car ≈30 m/s · airliner cruise ≈250 m/s',
      },
      {
        key: 'Re', label: 'Reynolds Number', unit: '', default: 3e6, min: 1e3, max: 1e9, step: 1e5,
        noSlider: true,
        hint: 'Re = U × chord / ν. Below 5×10⁵ = laminar · above 5×10⁶ = fully turbulent · typical aircraft: 1M–50M',
      },
      {
        key: 'naca_code', label: 'NACA Code', unit: '', default: '2412', type: 'text',
        hint: '4-digit code: camber% · camber-position×10 · thickness%. 2412 = 2% camber at 40% chord, 12% thick. Symmetric: 0012',
      },
    ],
  },
  swept_wing: {
    label: 'Swept Wing',
    desc: 'Swept lifting surface — transonic and supersonic aircraft.',
    params: [
      {
        key: 'chord', label: 'Root Chord', unit: 'm', default: 2.0, min: 0.01, max: 20, step: 0.1,
        hint: 'Chord at wing root. Subsonic GA ≈1–2m · commercial jet ≈4–8m',
      },
      {
        key: 'span', label: 'Span', unit: 'm', default: 10.0, min: 0.1, max: 100, step: 0.5,
        hint: 'Full tip-to-tip span. Narrow-body jet ≈35m · wide-body ≈60m',
      },
      {
        key: 'sweep_angle', label: 'Sweep Angle', unit: '°', default: 30.0, min: -10, max: 70, step: 1,
        hint: 'Leading-edge sweep angle. GA ≈0–10° · transonic jet ≈25–45° · delta wing ≈55–65°',
      },
      {
        key: 'dihedral', label: 'Dihedral', unit: '°', default: 5.0, min: -15, max: 15, step: 0.5,
        hint: 'Wing upward tilt. Positive = roll stability. Commercial jets ≈3–7° · fighters ≈−5 to +3°',
      },
      {
        key: 'alpha_deg', label: 'Angle of Attack', unit: '°', default: 3.0, min: -30, max: 30, step: 0.5,
        hint: 'Cruise ≈2–4° for swept wings. Swept wings have higher stall margin than straight',
      },
      {
        key: 'U_inf', label: 'Freestream Velocity', unit: 'm/s', default: 60.0, min: 0.1, max: 300, step: 1,
        hint: 'Cruise speed for swept-wing jets ≈100–250 m/s',
      },
      {
        key: 'Re', label: 'Reynolds Number', unit: '', default: 5e6, min: 1e3, max: 1e9, step: 1e5,
        noSlider: true,
        hint: 'Re = U × chord / ν. Swept-wing jets typically 5M–50M',
      },
      {
        key: 'naca_code', label: 'NACA Code', unit: '', default: '2412', type: 'text',
        hint: 'Common swept-wing profiles: 0012 (symmetric) · 2412 · 64-210',
      },
    ],
  },
  bluff_body: {
    label: 'Bluff Body / Vehicle',
    desc: 'Non-lifting bluff body — automotive, buildings, and industrial structures.',
    params: [
      {
        key: 'length', label: 'Body Length', unit: 'm', default: 4.5, min: 0.1, max: 100, step: 0.1,
        hint: 'Streamwise body dimension. Compact car ≈4m · SUV ≈4.7m · truck ≈8m · building ≈10–100m',
      },
      {
        key: 'height', label: 'Body Height', unit: 'm', default: 1.5, min: 0.1, max: 20, step: 0.1,
        hint: 'Cross-flow height. Car ≈1.4–1.6m · van ≈2m · truck ≈3.5m · building up to 20m',
      },
      {
        key: 'flow_speed', label: 'Flow Speed', unit: 'm/s', default: 30.0, min: 0.1, max: 150, step: 1,
        hint: 'Incident flow velocity. Urban ≈10 m/s (36 km/h) · highway ≈30 m/s · storm ≈50 m/s',
      },
    ],
  },
  cylinder: {
    label: 'Cylinder / Pipe',
    desc: 'Circular cross-section — structural cables, pipelines, and bridge piers.',
    params: [
      {
        key: 'diameter', label: 'Diameter', unit: 'm', default: 0.1, min: 0.001, max: 10, step: 0.01,
        hint: 'Cross-section diameter. Bridge cable ≈0.05–0.3m · chimney ≈1–5m · bridge pier ≈3–10m',
      },
      {
        key: 'length', label: 'Length', unit: 'm', default: 1.0, min: 0.01, max: 1000, step: 0.1,
        hint: 'Axial cylinder length. End effects matter when length/diameter < 20',
      },
      {
        key: 'flow_velocity', label: 'Flow Velocity', unit: 'm/s', default: 1.0, min: 0.01, max: 100, step: 0.1,
        hint: 'River current ≈0.5–3 m/s · tidal flow ≈1–4 m/s · wind on structure ≈5–40 m/s',
      },
    ],
  },
  turbine_blade: {
    label: 'Wind Turbine Blade',
    desc: 'Rotating aerofoil — horizontal-axis wind turbine blades.',
    params: [
      {
        key: 'radius', label: 'Blade Radius', unit: 'm', default: 40.0, min: 1, max: 150, step: 1,
        hint: 'Full blade length from hub. Small turbine ≈10m · onshore utility ≈40–60m · offshore ≈80–100m',
      },
      {
        key: 'pitch_angle', label: 'Pitch Angle', unit: '°', default: 5.0, min: -15, max: 30, step: 0.5,
        hint: '0° = max power extraction. Increase toward 90° to reduce load or shut down. Typical operation: 0–15°',
      },
      {
        key: 'tip_speed_ratio', label: 'Tip Speed Ratio', unit: '', default: 7.0, min: 1, max: 20, step: 0.5,
        hint: 'Blade tip speed ÷ wind speed. Optimal for 3-blade turbines: 6–9. Below 4 = stall · above 10 = excess drag',
      },
      {
        key: 'U_inf', label: 'Wind Speed', unit: 'm/s', default: 10.0, min: 3, max: 25, step: 0.5,
        hint: 'Cut-in ≈3 m/s · rated power ≈10–15 m/s · cut-out ≈25 m/s (turbine stops above this)',
      },
    ],
  },
  flat_plate: {
    label: 'Flat Plate',
    desc: 'Zero-thickness plate — panels, structural surfaces, and simple test cases.',
    params: [
      {
        key: 'length', label: 'Length', unit: 'm', default: 1.0, min: 0.01, max: 50, step: 0.05,
        hint: 'Streamwise plate dimension. Lab experiments ≈0.1–1m · large structural panels ≈5–20m',
      },
      {
        key: 'width', label: 'Width', unit: 'm', default: 0.5, min: 0.01, max: 50, step: 0.05,
        hint: 'Spanwise plate dimension. High aspect ratio (width/length > 5) behaves like infinite plate',
      },
      {
        key: 'alpha_deg', label: 'Angle of Attack', unit: '°', default: 10.0, min: -20, max: 20, step: 0.5,
        hint: '0° = aligned with flow (thin BL only) · 10–20° = significant separation · 90° = broadside, max drag',
      },
    ],
  },
  ship_hull: {
    label: 'Ship Hull',
    desc: 'Marine displacement hull — yachts to container ships.',
    params: [
      {
        key: 'length', label: 'Hull Length', unit: 'm', default: 100.0, min: 1, max: 500, step: 1,
        hint: 'Waterline length. Sailing yacht ≈8–15m · patrol boat ≈40m · container ship ≈300–400m',
      },
      {
        key: 'beam', label: 'Beam', unit: 'm', default: 15.0, min: 0.5, max: 100, step: 0.5,
        hint: 'Hull width at waterline. Yacht ≈3–4m · ferry ≈20–30m · supertanker ≈60m',
      },
      {
        key: 'draft', label: 'Draft', unit: 'm', default: 5.0, min: 0.1, max: 30, step: 0.1,
        hint: 'Depth below waterline. Yacht ≈1–2m · container ship loaded ≈14m · supertanker ≈20m',
      },
      {
        key: 'speed', label: 'Speed', unit: 'm/s', default: 7.7, min: 0.1, max: 30, step: 0.1,
        hint: 'Hull speed. Displacement yacht ≈3–5 m/s (6–10 kn) · cargo ≈8 m/s (16 kn) · fast ferry ≈15 m/s',
      },
    ],
  },
  bluff_body_wake: {
    label: 'Bluff Body + Wake',
    desc: 'Bluff body with explicit downstream wake region modelling.',
    params: [
      {
        key: 'length', label: 'Body Length', unit: 'm', default: 4.5, min: 0.1, max: 100, step: 0.1,
        hint: 'Streamwise body dimension. Car ≈4.5m · van ≈5m · truck ≈8m',
      },
      {
        key: 'height', label: 'Body Height', unit: 'm', default: 1.5, min: 0.1, max: 20, step: 0.1,
        hint: 'Cross-flow height. Car ≈1.5m · SUV ≈1.8m · truck ≈3.5m',
      },
      {
        key: 'flow_speed', label: 'Flow Speed', unit: 'm/s', default: 30.0, min: 0.1, max: 150, step: 1,
        hint: 'Highway ≈30 m/s (108 km/h) · urban ≈10 m/s · storm ≈50 m/s',
      },
      {
        key: 'wake_length', label: 'Wake Length', unit: 'm', default: 20.0, min: 0.5, max: 200, step: 0.5,
        hint: 'How far downstream to model the wake. Typically 3–10× body length. Car at 4.5m → use 15–45m',
      },
    ],
  },
};

function defaultParams(geometryType) {
  const cfg = GEOMETRY_CONFIGS[geometryType];
  const out = {};
  cfg.params.forEach(p => { out[p.key] = p.default; });
  return out;
}

export { GEOMETRY_CONFIGS, defaultParams };

function formatNum(v) {
  if (Math.abs(v) >= 1e6) return (v / 1e6).toPrecision(3) + 'M';
  if (Math.abs(v) >= 1e3) return (v / 1e3).toPrecision(3) + 'k';
  return String(v);
}

export default function GeometryForm({ onSubmit, onExplore, isRunning, isExploring, gtype, params, onGtypeChange, onParamsChange }) {
  const [expandedHints, setExpandedHints] = useState({});

  const cfg = GEOMETRY_CONFIGS[gtype];

  function handleGeometryChange(e) {
    const next = e.target.value;
    onGtypeChange(next);
    onParamsChange(defaultParams(next));
  }

  function handleParam(key, value) {
    onParamsChange(prev => ({ ...prev, [key]: value }));
  }

  function handleParamBlur(key, value, p) {
    // Clamp to valid range on blur
    const num = parseFloat(value);
    if (!isNaN(num)) {
      const clamped = Math.min(p.max, Math.max(p.min, num));
      if (clamped !== num) handleParam(key, clamped);
    }
  }

  function toggleHint(key) {
    setExpandedHints(prev => ({ ...prev, [key]: !prev[key] }));
  }

  function sanitizeParams() {
    const cfg = GEOMETRY_CONFIGS[gtype];
    const sanitized = {};
    cfg.params.forEach(p => {
      const v = params[p.key];
      if (p.type === 'text') {
        sanitized[p.key] = v ?? p.default;
      } else {
        const num = parseFloat(v);
        sanitized[p.key] = isNaN(num) ? p.default : num;
      }
    });
    return sanitized;
  }

  function handleSubmit(e) {
    e.preventDefault();
    onSubmit(gtype, sanitizeParams());
  }

  function handleExplore(e) {
    e.preventDefault();
    onExplore(gtype, sanitizeParams());
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">

      {/* Geometry type selector */}
      <div>
        <label className="block text-xs font-semibold text-slate-500 uppercase tracking-widest mb-2">
          Geometry Type
        </label>
        <select
          value={gtype}
          onChange={handleGeometryChange}
          className="input-glass w-full px-3 py-2.5 text-sm rounded-xl"
        >
          {Object.entries(GEOMETRY_CONFIGS).map(([key, { label }]) => (
            <option key={key} value={key} style={{ background: '#0a0d14' }}>{label}</option>
          ))}
        </select>
        <p className="mt-1.5 text-xs text-slate-600 leading-relaxed">{cfg.desc}</p>
      </div>

      {/* Parameters */}
      <div className="flex flex-col gap-4">
        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Parameters</span>
        {cfg.params.map(p => {
          const val = params[p.key] ?? p.default;
          const numVal = parseFloat(val);
          const outOfRange = !isNaN(numVal) && p.type !== 'text' && (numVal < p.min || numVal > p.max);
          const hintOpen = expandedHints[p.key];

          return (
            <div key={p.key} className="flex flex-col gap-1">
              {/* Label row */}
              <div className="flex items-baseline justify-between">
                <label className="text-xs text-slate-300 tracking-wide">
                  {p.label}{p.unit ? <span className="text-slate-600 ml-1">({p.unit})</span> : null}
                </label>
                <button
                  type="button"
                  onClick={() => toggleHint(p.key)}
                  className="text-[10px] text-slate-600 hover:text-teal-400 transition-colors ml-1"
                  title="Show/hide description"
                >
                  {hintOpen ? '▲' : '▼ what?'}
                </button>
              </div>

              {/* Hint */}
              {hintOpen && (
                <p className="text-[10px] text-slate-500 leading-relaxed bg-white/[0.03] rounded-lg px-2.5 py-2 border border-white/[0.04]">
                  {p.hint}
                </p>
              )}

              {/* Input */}
              {p.type === 'text' ? (
                <input
                  type="text"
                  value={val}
                  onChange={e => handleParam(p.key, e.target.value)}
                  className="input-glass w-full px-3 py-2 text-sm font-mono rounded-xl"
                  placeholder={String(p.default)}
                />
              ) : p.noSlider ? (
                <input
                  type="number"
                  value={val}
                  onChange={e => handleParam(p.key, e.target.value === '' ? '' : parseFloat(e.target.value))}
                  onBlur={e => handleParamBlur(p.key, e.target.value, p)}
                  className={`input-glass w-full px-3 py-2 text-sm font-mono rounded-xl ${outOfRange ? 'border-amber-500/50 text-amber-300' : ''}`}
                  placeholder={String(p.default)}
                />
              ) : (
                <div className="flex items-center gap-2.5">
                  <input
                    type="range"
                    min={p.min}
                    max={p.max}
                    step={p.step}
                    value={isNaN(numVal) ? p.default : Math.min(p.max, Math.max(p.min, numVal))}
                    onChange={e => handleParam(p.key, parseFloat(e.target.value))}
                    className="flex-1"
                  />
                  <input
                    type="number"
                    value={val}
                    onChange={e => handleParam(p.key, e.target.value === '' ? '' : parseFloat(e.target.value))}
                    onBlur={e => handleParamBlur(p.key, e.target.value, p)}
                    className={`input-glass px-2.5 py-1.5 text-sm font-mono rounded-xl ${outOfRange ? 'border-amber-500/50 text-amber-300' : ''}`}
                    style={{ width: '5.5rem' }}
                    placeholder={String(p.default)}
                  />
                </div>
              )}

              {/* Out-of-range warning + range hint */}
              <div className="flex items-center justify-between gap-1">
                {outOfRange ? (
                  <span className="text-[10px] text-amber-500">
                    Valid range: {formatNum(p.min)}–{formatNum(p.max)}{p.unit ? ' ' + p.unit : ''}
                  </span>
                ) : (
                  <span className="text-[10px] text-slate-700">
                    {formatNum(p.min)} – {formatNum(p.max)}{p.unit ? ' ' + p.unit : ''}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Actions */}
      <div className="flex flex-col gap-2 pt-3 border-t border-white/[0.05]">
        <button
          type="submit"
          disabled={isRunning}
          className="btn-primary w-full py-3 rounded-xl text-sm flex items-center justify-center gap-2"
        >
          {isRunning && (
            <svg width="14" height="14" viewBox="0 0 14 14"
              style={{ animation: 'shear-spin 0.8s linear infinite', flexShrink: 0 }}>
              <style>{`@keyframes shear-spin{to{transform:rotate(360deg)}}`}</style>
              <circle cx="7" cy="7" r="5" fill="none" stroke="rgba(255,255,255,0.25)" strokeWidth="2"/>
              <path d="M7 2 A5 5 0 0 1 12 7" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          )}
          {isRunning ? 'Running V1 Analysis…' : 'Run V1 Analysis'}
        </button>
        <button
          type="button"
          onClick={handleExplore}
          disabled={isExploring || isRunning}
          className="btn-ghost w-full py-2.5 rounded-xl text-sm flex items-center justify-center gap-2"
        >
          {isExploring && (
            <svg width="14" height="14" viewBox="0 0 14 14"
              style={{ animation: 'shear-spin 0.8s linear infinite', flexShrink: 0 }}>
              <circle cx="7" cy="7" r="5" fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth="2"/>
              <path d="M7 2 A5 5 0 0 1 12 7" fill="none" stroke="rgba(129,140,248,0.9)" strokeWidth="2" strokeLinecap="round"/>
            </svg>
          )}
          {isExploring ? 'Running Sensitivity Analysis…' : 'Run Parametric Explorer'}
        </button>
      </div>
    </form>
  );
}
