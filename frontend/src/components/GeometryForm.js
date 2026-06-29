import { useState } from 'react';

const GEOMETRY_CONFIGS = {
  aerofoil: {
    label: 'Aerofoil',
    desc: 'Lifting surface — aircraft wings and control surfaces',
    params: [
      { key: 'chord',     label: 'Chord (m)',              default: 1.0,    min: 0.01, max: 20,   step: 0.05 },
      { key: 'span',      label: 'Span (m)',               default: 5.0,    min: 0.1,  max: 100,  step: 0.5  },
      { key: 'alpha_deg', label: 'Angle of Attack (°)',    default: 5.0,    min: -30,  max: 30,   step: 0.5  },
      { key: 'U_inf',     label: 'Freestream Velocity (m/s)', default: 50.0, min: 0.1, max: 300, step: 1    },
      { key: 'Re',        label: 'Reynolds Number',        default: 3e6,    min: 1e3,  max: 1e9,  step: 1e5  },
      { key: 'naca_code', label: 'NACA Code',              default: '2412', type: 'text' },
    ],
  },
  swept_wing: {
    label: 'Swept Wing',
    desc: 'Swept lifting surface — transonic and supersonic aircraft',
    params: [
      { key: 'chord',       label: 'Chord (m)',               default: 2.0,  min: 0.01, max: 20,  step: 0.1  },
      { key: 'span',        label: 'Span (m)',                default: 10.0, min: 0.1,  max: 100, step: 0.5  },
      { key: 'sweep_angle', label: 'Sweep Angle (°)',         default: 30.0, min: -10,  max: 70,  step: 1    },
      { key: 'dihedral',    label: 'Dihedral (°)',            default: 5.0,  min: -15,  max: 15,  step: 0.5  },
      { key: 'alpha_deg',   label: 'Angle of Attack (°)',     default: 3.0,  min: -30,  max: 30,  step: 0.5  },
      { key: 'U_inf',       label: 'Freestream Velocity (m/s)', default: 60.0, min: 0.1, max: 300, step: 1  },
      { key: 'Re',          label: 'Reynolds Number',         default: 5e6,  min: 1e3,  max: 1e9, step: 1e5  },
      { key: 'naca_code',   label: 'NACA Code',               default: '2412', type: 'text' },
    ],
  },
  bluff_body: {
    label: 'Bluff Body / Vehicle',
    desc: 'Non-lifting bluff body — automotive, building, or industrial',
    params: [
      { key: 'length',     label: 'Length (m)',     default: 4.5,  min: 0.1, max: 100, step: 0.1 },
      { key: 'height',     label: 'Height (m)',     default: 1.5,  min: 0.1, max: 20,  step: 0.1 },
      { key: 'flow_speed', label: 'Flow Speed (m/s)', default: 30.0, min: 0.1, max: 150, step: 1 },
    ],
  },
  cylinder: {
    label: 'Cylinder / Pipe',
    desc: 'Circular cross-section — structural members and pipelines',
    params: [
      { key: 'diameter',       label: 'Diameter (m)',       default: 0.1, min: 0.001, max: 10,   step: 0.01 },
      { key: 'length',         label: 'Length (m)',         default: 1.0, min: 0.01,  max: 1000, step: 0.1  },
      { key: 'flow_velocity',  label: 'Flow Velocity (m/s)', default: 1.0, min: 0.01, max: 100,  step: 0.1  },
    ],
  },
  turbine_blade: {
    label: 'Wind Turbine Blade',
    desc: 'Rotating aerofoil — horizontal-axis wind turbine',
    params: [
      { key: 'radius',          label: 'Blade Radius (m)',    default: 40.0, min: 1,   max: 150, step: 1   },
      { key: 'pitch_angle',     label: 'Pitch Angle (°)',     default: 5.0,  min: -15, max: 30,  step: 0.5 },
      { key: 'tip_speed_ratio', label: 'Tip Speed Ratio',     default: 7.0,  min: 1,   max: 20,  step: 0.5 },
      { key: 'U_inf',           label: 'Wind Speed (m/s)',    default: 10.0, min: 3,   max: 25,  step: 0.5 },
    ],
  },
  flat_plate: {
    label: 'Flat Plate',
    desc: 'Zero-thickness plate — structural and aerodynamic panels',
    params: [
      { key: 'length',    label: 'Length (m)',          default: 1.0,  min: 0.01, max: 50, step: 0.05 },
      { key: 'width',     label: 'Width (m)',           default: 0.5,  min: 0.01, max: 50, step: 0.05 },
      { key: 'alpha_deg', label: 'Angle of Attack (°)', default: 10.0, min: -20,  max: 20, step: 0.5  },
    ],
  },
  ship_hull: {
    label: 'Ship Hull',
    desc: 'Marine hull — monohull displacement vessels',
    params: [
      { key: 'length', label: 'Hull Length (m)', default: 100.0, min: 1,   max: 500, step: 1   },
      { key: 'beam',   label: 'Beam (m)',         default: 15.0,  min: 0.5, max: 100, step: 0.5 },
      { key: 'draft',  label: 'Draft (m)',        default: 5.0,   min: 0.1, max: 30,  step: 0.1 },
      { key: 'speed',  label: 'Speed (m/s)',      default: 7.7,   min: 0.1, max: 30,  step: 0.1 },
    ],
  },
  bluff_body_wake: {
    label: 'Bluff Body + Wake',
    desc: 'Bluff body with downstream wake region modelled explicitly',
    params: [
      { key: 'length',      label: 'Body Length (m)',  default: 4.5,  min: 0.1, max: 100, step: 0.1 },
      { key: 'height',      label: 'Body Height (m)',  default: 1.5,  min: 0.1, max: 20,  step: 0.1 },
      { key: 'flow_speed',  label: 'Flow Speed (m/s)', default: 30.0, min: 0.1, max: 150, step: 1   },
      { key: 'wake_length', label: 'Wake Length (m)',  default: 20.0, min: 0.5, max: 200, step: 0.5 },
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

export default function GeometryForm({ onSubmit, onExplore, isRunning, isExploring }) {
  const [gtype, setGtype] = useState('aerofoil');
  const [params, setParams] = useState(defaultParams('aerofoil'));

  const cfg = GEOMETRY_CONFIGS[gtype];

  function handleGeometryChange(e) {
    const next = e.target.value;
    setGtype(next);
    setParams(defaultParams(next));
  }

  function handleParam(key, value) {
    setParams(prev => ({ ...prev, [key]: value }));
  }

  function handleSubmit(e) {
    e.preventDefault();
    onSubmit(gtype, params);
  }

  function handleExplore(e) {
    e.preventDefault();
    onExplore(gtype, params);
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
        <p className="mt-2 text-xs text-slate-600 leading-relaxed">{cfg.desc}</p>
      </div>

      {/* Parameters */}
      <div className="flex flex-col gap-3.5">
        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
          Parameters
        </span>
        {cfg.params.map(p => (
          <div key={p.key}>
            <label className="block text-xs text-slate-500 mb-1.5 tracking-wide">{p.label}</label>
            {p.type === 'text' ? (
              <input
                type="text"
                value={params[p.key] ?? p.default}
                onChange={e => handleParam(p.key, e.target.value)}
                className="input-glass w-full px-3 py-2 text-sm font-mono rounded-xl"
              />
            ) : (
              <div className="flex items-center gap-2.5">
                <input
                  type="range"
                  min={p.min}
                  max={p.max}
                  step={p.step}
                  value={params[p.key] ?? p.default}
                  onChange={e => handleParam(p.key, parseFloat(e.target.value))}
                  className="flex-1"
                />
                <input
                  type="number"
                  min={p.min}
                  max={p.max}
                  step={p.step}
                  value={params[p.key] ?? p.default}
                  onChange={e => handleParam(p.key, parseFloat(e.target.value))}
                  className="input-glass w-22 px-2.5 py-1.5 text-sm font-mono rounded-xl"
                  style={{ width: '5.5rem' }}
                />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Actions */}
      <div className="flex flex-col gap-2 pt-3 border-t border-white/[0.05]">
        <button
          type="submit"
          disabled={isRunning}
          className="btn-primary w-full py-3 rounded-xl text-sm"
        >
          {isRunning ? 'Running V1…' : 'Run V1 Analysis'}
        </button>
        <button
          type="button"
          onClick={handleExplore}
          disabled={isExploring || isRunning}
          className="btn-ghost w-full py-2.5 rounded-xl text-sm"
        >
          {isExploring ? 'Exploring…' : 'Run Parametric Explorer'}
        </button>
      </div>
    </form>
  );
}
