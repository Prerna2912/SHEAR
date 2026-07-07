// Kept for import compatibility — rendering moved to InferenceLoadingOverlay / ExplorerLoadingOverlay
export default function ProgressTracker() { return null; }

// ── Spinner ──────────────────────────────────────────────────────────────────
function Spinner({ size = 32, color = '#14b8a6' }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32"
      style={{ animation: 'shear-spin 0.9s linear infinite' }}>
      <style>{`@keyframes shear-spin { to { transform: rotate(360deg) } }`}</style>
      <circle cx="16" cy="16" r="12" fill="none" stroke="rgba(255,255,255,0.07)" strokeWidth="3" />
      <path d="M16 4 A12 12 0 0 1 28 16" fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

// ── Step row ─────────────────────────────────────────────────────────────────
function StepRow({ step, active, done }) {
  const state = done ? 'done' : active ? 'active' : 'pending';
  return (
    <div className={`flex items-start gap-3 transition-opacity duration-300 ${state === 'pending' ? 'opacity-25' : 'opacity-100'}`}>
      <div className={`mt-0.5 w-5 h-5 rounded-full flex items-center justify-center shrink-0 text-[10px] font-bold
        ${state === 'done'   ? 'bg-green-500/20 text-green-400 border border-green-500/40' :
          state === 'active' ? 'bg-teal-500/20 text-teal-300 border border-teal-500/40' :
                               'bg-white/[0.04] text-slate-600 border border-white/[0.07]'}`}>
        {state === 'done' ? '✓' : step.id}
      </div>
      <div>
        <div className={`text-xs font-semibold
          ${state === 'active' ? 'text-slate-200' : state === 'done' ? 'text-green-400' : 'text-slate-600'}`}>
          {step.label}
        </div>
        <div className="text-[10px] text-slate-600 mt-0.5">{step.desc}</div>
      </div>
    </div>
  );
}

const STEPS = [
  { id: 1, label: 'Panel method',    desc: 'Computing velocity gradient field ∇u at every grid node' },
  { id: 2, label: 'ODE integration', desc: 'Generating stress samples via conditional flow matching' },
  { id: 3, label: 'Diagnostics',     desc: 'Computing Q-criterion, backscatter, regime labels' },
];

// ── V1 Analysis overlay ───────────────────────────────────────────────────────
export function InferenceLoadingOverlay({ isRunning, messages, progress, error }) {
  if (!isRunning && !error) return null;

  const msg = messages?.[messages.length - 1]?.message ?? '';
  const activeStep =
    msg.toLowerCase().includes('sample') || msg.toLowerCase().includes('ode') ? 2
    : msg.toLowerCase().includes('diagn') || msg.toLowerCase().includes('complete') ? 3
    : 1;

  return (
    <div className="absolute inset-0 flex items-center justify-center z-20 bg-[#0d1117]/75 backdrop-blur-sm">
      <div className="glass-card p-7 w-[22rem] flex flex-col gap-5 shadow-2xl">

        {/* Header */}
        <div className="flex items-center gap-3">
          {error
            ? <div className="w-8 h-8 rounded-full bg-red-500/20 border border-red-500/30 flex items-center justify-center text-red-400 text-sm font-bold">✕</div>
            : <Spinner size={30} />
          }
          <div>
            <div className="text-sm font-semibold text-slate-100">
              {error ? 'Analysis failed' : progress >= 100 ? 'Finalising results…' : 'Running V1 Analysis'}
            </div>
            {!error && (
              <div className="text-[11px] text-slate-500 mt-0.5">
                SE(3)-equivariant flow matching · {progress}%
              </div>
            )}
          </div>
        </div>

        {/* Progress bar */}
        {!error && (
          <div className="h-1.5 bg-white/[0.05] rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${progress}%`,
                background: progress >= 100
                  ? 'linear-gradient(90deg,#22c55e,#4ade80)'
                  : 'linear-gradient(90deg,#0d9488,#14b8a6,#38bdf8)',
              }}
            />
          </div>
        )}

        {/* Pipeline steps */}
        {!error && (
          <div className="flex flex-col gap-3 pt-1">
            {STEPS.map(s => (
              <StepRow key={s.id} step={s} active={s.id === activeStep} done={s.id < activeStep} />
            ))}
          </div>
        )}

        {/* Message / error */}
        <div className={`text-[11px] font-mono rounded-lg px-3 py-2 leading-relaxed
          ${error
            ? 'bg-red-500/10 text-red-400 border border-red-500/20'
            : 'bg-white/[0.03] text-slate-500 border border-white/[0.05]'}`}>
          {error || msg || 'Submitting job…'}
        </div>
      </div>
    </div>
  );
}

// ── Explorer loading overlay ──────────────────────────────────────────────────
const SWEEP_PARAMS = ['chord / span', 'angle of attack', 'freestream vel.', 'Reynolds no.'];

export function ExplorerLoadingOverlay({ isExploring, error }) {
  if (!isExploring && !error) return null;

  return (
    <div className="absolute inset-0 flex items-center justify-center z-20 bg-[#0d1117]/75 backdrop-blur-sm">
      <div className="glass-card p-7 w-[22rem] flex flex-col gap-5 shadow-2xl">

        {/* Header */}
        <div className="flex items-center gap-3">
          {error
            ? <div className="w-8 h-8 rounded-full bg-red-500/20 border border-red-500/30 flex items-center justify-center text-red-400 text-sm font-bold">✕</div>
            : <Spinner size={30} color="#818cf8" />
          }
          <div>
            <div className="text-sm font-semibold text-slate-100">
              {error ? 'Explorer failed' : 'Sensitivity Analysis'}
            </div>
            {!error && (
              <div className="text-[11px] text-slate-500 mt-0.5">
                Sweeping each parameter ±10% · ±20%
              </div>
            )}
          </div>
        </div>

        {!error ? (
          <>
            {/* Animated sweep bars */}
            <div className="flex flex-col gap-2.5">
              <style>{`
                @keyframes shear-sweep {
                  0%   { width: 8%;  opacity: 0.2; }
                  50%  { width: 92%; opacity: 0.7; }
                  100% { width: 8%;  opacity: 0.2; }
                }
              `}</style>
              {SWEEP_PARAMS.map((p, i) => (
                <div key={p} className="flex items-center gap-3">
                  <span className="text-[10px] text-slate-500 w-28 truncate">{p}</span>
                  <div className="flex-1 h-1 bg-white/[0.05] rounded-full overflow-hidden">
                    <div className="h-full bg-indigo-400/50 rounded-full" style={{
                      animation: `shear-sweep ${1.2 + i * 0.25}s ease-in-out infinite`,
                      animationDelay: `${i * 0.18}s`,
                    }} />
                  </div>
                </div>
              ))}
            </div>

            <div className="text-[11px] text-slate-500 leading-relaxed border-t border-white/[0.05] pt-3">
              Running fast inference for each perturbation.<br />
              Results will appear in the <span className="text-indigo-400 font-medium">Sensitivity tab</span> when complete.
            </div>
          </>
        ) : (
          <div className="text-[11px] font-mono rounded-lg px-3 py-2 bg-red-500/10 text-red-400 border border-red-500/20">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}
