export default function ProgressTracker({ messages, progress, error }) {
  if (!messages || messages.length === 0) return null;

  const latest = messages[messages.length - 1];

  return (
    <div className="fixed bottom-4 right-4 w-72 bg-slate-900 border border-slate-700 rounded-lg shadow-xl text-sm z-50">
      <div className="flex items-center justify-between px-3 py-2 border-b border-slate-700">
        <span className="font-semibold text-slate-200">
          {error ? 'Error' : progress < 100 ? 'Running V1…' : 'Complete'}
        </span>
        {!error && (
          <span className="text-xs text-slate-400">{progress}%</span>
        )}
      </div>

      {/* Progress bar */}
      {!error && (
        <div className="h-1 bg-slate-800 mx-3 mt-2 rounded-full overflow-hidden">
          <div
            className={`h-1 rounded-full transition-all duration-300 ${progress < 100 ? 'bg-teal-500' : 'bg-green-500'}`}
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      {/* Latest message */}
      <div className={`px-3 py-2 text-xs ${error ? 'text-red-400' : 'text-slate-400'}`}>
        {error || latest?.message || 'Processing…'}
      </div>

      {/* Log (collapsed) */}
      {messages.length > 1 && (
        <details className="px-3 pb-2">
          <summary className="text-xs text-slate-600 cursor-pointer hover:text-slate-400">
            {messages.length} events
          </summary>
          <div className="mt-1 flex flex-col gap-0.5 max-h-24 overflow-y-auto">
            {messages.map((m, i) => (
              <div key={i} className="text-xs text-slate-600 font-mono">
                {m.step != null && <span className="text-slate-500">[{m.step}] </span>}
                {m.message}
              </div>
            ))}
          </div>
        </details>
      )}
    </div>
  );
}
