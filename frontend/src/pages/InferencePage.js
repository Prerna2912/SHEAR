import { useState, useCallback, useRef } from 'react';
import GeometryForm from '../components/GeometryForm';
import StressViewer from '../components/StressViewer';
import CF4Panel from '../components/CF4Panel';
import CF5Explorer from '../components/CF5Explorer';
import AF1Panel from '../components/AF1Panel';
import ProgressTracker from '../components/ProgressTracker';
import CopilotPanel from '../components/CopilotPanel';
import { submitInference, connectProgress, fetchCF4, runExplorer } from '../data/api';
import { generateMockResult, generateMockExplorerResult } from '../data/mockFixture';
import { computeCF4 } from '../data/cf4Compute';
import { saveProject } from './ProjectsPage';

const API_BASE = process.env.REACT_APP_API_URL || '';
const MOCK_MODE = process.env.REACT_APP_MOCK_MODE === 'true' || !process.env.REACT_APP_API_URL;

const TABS = [
  { id: 0, label: '3D Field',    icon: '◉' },
  { id: 1, label: 'CF4',         icon: '⚡' },
  { id: 2, label: 'AF1',         icon: '⬡' },
  { id: 3, label: 'CF5 Explorer',icon: '≋' },
];

export default function InferencePage({ onResult }) {
  const [activeTab,      setActiveTab]      = useState(0);
  const [result,         setResult]         = useState(null);
  const [explorerResult, setExplorerResult] = useState(null);
  const [isRunning,      setIsRunning]      = useState(false);
  const [isExploring,    setIsExploring]    = useState(false);
  const [progressMsgs,   setProgressMsgs]   = useState([]);
  const [progress,       setProgress]       = useState(0);
  const [progressError,  setProgressError]  = useState(null);
  const [saveStatus,     setSaveStatus]     = useState(null);
  const [copilotOpen,    setCopilotOpen]    = useState(false);
  const [lastGtype,      setLastGtype]      = useState('aerofoil');
  const [lastParams,     setLastParams]     = useState({});
  const wsRef = useRef(null);

  const clearProgress = useCallback(() => { setProgressMsgs([]); setProgress(0); setProgressError(null); }, []);

  const handleSubmit = useCallback(async (geometryType, params) => {
    clearProgress();
    setIsRunning(true);
    setResult(null);
    setExplorerResult(null);
    setLastGtype(geometryType);
    setLastParams(params);
    setSaveStatus(null);

    if (MOCK_MODE) {
      setProgressMsgs([{ message: 'Synthesising ∇u field via panel method…', step: 1 }]);
      setProgress(30);
      await new Promise(r => setTimeout(r, 400));
      setProgressMsgs(prev => [...prev, { message: 'Running V1 ODE solve (20 samples)…', step: 2 }]);
      setProgress(70);
      await new Promise(r => setTimeout(r, 500));
      const mock = generateMockResult(geometryType, params);
      setResult(mock);
      onResult?.(mock);
      setProgress(100);
      setProgressMsgs(prev => [...prev, { message: 'Complete', step: 3 }]);
      setIsRunning(false);
      return;
    }

    try {
      const job = await submitInference(geometryType, params);
      wsRef.current?.close();
      wsRef.current = connectProgress(
        job.job_id,
        msg => { setProgressMsgs(prev => [...prev, msg]); if (msg.progress != null) setProgress(Math.round(msg.progress * 100)); },
        async msg => {
          setProgress(100);
          try {
            const cf4 = await fetchCF4(msg.result);
            const r = { ...msg.result, _cf4: cf4 };
            setResult(r); onResult?.(r);
          } catch {
            setResult(msg.result); onResult?.(msg.result);
          }
          setIsRunning(false);
        },
        err => { setProgressError(err.message || 'Inference failed'); setIsRunning(false); },
      );
    } catch (err) { setProgressError(err.message); setIsRunning(false); }
  }, [clearProgress, onResult]);

  const handleExplore = useCallback(async (geometryType, params) => {
    clearProgress();
    setIsExploring(true);
    setExplorerResult(null);

    if (MOCK_MODE) {
      setProgressMsgs([{ message: 'Running parametric sweep…' }]);
      setProgress(30);
      await new Promise(r => setTimeout(r, 800));
      setExplorerResult(generateMockExplorerResult(geometryType, params));
      setProgress(100);
      setProgressMsgs(prev => [...prev, { message: 'Explorer complete' }]);
      setIsExploring(false);
      setActiveTab(3);
      return;
    }
    try {
      const raw = await runExplorer(geometryType, params);
      setExplorerResult(raw);
      setActiveTab(3);
    } catch (err) { setProgressError(err.message); }
    finally { setIsExploring(false); }
  }, [clearProgress]);

  const handleSave = useCallback(async () => {
    if (!result) return;
    const name = window.prompt('Project name:', `${lastGtype} — ${new Date().toLocaleDateString()}`);
    if (!name) return;
    const tagsRaw = window.prompt('Tags (comma-separated):', lastGtype) || '';
    setSaveStatus('saving');
    try {
      await saveProject({ name, tags: tagsRaw.split(',').map(t => t.trim()).filter(Boolean), geometryType: lastGtype, params: lastParams, result, metrics: {} });
      setSaveStatus('saved');
      setTimeout(() => setSaveStatus(null), 3000);
    } catch { setSaveStatus('error'); setTimeout(() => setSaveStatus(null), 3000); }
  }, [result, lastGtype, lastParams]);

  const handleDownloadReport = useCallback(async () => {
    if (!result) return;
    if (MOCK_MODE) { alert('PDF export requires the backend server.'); return; }
    const res = await fetch(`${API_BASE}/api/v1/report`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ geometry_type: lastGtype, params: lastParams, metrics: {}, cf4_data: {}, cf5_data: explorerResult || {} }),
    });
    const blob = await res.blob();
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = `shear_report_${lastGtype}.pdf`; a.click();
  }, [result, lastGtype, lastParams, explorerResult]);

  const cf4Live = result ? computeCF4(result.mean_tau, result.variance_tau, result.grad_u) : null;
  const showProgress = progressMsgs.length > 0 || progressError;

  return (
    <div className="flex h-full" style={{ marginRight: copilotOpen ? '320px' : 0, transition: 'margin 0.3s ease' }}>

      {/* ── Left sidebar ─────────────────────────────────────── */}
      <aside className="w-64 shrink-0 glass-dark flex flex-col overflow-hidden">

        {/* Brand */}
        <div className="px-5 pt-6 pb-5 border-b border-white/[0.06]">
          <div className="text-xl font-display font-bold text-gradient tracking-tight">SHEAR</div>
          <div className="text-xs text-slate-500 mt-1 leading-relaxed">SE(3)-Equivariant Turbulence<br/>Flow Intelligence</div>
        </div>

        {/* Form */}
        <div className="flex-1 overflow-y-auto px-4 py-4">
          <GeometryForm
            onSubmit={handleSubmit}
            onExplore={handleExplore}
            isRunning={isRunning}
            isExploring={isExploring}
          />
        </div>

        {/* Actions */}
        {result && (
          <div className="px-4 pb-4 flex flex-col gap-2 border-t border-white/[0.04] pt-3">
            <button onClick={handleSave} disabled={saveStatus === 'saving'} className="btn-ghost w-full py-2 rounded-lg text-xs">
              {saveStatus === 'saving' ? 'Saving…' : saveStatus === 'saved' ? '✓ Saved to Projects' : 'Save Project'}
            </button>
            <button onClick={handleDownloadReport} className="btn-ghost w-full py-2 rounded-lg text-xs">
              Download PDF Report
            </button>
          </div>
        )}
      </aside>

      {/* ── Main area ─────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0">

        {/* Tab bar */}
        <div className="flex items-center border-b border-white/[0.07] bg-black/20 shrink-0 px-3">
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className={`flex items-center gap-2 px-5 py-4 text-sm font-medium transition-all border-b-2 tracking-wide ${
                activeTab === t.id
                  ? 'border-teal-400 text-teal-300'
                  : 'border-transparent text-slate-500 hover:text-slate-300'
              }`}
            >
              {t.label}
            </button>
          ))}

          {/* Copilot toggle */}
          <button
            onClick={() => setCopilotOpen(o => !o)}
            className={`ml-auto mr-1 flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all ${
              copilotOpen
                ? 'bg-gradient-to-r from-teal-600/25 to-indigo-600/25 text-teal-300 border border-teal-500/35'
                : 'btn-ghost'
            }`}
          >
            <span className="text-base">✦</span>
            {copilotOpen ? 'Close Copilot' : 'AI Copilot'}
          </button>
        </div>

        {/* Panel */}
        <div className="flex-1 min-h-0">
          {activeTab === 0 && <StressViewer result={result} />}
          {activeTab === 1 && <CF4Panel result={result} />}
          {activeTab === 2 && <AF1Panel result={result} af1Data={result?._af1 ?? null} />}
          {activeTab === 3 && <CF5Explorer explorerResult={explorerResult} />}
        </div>
      </div>

      {/* ── Progress toast ────────────────────────────────────── */}
      {showProgress && (
        <ProgressTracker messages={progressMsgs} progress={progress} error={progressError} />
      )}

      {/* ── AI Copilot panel ─────────────────────────────────── */}
      <CopilotPanel
        isOpen={copilotOpen}
        onClose={() => setCopilotOpen(false)}
        result={result}
        cf4={cf4Live}
        explorerResult={explorerResult}
        af1={result?._af1 ?? null}
        geometryType={lastGtype}
        params={lastParams}
      />
    </div>
  );
}
