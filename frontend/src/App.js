import { BrowserRouter, Routes, Route, Navigate, Link, useLocation } from 'react-router-dom';
import InferencePage from './pages/InferencePage';
import ProjectsPage from './pages/ProjectsPage';
import BenchmarkPage from './pages/BenchmarkPage';
import { useState } from 'react';
import { defaultParams } from './components/GeometryForm';

function NavItem({ to, label }) {
  const loc = useLocation();
  const active = loc.pathname === to;
  return (
    <Link
      to={to}
      className={`px-5 py-4 text-sm font-medium transition-all border-b-2 tracking-wide ${
        active
          ? 'border-teal-400 text-teal-300'
          : 'border-transparent text-slate-500 hover:text-slate-200'
      }`}
    >
      {label}
    </Link>
  );
}

function Layout() {
  const loc = useLocation();
  const page = loc.pathname;

  const [lastResult,    setLastResult]    = useState(null);
  const [gtype,         setGtype]         = useState('aerofoil');
  const [formParams,    setFormParams]    = useState(() => defaultParams('aerofoil'));
  const [inferenceKey,  setInferenceKey]  = useState(0); // bump to reset InferencePage state

  // Keep all pages mounted — only show/hide with CSS so state is never lost.
  // absolute+inset:0 ensures each page fills the parent regardless of flex context.
  const show = (path) => page === path
    ? { position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', overflow: 'hidden' }
    : { display: 'none' };

  function handleReset() {
    setLastResult(null);
    setGtype('aerofoil');
    setFormParams(defaultParams('aerofoil'));
    setInferenceKey(k => k + 1); // force full remount of InferencePage
  }

  return (
    <div className="flex flex-col h-screen">
      <nav className="flex items-center glass border-b border-white/[0.07] px-6 shrink-0" style={{ borderRadius: 0, minHeight: '52px' }}>
        <div className="flex items-center gap-2.5 mr-8">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-teal-400 to-indigo-500 flex items-center justify-center shadow-lg shadow-teal-500/20">
            <span className="text-white text-xs font-bold">S</span>
          </div>
          <span className="font-display font-bold text-base text-gradient tracking-tight">SHEAR</span>
        </div>
        <NavItem to="/"          label="Inference" />
        <NavItem to="/projects"  label="Projects"  />
        <NavItem to="/benchmark" label="Benchmark" />
        <div className="ml-auto flex items-center gap-3">
          <button
            onClick={handleReset}
            title="Reset all analysis state"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-slate-500 hover:text-slate-300 hover:bg-white/[0.05] transition-all border border-transparent hover:border-white/[0.08]"
          >
            <svg width="11" height="11" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10 2A5 5 0 1 0 11 6" />
              <polyline points="10 0 10 2.5 7.5 2.5" />
            </svg>
            Reset
          </button>
          <span className="text-xs text-slate-600 font-mono tracking-wider">V1 · 10k</span>
        </div>
      </nav>

      {/* Redirect unknown paths */}
      <Routes>
        <Route path="*" element={<Navigate to="/" replace />} />
        <Route path="/" element={null} />
        <Route path="/projects" element={null} />
        <Route path="/benchmark" element={null} />
      </Routes>

      {/* Always-mounted pages — CSS visibility keeps state alive across nav */}
      <div className="flex-1 min-h-0 overflow-hidden" style={{ position: 'relative' }}>
        <div style={show('/')}>
          <InferencePage
            key={inferenceKey}
            onResult={setLastResult}
            gtype={gtype}
            formParams={formParams}
            onGtypeChange={setGtype}
            onFormParamsChange={setFormParams}
          />
        </div>
        <div style={show('/projects')}>
          <ProjectsPage onLoadProject={() => {}} />
        </div>
        <div style={show('/benchmark')}>
          <BenchmarkPage engineerResult={lastResult} />
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Layout />
    </BrowserRouter>
  );
}
