import { BrowserRouter, Routes, Route, Navigate, Link, useLocation } from 'react-router-dom';
import InferencePage from './pages/InferencePage';
import ProjectsPage from './pages/ProjectsPage';
import BenchmarkPage from './pages/BenchmarkPage';
import { useState } from 'react';

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
  const [lastResult, setLastResult] = useState(null);

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
        <div className="ml-auto text-xs text-slate-600 font-mono tracking-wider">V1 · 10k</div>
      </nav>
      <div className="flex-1 min-h-0">
        <Routes>
          <Route path="/"          element={<InferencePage onResult={setLastResult} />} />
          <Route path="/projects"  element={<ProjectsPage onLoadProject={() => {}} />} />
          <Route path="/benchmark" element={<BenchmarkPage engineerResult={lastResult} />} />
          <Route path="*"          element={<Navigate to="/" replace />} />
        </Routes>
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
