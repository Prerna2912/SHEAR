import { useState, useEffect, useCallback } from 'react';

const API_BASE = process.env.REACT_APP_API_URL || '';
const MOCK_MODE = process.env.REACT_APP_MOCK_MODE === 'true' || !process.env.REACT_APP_API_URL;

// localStorage-based mock store
function loadMockProjects() {
  try { return JSON.parse(localStorage.getItem('shear_projects') || '[]'); } catch { return []; }
}
function saveMockProjects(projects) {
  localStorage.setItem('shear_projects', JSON.stringify(projects));
}

function TagBadge({ tag }) {
  return (
    <span className="px-1.5 py-0.5 rounded-md text-[10px] bg-teal-500/10 text-teal-400 border border-teal-500/20 font-mono">
      {tag}
    </span>
  );
}

function ProjectCard({ project, onLoad, onDelete, isSelected, onSelect }) {
  const date = project.created_at
    ? new Date(project.created_at).toLocaleDateString()
    : '—';

  return (
    <div
      className={`glass-card cursor-pointer flex flex-col transition-all ${
        isSelected ? 'border-teal-500/50 shadow-[0_0_20px_rgba(20,184,166,0.12)]' : ''
      }`}
      onClick={() => onSelect(project.id)}
    >
      {/* Thumbnail */}
      <div className="h-28 rounded-t-xl overflow-hidden flex items-center justify-center bg-gradient-to-br from-teal-500/[0.08] to-indigo-500/[0.08] border-b border-white/[0.05]">
        {project.thumbnail_b64 ? (
          <img src={`data:image/png;base64,${project.thumbnail_b64}`} alt="" className="w-full h-full object-cover" />
        ) : (
          <div className="text-slate-700 text-xs text-center px-2 capitalize">
            {project.geometry_type?.replace(/_/g, ' ')}
          </div>
        )}
      </div>

      <div className="p-3 flex flex-col gap-1.5 flex-1">
        <div className="text-xs font-semibold text-slate-200 truncate leading-snug">{project.name}</div>
        <div className="flex gap-1 flex-wrap">
          {(project.tags || []).map(t => <TagBadge key={t} tag={t} />)}
        </div>
        <div className="text-xs text-slate-600 capitalize">{project.geometry_type?.replace(/_/g, ' ')} · {date}</div>
        {project.metrics?.peak_tau != null && (
          <div className="text-xs text-slate-500 font-mono">τ = {Number(project.metrics.peak_tau).toFixed(4)}</div>
        )}
        <div className="flex gap-2 mt-auto pt-1.5">
          <button
            onClick={e => { e.stopPropagation(); onLoad(project); }}
            className="btn-primary flex-1 py-1.5 text-xs rounded-lg"
          >
            Load
          </button>
          <button
            onClick={e => { e.stopPropagation(); onDelete(project.id); }}
            className="btn-ghost px-2.5 py-1.5 text-xs rounded-lg hover:border-red-500/40 hover:text-red-400"
          >
            ✕
          </button>
        </div>
      </div>
    </div>
  );
}

export default function ProjectsPage({ onLoadProject }) {
  const [projects,    setProjects]    = useState([]);
  const [filterTag,   setFilterTag]   = useState('');
  const [selectedIds, setSelectedIds] = useState([]);
  const [loading,     setLoading]     = useState(false);

  const refresh = useCallback(async () => {
    setLoading(true);
    if (MOCK_MODE) {
      setProjects(loadMockProjects());
      setLoading(false);
      return;
    }
    try {
      const url = filterTag ? `${API_BASE}/api/v1/projects?tag=${encodeURIComponent(filterTag)}` : `${API_BASE}/api/v1/projects`;
      const res = await fetch(url);
      setProjects(await res.json());
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [filterTag]);

  useEffect(() => { refresh(); }, [refresh]);

  function toggleSelect(id) {
    setSelectedIds(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : prev.length < 2 ? [...prev, id] : [prev[1], id]
    );
  }

  async function handleDelete(id) {
    if (MOCK_MODE) {
      const updated = loadMockProjects().filter(p => p.id !== id);
      saveMockProjects(updated);
      setProjects(updated);
      return;
    }
    await fetch(`${API_BASE}/api/v1/projects/${id}`, { method: 'DELETE' });
    refresh();
  }

  const allTags = [...new Set(projects.flatMap(p => p.tags || []))];
  const filtered = filterTag ? projects.filter(p => (p.tags || []).includes(filterTag)) : projects;
  const compareReady = selectedIds.length === 2;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-6 pt-5 pb-3 border-b border-white/[0.06] flex items-center justify-between gap-4 bg-black/10">
        <div>
          <div className="text-base font-bold font-display text-gradient">Project Library</div>
          <div className="text-xs text-slate-600 mt-0.5">{projects.length} saved {projects.length === 1 ? 'run' : 'runs'}</div>
        </div>
        <div className="flex gap-2 items-center flex-wrap">
          <div className="flex gap-1 flex-wrap">
            <button
              onClick={() => setFilterTag('')}
              className={`px-2.5 py-1 rounded-lg text-xs transition-all ${!filterTag ? 'bg-gradient-to-r from-teal-600/40 to-teal-500/30 text-teal-300 border border-teal-500/30' : 'btn-ghost'}`}
            >
              All
            </button>
            {allTags.map(t => (
              <button
                key={t}
                onClick={() => setFilterTag(t === filterTag ? '' : t)}
                className={`px-2.5 py-1 rounded-lg text-xs transition-all ${filterTag === t ? 'bg-gradient-to-r from-teal-600/40 to-teal-500/30 text-teal-300 border border-teal-500/30' : 'btn-ghost'}`}
              >
                {t}
              </button>
            ))}
          </div>
          {compareReady && (
            <button
              onClick={() => onLoadProject?.('compare', selectedIds.map(id => projects.find(p => p.id === id)))}
              className="px-3 py-1.5 rounded-lg text-xs font-semibold bg-gradient-to-r from-indigo-600/80 to-indigo-500/80 hover:from-indigo-500 hover:to-indigo-400 text-white transition-all border border-indigo-500/40"
            >
              Compare ({selectedIds.length})
            </button>
          )}
        </div>
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-y-auto p-6">
        {loading ? (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
            {[...Array(5)].map((_, i) => <div key={i} className="skeleton h-52 rounded-xl" />)}
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-52 gap-3">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-teal-500/15 to-indigo-500/15 flex items-center justify-center border border-white/[0.06]">
              <span className="text-xl">⊹</span>
            </div>
            <div className="text-center">
              <div className="text-slate-400 font-semibold text-sm font-display mb-1">No projects yet{filterTag ? ` tagged "${filterTag}"` : ''}</div>
              <div className="text-xs text-slate-600">Run an analysis and click "Save Project" to add one.</div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
            {filtered.map(p => (
              <ProjectCard
                key={p.id}
                project={p}
                onLoad={proj => onLoadProject?.('load', proj)}
                onDelete={handleDelete}
                isSelected={selectedIds.includes(p.id)}
                onSelect={toggleSelect}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ---- exported helper: save a project (called from InferencePage) ----
export async function saveProject({ name, tags, geometryType, params, result, metrics }) {
  const payload = {
    name,
    tags,
    geometry_type: geometryType,
    params,
    mean_tau:      result.mean_tau,
    variance_tau:  result.variance_tau,
    positions:     result.positions,
    grad_u:        result.grad_u,
    metrics,
    thumbnail_b64: null,
  };

  if (MOCK_MODE) {
    const projects = loadMockProjects();
    const newProj  = { ...payload, id: Date.now(), created_at: new Date().toISOString() };
    saveMockProjects([newProj, ...projects]);
    return newProj;
  }

  const res = await fetch(`${API_BASE}/api/v1/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Save failed: ${res.status}`);
  return res.json();
}
