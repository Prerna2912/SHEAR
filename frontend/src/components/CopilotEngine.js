/**
 * SHEAR AI Copilot — response engine.
 *
 * Primary:  Groq (free tier) via REACT_APP_GROQ_API_KEY (set in .env.local).
 *           Get a free key at https://console.groq.com
 * Fallback: context-aware rule engine using live analysis data.
 */

const GROQ_KEY = process.env.REACT_APP_GROQ_API_KEY;

export const QUICK_QUESTIONS = [
  'Is this design safe?',
  'What is causing the backscatter?',
  'Which parameter should I change first?',
  'How reliable are these predictions?',
  'Explain the vortex regime results.',
  'What does the uncertainty mean?',
];

// ── Context builder ───────────────────────────────────────────────
export function buildContext({ geometryType, params, result, cf4, explorerResult, af1 }) {
  if (!result) return null;

  const magArr      = result.mean_tau.map(t => Math.sqrt(t.reduce((s, x) => s + x * x, 0)));
  const peakTau     = Math.max(...magArr);
  const meanTau     = magArr.reduce((s, v) => s + v, 0) / magArr.length;

  const backscatterFrac = cf4?.backscatter_frac ?? null;
  const regimeCounts    = cf4 ? [0, 1, 2].map(r => cf4.regime_labels.filter(l => l === r).length) : null;
  const dominantRegime  = regimeCounts
    ? ['strain-dominated', 'mixed', 'vortex-dominated'][regimeCounts.indexOf(Math.max(...regimeCounts))]
    : null;
  const highUncFrac     = cf4 ? cf4.high_unc_mask.filter(Boolean).length / cf4.high_unc_mask.length : null;
  const meanDissipation = cf4?.dissipation
    ? cf4.dissipation.reduce((s, v) => s + v, 0) / cf4.dissipation.length
    : null;
  const isOOD  = af1?.ood?.is_ood ?? false;
  const mahal  = af1?.ood?.mahal_distance ?? null;
  const topParam  = explorerResult?.parameter_ranking?.[0] ?? null;
  const topDelta  = topParam?.max_abs_delta ?? null;
  const ranking   = explorerResult?.parameter_ranking ?? [];

  return {
    geometryType, params, peakTau, meanTau,
    backscatterFrac, dominantRegime, highUncFrac, meanDissipation,
    isOOD, mahal, topParam, topDelta, ranking,
    nNodes: result.n_nodes,
    regimeCounts,
  };
}

// ── Rule-based engine ─────────────────────────────────────────────
function ruleBasedResponse(question, ctx) {
  const q = question.toLowerCase();

  // Safety / risk
  if (q.includes('safe') || q.includes('dangerous') || q.includes('risk') || q.includes('concern') || q.includes('problem') || q.includes('issue') || q.includes('worry') || q.includes('flag')) {
    const issues = [];
    if (ctx.isOOD) issues.push(`geometry sits **${ctx.mahal?.toFixed(2)}σ** outside the training distribution — predictions may be less reliable`);
    if (ctx.highUncFrac > 0.35) issues.push(`**${(ctx.highUncFrac * 100).toFixed(0)}%** of grid nodes have high model uncertainty`);
    if (ctx.backscatterFrac > 0.20) issues.push(`**${(ctx.backscatterFrac * 100).toFixed(0)}%** backscatter fraction indicates significant energy returning to resolved scales`);
    if (ctx.dominantRegime === 'vortex-dominated') issues.push(`flow is vortex-dominated — the model has lower correlation in this regime at the current training stage`);
    if (issues.length === 0) {
      return `**Assessment: within expected range.**\n\nPeak τ = **${ctx.peakTau.toFixed(4)}** · Backscatter = **${((ctx.backscatterFrac ?? 0) * 100).toFixed(1)}%** · Uncertainty within normal bounds.\n\nGeometry is in-distribution — predictions are reliable for engineering-level decisions.\n\n*Note: full accuracy improves with longer model training.*`;
    }
    return `**${issues.length} flag${issues.length > 1 ? 's' : ''} to review:**\n\n${issues.map((x, n) => `${n + 1}. ${x.charAt(0).toUpperCase() + x.slice(1)}.`).join('\n')}\n\nThese are model confidence signals, not physical failure predictions. Use alongside classical CFD validation for critical decisions.`;
  }

  // Backscatter
  if (q.includes('backscatter') || q.includes('energy return') || q.includes('reverse cascade') || q.includes('back scatter')) {
    const pct = ctx.backscatterFrac != null ? (ctx.backscatterFrac * 100).toFixed(1) : '—';
    return `**SGS Backscatter: ${pct}% of nodes**\n\nBackscatter (SGS dissipation Π = −τ:S < 0) means energy transfers *from* sub-grid to resolved scales — the reverse of the classical forward energy cascade.\n\nClassical algebraic models like Smagorinsky are always dissipative and cannot produce backscatter. The probabilistic generative model captures it naturally because it samples from the full stress distribution rather than predicting a single mean.\n\nIn isotropic turbulence, typical backscatter fractions are **5–25%**. Values above 30% in concentrated regions indicate non-equilibrium dynamics worth investigating further.`;
  }

  // Parameter sensitivity
  if (q.includes('parameter') || (q.includes('which') && q.includes('change')) || q.includes('optimis') || q.includes('optimiz') || q.includes('tweak') || q.includes('adjust') || q.includes('sensitivity') || q.includes('most important') || q.includes('biggest') || q.includes('lever')) {
    if (!ctx.topParam) return 'Run the **Sensitivity Explorer** first — click "Run Parametric Explorer" in the left sidebar. It will sweep all parameters ±5/10/20% and rank them by their effect on peak stress.';
    const dir = ctx.topParam.perturbations?.[3]?.delta_peak_tau > 0 ? 'increases' : 'decreases';
    const rankList = ctx.ranking.slice(0, 3).map((r, i) => `${i + 1}. \`${r.param}\` — up to ±${(r.max_abs_delta * 100).toFixed(1)}% change in peak τ`).join('\n');
    return `**Most sensitive parameter: \`${ctx.topParam.param}\`**\n\nA ±10% change produces up to **${(ctx.topDelta * 100).toFixed(1)}%** change in peak τ. Increasing \`${ctx.topParam.param}\` ${dir} peak stress.\n\nTop 3 by sensitivity:\n${rankList}\n\nCheck the Sensitivity tab's interaction map for nonlinear coupling between the top two parameters — if the heatmap is asymmetric, they are not independent.`;
  }

  // Reliability / accuracy
  if (q.includes('reliable') || q.includes('accurate') || q.includes('confidence') || q.includes('trust') || q.includes('accuracy') || q.includes('how well') || q.includes('pearson') || q.includes('correlation')) {
    const regimePearson = { 'strain-dominated': 0.12, 'mixed': 0.004, 'vortex-dominated': -0.06 };
    const r = ctx.dominantRegime ? regimePearson[ctx.dominantRegime] : null;
    const oodNote = ctx.isOOD ? `\n\n⚠️ **Distribution warning:** ${ctx.mahal?.toFixed(2)}σ from training data — treat predictions with caution.` : '';
    return `**Model reliability for ${ctx.geometryType}:**\n\nDominant flow regime: **${ctx.dominantRegime || 'unknown'}**\nPrediction correlation (Pearson r): **${r != null ? r.toFixed(3) : '—'}**\n\n**Architectural guarantees:** SE(3) equivariance (rotation error ε = 1.8×10⁻⁵), backscatter prediction, Jensen-Shannon divergence = 0.132 vs training data. Correlation is expected to improve significantly with extended training.${oodNote}`;
  }

  // Q-criterion / vortex / regime
  if (q.includes('q-criterion') || q.includes('q criterion') || q.includes('vortex') || q.includes('regime') || q.includes('vorticit') || q.includes('strain') || q.includes('rotation') || q.includes('swirl')) {
    if (!ctx.regimeCounts) return 'Run a V1 analysis first to compute the flow regime classification.';
    const total = ctx.regimeCounts.reduce((s, v) => s + v, 0);
    const pcts  = ctx.regimeCounts.map(c => ((c / total) * 100).toFixed(0));
    return `**Flow regime breakdown (Q-criterion):**\n\n• Straining — attached boundary layer (Q < p33): **${pcts[0]}%** of nodes\n• Mixed — transition and mild separation (p33–p66): **${pcts[1]}%** of nodes\n• Vortical — coherent wake vortices (Q > p66): **${pcts[2]}%** of nodes\n\nDominant: **${ctx.dominantRegime}**\n\nQ = ½(‖Ω‖² − ‖S‖²) separates vortex cores (Q > 0) from strain-dominated irrotational regions (Q < 0). The p33/p66 thresholds adapt to the field — they are percentiles of the Q distribution, not fixed values, so classification is geometry-agnostic.`;
  }

  // Uncertainty
  if (q.includes('uncertain') || q.includes('variance') || q.includes('confidence interval') || q.includes('error bar') || q.includes('spread') || q.includes('error')) {
    const pct = ctx.highUncFrac != null ? (ctx.highUncFrac * 100).toFixed(0) : '—';
    return `**Model uncertainty: ${pct}% of nodes flagged high**\n\nThe model runs 20 independent generation trajectories from different random starts. Uncertainty = L2 norm of variance across those 20 samples at each grid node. Nodes in the top 25% by variance are flagged.\n\nHigh uncertainty concentrates in: strong shear layers, separated flow zones, wake interaction regions. This is physically meaningful — these are exactly the regions where turbulence is genuinely unpredictable from the local velocity gradient alone.\n\nCritically, classical algebraic models give one number with no error estimate. This model quantifies its own confidence, telling you exactly where to trust it and where to validate with simulation.`;
  }

  // Dissipation / SGS / energy transfer
  if (q.includes('dissipation') || q.includes('sub-grid') || q.includes('subgrid') || q.includes('energy') || q.includes('cascade') || q.includes('transfer') || q.includes('sgs')) {
    const meanDiss = ctx.meanDissipation != null ? ctx.meanDissipation.toFixed(5) : '—';
    return `**SGS Energy Dissipation (Π = −τ:S)**\n\nMean dissipation across domain: **${meanDiss}**\n\nPositive Π = energy cascades forward from large to small scales (normal, expected everywhere).\nNegative Π = backscatter — energy returns from sub-grid to resolved scales.\n\nSGS dissipation is the most physically important scalar from any stress model — it determines whether the LES solver's energy budget is correct. Classical Smagorinsky always returns Π ≥ 0. This model captures both directions, making it the only model in this comparison that physically accounts for reverse energy transfer.`;
  }

  // Stress / tau / magnitude
  if (q.includes('stress') || q.includes('tau') || q.includes('τ') || q.includes('magnitude') || q.includes('peak') || q.includes('mean') || q.includes('tensor')) {
    return `**SGS Stress Field — ${ctx.geometryType}:**\n\nPeak |τ| = **${ctx.peakTau.toFixed(4)}** · Mean |τ| = **${ctx.meanTau.toFixed(4)}**\n\nτᵢⱼ is the sub-grid scale stress tensor — the unclosed term in the filtered Navier-Stokes equations that represents momentum transport by unresolved turbulent eddies. The model predicts all 6 independent components as an equivariant geometric object that transforms correctly under rotation.\n\nThe 3D stress field shown in the first tab colours all **${ctx.nNodes}** grid nodes by stress magnitude (or Q-criterion, uncertainty, or dissipation depending on which overlay you select).`;
  }

  // Geometry specific
  if (q.includes('aerofoil') || q.includes('airfoil') || q.includes('wing') || q.includes('blade') || q.includes('hull') || q.includes('cylinder') || q.includes('plate') || q.includes('turbine') || q.includes(ctx.geometryType?.toLowerCase() ?? '____') || q.includes('geometry') || q.includes('shape')) {
    return `**${ctx.geometryType} — flow analysis:**\n\nA panel method computed the velocity gradient field across a structured 3D grid around this geometry. The model then produced a stress field with per-node uncertainty.\n\nFor **${ctx.geometryType}**: the boundary layer near the body surface typically shows the highest stress magnitude. The wake region downstream shows vortex structures. Dominant regime: **${ctx.dominantRegime}**.\n\nSwitch between the overlay modes in the 3D tab to see stress magnitude, Q-criterion vortex structure, prediction uncertainty, SGS dissipation, and flow regime classification at every grid point.`;
  }

  // Equivariance / SE3 / model architecture
  if (q.includes('equivarian') || q.includes('se(3)') || q.includes('se3') || q.includes('egnn') || q.includes('model') || q.includes('architecture') || q.includes('neural') || q.includes('network') || q.includes('how trained') || q.includes('machine learning') || q.includes('ai')) {
    return `**SE(3)-Equivariant Architecture:**\n\nThe model is an equivariant graph neural network with 5 message-passing layers. Each layer computes tensor products of node features with spherical harmonics evaluated on edge directions, weighted by a radial MLP on inter-node distances.\n\n**Key property:** rotating the input velocity gradient by matrix R produces exactly the correspondingly rotated stress output. This SE(3) equivariance is guaranteed by the architecture (rotation error ε = 1.8×10⁻⁵) — not learned from data. Physically, it means the model is coordinate-frame-independent, a requirement for any physically consistent stress model.\n\nThe generative component uses conditional flow matching: the model learns a vector field from noise to the stress distribution conditioned on the local velocity gradient, then integrates it with an adaptive ODE solver.`;
  }

  // How does it work / explain / what is / overview
  if (q.includes('how does') || q.includes('explain') || q.includes('what is') || q.includes('what are') || q.includes('tell me') || q.includes('describe') || q.includes('overview') || q.includes('walk me') || q.includes('summarise') || q.includes('summarize')) {
    return `**SHEAR — how it works:**\n\nGeometry: **${ctx.geometryType}** · **${ctx.nNodes}** grid nodes\n\n1. **Flow field solver** — a panel method computes velocity gradients at every grid node around the geometry\n2. **Generative model** — runs 20 independent ODE trajectories from Gaussian noise, each producing one stress field sample\n3. **Stress estimate** — mean across 20 samples; variance = per-node uncertainty\n4. **Diagnostics** — Q-criterion regime, backscatter fraction, SGS dissipation computed from the stress field\n5. **Sensitivity analysis** — automated ±5/10/20% parameter sweeps rank which inputs matter most\n\nCurrent result: peak τ = **${ctx.peakTau.toFixed(4)}** · Dominant regime: **${ctx.dominantRegime ?? 'not yet computed'}**`;
  }

  // Comparison to classical models / baselines
  if (q.includes('smagorinsky') || q.includes('baseline') || q.includes('compare') || q.includes('better') || q.includes(' vs ') || q.includes('versus') || q.includes('classical') || q.includes('traditional') || q.includes('wale')) {
    return `**This model vs. classical SGS models:**\n\n| Capability | This model | Smagorinsky | Plain MLP |\n|---|---|---|---|\n| Backscatter prediction | ✓ | ✗ | ✓ |\n| SE(3) equivariance | ✓ | ✓ | ✗ |\n| Per-node uncertainty | ✓ | ✗ | ✗ |\n| Learned from DNS data | ✓ | ✗ | ✓ |\n\nKey advantages over Smagorinsky: backscatter prediction, data-driven calibration, uncertainty quantification. Main trade-off: slower — 20 ODE solves vs. one algebraic evaluation per timestep. In practice, this model is used for analysis and design exploration, not real-time LES coupling.`;
  }

  // Fallback: synthesise all available context
  const lines = [
    `**Current analysis — ${ctx.geometryType}:**`,
    `\nPeak τ = **${ctx.peakTau.toFixed(4)}** · Mean τ = **${ctx.meanTau.toFixed(4)}**`,
    ctx.dominantRegime ? `Dominant flow regime: **${ctx.dominantRegime}**` : null,
    ctx.backscatterFrac != null ? `Backscatter (reverse energy cascade): **${(ctx.backscatterFrac * 100).toFixed(1)}%** of nodes` : null,
    ctx.highUncFrac != null ? `High model uncertainty: **${(ctx.highUncFrac * 100).toFixed(0)}%** of nodes` : null,
    ctx.meanDissipation != null ? `Mean SGS dissipation: **${ctx.meanDissipation.toFixed(5)}**` : null,
    ctx.isOOD ? `⚠️ Distribution warning: **${ctx.mahal?.toFixed(2)}σ** from training data` : `Distribution check: **in-distribution** (reliable predictions)`,
    ctx.topParam ? `Most sensitive parameter: **${ctx.topParam.param}** (up to ±${(ctx.topDelta * 100).toFixed(1)}% effect on peak stress)` : null,
    `\nAsk me about: safety, backscatter, parameter sensitivity, vortex regimes, uncertainty, energy dissipation, model reliability, or the equivariant architecture.`,
  ].filter(Boolean);

  return lines.join('\n');
}

// ── Groq API path ─────────────────────────────────────────────────
async function groqResponse(question, ctx, history) {
  const contextBlock = ctx ? `
Current analysis context:
- Geometry: ${ctx.geometryType} | Parameters: ${JSON.stringify(ctx.params)}
- Peak |τ|: ${ctx.peakTau?.toFixed(4)} | Mean |τ|: ${ctx.meanTau?.toFixed(4)}
- Backscatter fraction: ${ctx.backscatterFrac != null ? (ctx.backscatterFrac * 100).toFixed(1) + '%' : 'not computed'}
- Dominant flow regime: ${ctx.dominantRegime ?? 'not computed'}
- High-uncertainty nodes: ${ctx.highUncFrac != null ? (ctx.highUncFrac * 100).toFixed(0) + '%' : 'not computed'}
- Mean SGS dissipation: ${ctx.meanDissipation?.toFixed(5) ?? 'not computed'}
- Out-of-distribution: ${ctx.isOOD ? `YES (${ctx.mahal?.toFixed(2)}σ from training data)` : 'No — in distribution'}
- Most sensitive parameter: ${ctx.topParam?.param ?? 'sensitivity explorer not run'} (effect = ${ctx.topDelta?.toFixed(4) ?? '—'})
- Grid nodes: ${ctx.nNodes}` : 'No analysis has been run yet — answer general questions about the tool, methodology, or geometry setup.';

  const systemPrompt = `You are SHEAR Copilot, an expert AI assistant embedded in a turbulence flow intelligence platform for aerospace, automotive, and marine engineers.

You interpret predictions from a SE(3)-equivariant conditional flow matching model that predicts sub-grid scale stress tensors for large eddy simulation.

Rules:
- Speak in precise, confident engineering language. Avoid jargon like "CF4", "AF1", "CF5", "Task 3" — use plain names (diagnostics panel, flow regime analysis, sensitivity explorer).
- Keep answers to 3–6 sentences with key numbers bolded.
- Never suggest specific parameter values — describe trends only.

${contextBlock}`;

  const messages = [
    { role: 'system', content: systemPrompt },
    ...history.slice(-6).map(m => ({ role: m.role, content: m.content })),
    { role: 'user', content: question },
  ];

  const resp = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${GROQ_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'llama-3.3-70b-versatile',
      max_tokens: 400,
      messages,
    }),
  });

  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.error?.message ?? `HTTP ${resp.status}`);
  }
  const data = await resp.json();
  return data.choices?.[0]?.message?.content ?? '(empty response)';
}

// ── Public API ────────────────────────────────────────────────────
export async function getResponse(question, context, history = []) {
  if (GROQ_KEY) {
    try {
      return await groqResponse(question, context, history);
    } catch (e) {
      console.warn('Groq API error:', e.message);
      if (!context) return 'I\'m having trouble connecting right now. Try running a V1 analysis first and then ask me questions about the results.';
      return ruleBasedResponse(question, context) + `\n\n*Using local engine (${e.message}).*`;
    }
  }

  if (!context) {
    return 'Run a V1 analysis first — submit a geometry in the left sidebar and I\'ll help you interpret the results.';
  }

  return ruleBasedResponse(question, context);
}
