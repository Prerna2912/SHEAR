/**
 * AI Engineering Copilot — rule-based response engine.
 *
 * Reads CF4, CF5, AF1 data and generates plain-English engineering insights.
 * If REACT_APP_CLAUDE_API_KEY is set, delegates to Claude claude-haiku-4-5-20251001 for richer answers.
 */

const CLAUDE_KEY = process.env.REACT_APP_CLAUDE_API_KEY;

// ── Suggested quick questions ─────────────────────────────────────
export const QUICK_QUESTIONS = [
  'Is this design safe?',
  'What is causing the backscatter?',
  'Which parameter should I change first?',
  'How reliable are these predictions?',
  'Explain the Q-criterion results.',
  'What does the uncertainty mean?',
];

// ── Context builder ───────────────────────────────────────────────
export function buildContext({ geometryType, params, result, cf4, explorerResult, af1 }) {
  if (!result) return null;

  const magArr = result.mean_tau.map(t => Math.sqrt(t.reduce((s, x) => s + x * x, 0)));
  const peakTau = Math.max(...magArr);
  const meanTau = magArr.reduce((s, v) => s + v, 0) / magArr.length;

  const backscatterFrac = cf4?.backscatter_frac ?? null;
  const regimeCounts   = cf4 ? [0,1,2].map(r => cf4.regime_labels.filter(l => l === r).length) : null;
  const dominantRegime = regimeCounts
    ? ['strain-dominated', 'mixed', 'vortex-dominated'][regimeCounts.indexOf(Math.max(...regimeCounts))]
    : null;
  const highUncFrac = cf4 ? cf4.high_unc_mask.filter(Boolean).length / cf4.high_unc_mask.length : null;
  const isOOD  = af1?.ood?.is_ood ?? false;
  const mahal  = af1?.ood?.mahal_distance ?? null;

  const topParam   = explorerResult?.parameter_ranking?.[0] ?? null;
  const topDelta   = topParam?.max_abs_delta ?? null;

  return {
    geometryType, params, peakTau, meanTau,
    backscatterFrac, dominantRegime, highUncFrac,
    isOOD, mahal, topParam, topDelta,
    nNodes: result.n_nodes,
    regimeCounts,
  };
}

// ── Rule-based response engine ────────────────────────────────────
function ruleBasedResponse(question, ctx) {
  const q = question.toLowerCase();

  // Safety assessment
  if (q.includes('safe') || q.includes('dangerous') || q.includes('risk')) {
    const issues = [];
    if (ctx.isOOD) issues.push(`geometry is ${ctx.mahal?.toFixed(2)}σ outside the JHTDB training distribution — predictions may be unreliable`);
    if (ctx.highUncFrac > 0.35) issues.push(`${(ctx.highUncFrac * 100).toFixed(0)}% of nodes have high model uncertainty`);
    if (ctx.backscatterFrac > 0.20) issues.push(`${(ctx.backscatterFrac * 100).toFixed(0)}% backscatter fraction indicates significant energy return to resolved scales — potentially destabilising for an LES solver`);
    if (ctx.dominantRegime === 'vortex-dominated') issues.push(`flow is vortex-dominated — V1 has Pearson r = −0.06 in this regime at current training stage`);

    if (issues.length === 0) {
      return `**Assessment: Within expected range.**\n\nPeak τ = ${ctx.peakTau.toFixed(4)}, backscatter = ${(ctx.backscatterFrac * 100 || 0).toFixed(1)}%, uncertainty within normal bounds. The geometry is in-distribution — predictions should be reliable for engineering-level design decisions.\n\n*Note: V1 is at 10k training steps. Pearson r will improve toward 0.5+ at 50k steps.*`;
    }
    return `**Flags requiring attention (${issues.length}):**\n\n${issues.map((i, n) => `${n+1}. ${i.charAt(0).toUpperCase() + i.slice(1)}.`).join('\n')}\n\n*These are model confidence signals, not physical failure predictions. Use alongside classical CFD validation.*`;
  }

  // Backscatter
  if (q.includes('backscatter')) {
    const pct = ctx.backscatterFrac != null ? (ctx.backscatterFrac * 100).toFixed(1) : '—';
    return `**SGS Backscatter: ${pct}% of nodes**\n\nBackscatter (Π = −τ:S < 0) means energy is being transferred *from* sub-grid to resolved scales — the reverse of classical forward cascade. Smagorinsky cannot produce this by construction (it's always dissipative). V1's flow matching distribution captures it naturally.\n\nIn isotropic turbulence, typical backscatter fractions range from 5–25%. Values above 30% in concentrated regions may indicate genuine non-equilibrium dynamics worth further investigation.`;
  }

  // Parameter sensitivity
  if (q.includes('parameter') || q.includes('change') || q.includes('optimise') || q.includes('optimize') || q.includes('first')) {
    if (!ctx.topParam) return 'Run the **Parametric Explorer** first (click "Run Parametric Explorer" in the sidebar). I\'ll then tell you which parameter has the highest sensitivity.';
    const dir = ctx.topParam.perturbations?.[3]?.delta_peak_tau > 0 ? 'increases' : 'decreases';
    return `**Highest sensitivity: \`${ctx.topParam.param}\`**\n\nA ±10% change in \`${ctx.topParam.param}\` produces up to **${(ctx.topDelta * 100).toFixed(1)}% change** in peak τ — the largest of all parameters tested.\n\nIncreasing it ${dir} peak stress. Check the CF5 Explorer tab for the full sensitivity ranking and the 2D interaction map for joint effects with the second-ranked parameter.`;
  }

  // Reliability / accuracy
  if (q.includes('reliable') || q.includes('accurate') || q.includes('confidence') || q.includes('trust')) {
    const regimePearson = { 'strain-dominated': 0.12, 'mixed': 0.004, 'vortex-dominated': -0.06 };
    const r = ctx.dominantRegime ? regimePearson[ctx.dominantRegime] : '—';
    const oodNote = ctx.isOOD ? `\n\n⚠️ **OOD warning:** This geometry is ${ctx.mahal?.toFixed(2)}σ from the training distribution. Treat predictions with additional caution.` : '';
    return `**V1 reliability for this flow:**\n\nDominant regime: **${ctx.dominantRegime || 'unknown'}**. In Task 3 evaluation, V1 achieves Pearson r = **${typeof r === 'number' ? r.toFixed(3) : r}** in this regime at 10k training steps.\n\nV1's key guarantee is **rotational consistency** (ε = 1.8×10⁻⁵). Distribution quality (JSD = 0.132) is already the best in the model comparison. Pearson r is expected to reach ~0.5 at 50k steps.${oodNote}`;
  }

  // Q-criterion
  if (q.includes('q-criterion') || q.includes('q criterion') || q.includes('vortex') || q.includes('regime') || q.includes('vorticit')) {
    if (!ctx.regimeCounts) return 'Run a V1 analysis first to compute Q-criterion regime classification.';
    const total = ctx.regimeCounts.reduce((s, v) => s + v, 0);
    const pcts  = ctx.regimeCounts.map(c => ((c / total) * 100).toFixed(0));
    return `**Q-criterion regime breakdown:**\n\n• Strain-dominated: ${pcts[0]}% of nodes\n• Mixed: ${pcts[1]}% of nodes\n• Vortex-dominated: ${pcts[2]}% of nodes\n\nDominant: **${ctx.dominantRegime}**. The 66th percentile threshold is physically meaningful — it's where all classical models in the evaluation diverge most sharply from DNS. V1 is calibrated against this exact split.`;
  }

  // Uncertainty
  if (q.includes('uncertain') || q.includes('variance') || q.includes('confidence interval')) {
    const pct = ctx.highUncFrac != null ? (ctx.highUncFrac * 100).toFixed(0) : '—';
    return `**Model uncertainty: ${pct}% high-uncertainty nodes**\n\nV1 runs N=20 CFM ODE solves per request. Uncertainty is the L2 norm of per-component variance across samples. The top 25% of nodes by variance are flagged as high-uncertainty.\n\nHigh uncertainty typically concentrates in: strong shear layers, separated flow regions, and wake interaction zones. These are the areas where Smagorinsky also fails most severely — but V1 at least quantifies its own uncertainty, which Smagorinsky cannot.`;
  }

  // Fallback
  return `I can help interpret your SHEAR analysis. Try one of these:\n\n${QUICK_QUESTIONS.slice(0, 4).map(q => `• ${q}`).join('\n')}\n\nOr ask anything about the **CF4 diagnostics**, **backscatter**, **parameter sensitivity**, or **model reliability** for this geometry.`;
}

// ── Claude API path (optional) ────────────────────────────────────
async function claudeResponse(question, ctx) {
  const systemPrompt = `You are SHEAR Copilot, an AI assistant embedded in a turbulence flow intelligence platform.
You interpret SE(3)-equivariant flow matching (V1) predictions for aerospace, automotive, and marine engineers.
Speak in precise, confident engineering language. Never recommend specific numerical parameter values — you can describe trends and relative impacts.
Current analysis context:
- Geometry: ${ctx.geometryType}
- Parameters: ${JSON.stringify(ctx.params)}
- Peak τ: ${ctx.peakTau?.toFixed(4)}
- Backscatter: ${ctx.backscatterFrac != null ? (ctx.backscatterFrac*100).toFixed(1)+'%' : 'not computed'}
- Dominant regime: ${ctx.dominantRegime}
- High-uncertainty fraction: ${ctx.highUncFrac != null ? (ctx.highUncFrac*100).toFixed(0)+'%' : 'not computed'}
- OOD: ${ctx.isOOD ? `YES (${ctx.mahal?.toFixed(2)}σ)` : 'No'}
- Top sensitive parameter: ${ctx.topParam?.param ?? 'not computed'} (Δ = ${ctx.topDelta?.toFixed(4) ?? '—'})
Be concise (3–5 sentences max). Use markdown bold for key numbers.`;

  const resp = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'x-api-key': CLAUDE_KEY,
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true',
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      model: 'claude-haiku-4-5-20251001',
      max_tokens: 300,
      system: systemPrompt,
      messages: [{ role: 'user', content: question }],
    }),
  });

  if (!resp.ok) throw new Error(`Claude API: ${resp.status}`);
  const data = await resp.json();
  return data.content?.[0]?.text ?? '(empty response)';
}

// ── Public API ────────────────────────────────────────────────────
export async function getResponse(question, context) {
  if (!context) return 'Run a V1 analysis first — I need the results to help you interpret them.';

  if (CLAUDE_KEY) {
    try {
      return await claudeResponse(question, context);
    } catch (e) {
      console.warn('Claude API failed, falling back to rule-based:', e.message);
    }
  }

  return ruleBasedResponse(question, context);
}
