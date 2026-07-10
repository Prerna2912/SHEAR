// Seeded pseudo-random (LCG) so mock data is deterministic
function makePRNG(seed = 42) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(1664525, s) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export function generateMockResult(geometryType = 'aerofoil', params = {}) {
  const rand = makePRNG(42);
  const gridSize = 8;
  const N = gridSize ** 3;
  const U = params.U_inf || params.flow_speed || params.flow_velocity || 10.0;

  const positions = [], meanTau = [], varianceTau = [], gradU = [];

  for (let ix = 0; ix < gridSize; ix++) {
    for (let iy = 0; iy < gridSize; iy++) {
      for (let iz = 0; iz < gridSize; iz++) {
        const x = (ix / (gridSize - 1)) * 4 - 1;   // streamwise -1 → 3
        const y = (iy / (gridSize - 1)) * 2 - 1;   // normal -1 → 1
        const z = (iz / (gridSize - 1)) * 2 - 1;   // spanwise -1 → 1
        positions.push([x, y, z]);

        const BL   = Math.exp(-Math.abs(y) / 0.3);
        const wake = x > 0 ? Math.exp(-x / 2) : 1;
        const base = (0.75 * BL + 0.25 * wake) * Math.exp(-Math.sqrt(x*x + y*y) * 0.25);
        const mag  = Math.max(0, base + 0.12 * (rand() - 0.5));

        // 6 stress components: 1×0e (isotropic) + 5×2e (traceless sym)
        meanTau.push([
          mag * 0.60,
          mag * 0.35 * (2 * rand() - 1),
          mag * 0.25 * (2 * rand() - 1),
          mag * 0.18 * (2 * rand() - 1),
          mag * 0.12 * (2 * rand() - 1),
          mag * 0.08 * (2 * rand() - 1),
        ]);

        varianceTau.push([
          mag * 0.08 + 0.004 * rand(),
          mag * 0.06 + 0.003 * rand(),
          mag * 0.05 + 0.003 * rand(),
          mag * 0.04 + 0.002 * rand(),
          mag * 0.03 + 0.002 * rand(),
          mag * 0.02 + 0.001 * rand(),
        ]);

        // Simplified BL velocity gradient (dominant shear = ∂u_x/∂y)
        const shear = U * Math.exp(-Math.abs(y) / 0.1) / 0.1 * Math.sign(y || 1);
        gradU.push([
          -0.15 * U * BL,   // ∂u_x/∂x
          shear,             // ∂u_x/∂y ← dominant
          0.03 * rand(),     // ∂u_x/∂z
          0.01 * rand(),     // ∂u_y/∂x
          0.08 * U * BL,    // ∂u_y/∂y
          0.01 * rand(),     // ∂u_y/∂z
          0.01 * rand(),     // ∂u_z/∂x
          0.01 * rand(),     // ∂u_z/∂y
          0.04 * U * BL,    // ∂u_z/∂z
        ]);
      }
    }
  }

  return {
    mean_tau:      meanTau,
    variance_tau:  varianceTau,
    samples:       null,
    positions,
    grad_u:        gradU,
    grid_shape:    [gridSize, gridSize, gridSize],
    geometry_hash: `mock-${geometryType}`,
    geometry_type: geometryType,
    params,
    n_samples:     20,
    n_nodes:       N,
    ode_steps:     100,
    inference_time_ms: 1247,
    job_id:        'mock-job-001',
    status:        'complete',
    error:         null,
  };
}

export function generateMockExplorerResult(geometryType, params) {
  const rand = makePRNG(7);
  const numeric = Object.entries(params)
    .filter(([, v]) => typeof v === 'number' && Math.abs(v) > 1e-9)
    .slice(0, 5);

  if (numeric.length === 0) return null;

  const levels = [-0.20, -0.10, -0.05, 0.05, 0.10, 0.20];

  // Different curve shapes per parameter index so charts are visually distinct
  const SHAPES = [
    lv => 0.52 + 0.18 * lv + 0.22 * lv * lv,                      // convex upward (stiff param)
    lv => 0.52 - 0.14 * lv + 0.08 * Math.sin(lv * Math.PI * 3),   // oscillatory (resonance-like)
    lv => 0.52 + 0.26 * Math.tanh(lv * 4),                         // saturating (nonlinear clamp)
    lv => 0.52 + 0.10 * lv - 0.30 * lv * lv,                       // concave (optimal at midrange)
    lv => 0.52 + 0.08 * lv,                                         // weak linear
  ];

  const paramRanking = numeric.map(([pname, bval], i) => {
    const shape = SHAPES[i % SHAPES.length];
    const perturbList = levels.map(lv => ({
      level: lv,
      param: pname,
      new_value: bval * (1 + lv),
      delta_peak_tau: shape(lv) - shape(0),
      peak_tau: shape(lv) * (1 + 0.015 * (rand() - 0.5)),
    }));
    const maxDelta = Math.max(...perturbList.map(p => Math.abs(p.delta_peak_tau)));
    return {
      param: pname,
      base_value: bval,
      max_abs_delta: maxDelta,
      perturbations: perturbList,
    };
  });

  paramRanking.sort((a, b) => b.max_abs_delta - a.max_abs_delta);

  const iLevels = [-0.20, -0.10, 0.0, 0.10, 0.20];
  const shapeA = SHAPES[0];
  const shapeB = SHAPES[1 % SHAPES.length];

  const grid = iLevels.map(fa =>
    iLevels.map(fb => {
      const da = shapeA(fa) - shapeA(0);
      const db = shapeB(fb) - shapeB(0);
      // Interaction term: cross-effect that makes the heatmap non-separable
      const interaction = 0.06 * fa * fb * (1 + 0.1 * (rand() - 0.5));
      return da + db + interaction;
    })
  );

  return {
    geometry_type: geometryType,
    base_params:   params,
    base_peak_tau: 0.52,
    parameter_ranking: paramRanking,
    interaction_map: numeric.length >= 2 ? {
      param_a:      paramRanking[0]?.param,
      param_b:      paramRanking[1]?.param,
      base_value_a: paramRanking[0]?.base_value,
      base_value_b: paramRanking[1]?.base_value,
      levels:       iLevels,
      grid,
    } : null,
    n_inference_calls: 6 * numeric.length + 25,
    total_time_ms:     13400,
  };
}
