"""
SHEAR CF4 — Gradio demo app.

Runs the full pipeline:
  CF1 (panel method → ∇u graph)
  → CF2 (V1 ODE → mean τ, variance)
  → CF4 (Q-criterion, uncertainty, dissipation, backscatter)
  → 4-panel diagnostic figure

Run locally:
    python demos/cf4_demo.py

HuggingFace Spaces:
    The entrypoint is `demo.launch(server_name="0.0.0.0", server_port=7860)`.
    Set V1_CHECKPOINT and V1_STATS env vars, or defaults are used.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np

# Defaults
CHECKPOINT = os.environ.get("V1_CHECKPOINT", "runs/quick/v1/last.pt")
STATS_PATH = os.environ.get("V1_STATS",      "runs/quick/stats.pt")
MOCK_FIXTURE = "tests/fixtures/mock_cf2_response.json"

# ---------------------------------------------------------------------------
# Model singleton — loaded once at startup
# ---------------------------------------------------------------------------

_model  = None
_stats  = None
_device = None

def _ensure_model():
    global _model, _stats, _device
    if _model is None:
        from src.cf2.checkpoint import load_v1
        _model, _stats, _device = load_v1(CHECKPOINT, STATS_PATH)


# ---------------------------------------------------------------------------
# Geometry parameter presets per type
# ---------------------------------------------------------------------------

GEOMETRY_DEFAULTS = {
    "aerofoil": dict(
        chord=1.0, span=5.0, alpha_deg=5.0, U_inf=50.0,
        Re=3_000_000.0, naca_code="2412"
    ),
    "swept_wing": dict(
        chord=2.0, span=12.0, alpha_deg=4.0, sweep_deg=30.0,
        dihedral_deg=5.0, U_inf=80.0, Re=8_000_000.0, naca_code="2412"
    ),
    "bluff_body": dict(
        length=1.0, height=0.5, U_inf=20.0, Re=50_000.0
    ),
    "cylinder": dict(
        diameter=0.1, U_inf=10.0, Re=100.0
    ),
    "flat_plate": dict(
        length=1.0, span=2.0, alpha_deg=3.0, U_inf=30.0, Re=500_000.0
    ),
    "turbine_blade": dict(
        blade_radius=40.0, chord=2.5, pitch_deg=5.0,
        rpm=12.0, V_axial=10.0, n_panels=60
    ),
    "ship_hull": dict(
        length=100.0, beam=15.0, draft=5.0, U_inf=5.0, Re=1_000_000_000.0
    ),
    "bluff_body_wake": dict(
        length=1.0, height=0.5, Cd=1.0, U_inf=20.0, Re=50_000.0
    ),
}


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------

def run_pipeline(
    geometry_type: str,
    alpha_deg: float,
    U_inf: float,
    Re: float,
    chord: float,
    span: float,
    n_samples: int,
    use_mock: bool,
) -> tuple:
    """
    Called by Gradio on every button click.
    Returns (figure, status_text).
    """
    try:
        if use_mock:
            with open(MOCK_FIXTURE) as f:
                result = json.load(f)
            status = (
                f"Mock fixture loaded ({result['n_nodes']} nodes). "
                "Uncheck 'Use mock fixture' to run live inference."
            )
        else:
            _ensure_model()

            # Build params from UI inputs + defaults for non-shown params
            base = dict(GEOMETRY_DEFAULTS.get(geometry_type, {}))
            base.update({
                "U_inf": float(U_inf),
                "Re":    float(Re),
            })
            if "alpha_deg" in base:
                base["alpha_deg"] = float(alpha_deg)
            if "chord" in base:
                base["chord"] = float(chord)
            if "span" in base:
                base["span"] = float(span)

            from src.cf2.inference import run_inference, geometry_hash
            from src.cf1 import solve as cf1_solve

            cf1_out = cf1_solve(geometry_type, base, grid_size=8)
            raw = run_inference(
                graph=cf1_out.graph,
                model=_model, stats=_stats, device=_device,
                n_samples=int(n_samples), ode_steps=50,
                include_samples=False,
            )

            result = {
                "mean_tau":     raw["mean_tau"].tolist(),
                "variance_tau": raw["variance_tau"].tolist(),
                "grad_u":       cf1_out.grad_u.reshape(-1, 9).tolist(),
                "grid_shape":   list(cf1_out.grid_shape),
                "geometry_type": geometry_type,
                "params":        base,
                "n_nodes":       raw["n_nodes"],
                "inference_time_ms": 0,
            }
            status = (
                f"Live inference complete — {result['n_nodes']} nodes, "
                f"{n_samples} CFM samples"
            )

        from src.cf4 import diagnose
        fig = diagnose(result, geometry_type=result["geometry_type"])

        return fig, status

    except Exception as e:
        import traceback
        return None, f"Error: {e}\n\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
body { background-color: #0f1117; color: #e0e0e0; }
.gr-button-primary { background: #6366f1 !important; }
.gr-box { background: #1a1d27 !important; border: 1px solid #333 !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="SHEAR CF4 — Regime & Uncertainty Diagnostics") as demo:

    gr.Markdown(
        """
        # SHEAR — CF4: Uncertainty & Regime Map
        **SE(3)-Equivariant SGS Stress Inference** | Panel Method → V1 ODE → Diagnostics

        Enter a geometry below and click **Run SHEAR** to see:
        - Flow regime map (Q-criterion)
        - Prediction uncertainty heatmap
        - SGS dissipation field (Π = −τ:S)
        - Backscatter map (energy back-transfer V1 captures, Smagorinsky cannot)
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Geometry")

            geometry_type = gr.Dropdown(
                choices=list(GEOMETRY_DEFAULTS.keys()),
                value="aerofoil",
                label="Geometry type",
            )
            alpha_deg = gr.Slider(-20, 20, value=5.0, step=0.5,
                                  label="Angle of attack α (deg)")
            U_inf = gr.Slider(1, 200, value=50.0, step=1.0,
                              label="Freestream velocity U∞ (m/s)")
            Re = gr.Slider(1e3, 1e7, value=3e6, step=1e5,
                           label="Reynolds number Re")
            chord = gr.Slider(0.1, 10.0, value=1.0, step=0.1,
                              label="Chord / length (m)")
            span  = gr.Slider(0.5, 20.0, value=5.0, step=0.5,
                              label="Span (m)")

            gr.Markdown("### Inference")
            n_samples = gr.Slider(3, 20, value=5, step=1,
                                  label="CFM samples (more = slower but better uncertainty)")
            use_mock = gr.Checkbox(
                value=True,
                label="Use mock fixture (instant, no model needed)",
            )

            run_btn = gr.Button("Run SHEAR", variant="primary")

        with gr.Column(scale=2):
            plot_out = gr.Plot(label="CF4 Diagnostic Panels")
            status_out = gr.Textbox(label="Status", lines=2, interactive=False)

    # Examples
    gr.Examples(
        examples=[
            ["aerofoil",       5.0,  50.0, 3e6, 1.0, 5.0,  5, True],
            ["swept_wing",     4.0,  80.0, 8e6, 2.0, 12.0, 5, True],
            ["cylinder",       0.0,  10.0, 1e5, 0.1, 1.0,  5, True],
            ["bluff_body_wake",0.0,  20.0, 5e4, 1.0, 2.0,  5, True],
            ["turbine_blade",  0.0,  10.0, 1e6, 2.5, 40.0, 5, True],
        ],
        inputs=[geometry_type, alpha_deg, U_inf, Re, chord, span, n_samples, use_mock],
        label="Quick examples",
    )

    run_btn.click(
        fn=run_pipeline,
        inputs=[geometry_type, alpha_deg, U_inf, Re, chord, span, n_samples, use_mock],
        outputs=[plot_out, status_out],
    )

    gr.Markdown(
        """
        ---
        **Coordinate system:** x=streamwise, y=normal, z=spanwise. Mid-plane slice shown (z = Nz/2).
        **Regimes:** Blue = strain-dominated (Q<p33), mid = mixed, Red = vortex-dominated (Q>p66).
        **Backscatter:** Smagorinsky always predicts Π ≥ 0. V1 captures back-transfer to resolved scales.
        """
    )


if __name__ == "__main__":
    print(f"[CF4 demo] Checkpoint: {CHECKPOINT}")
    print(f"[CF4 demo] Mock fixture: {MOCK_FIXTURE}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        css=CSS,
    )
