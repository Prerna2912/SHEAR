"""
AF4 — PDF Report generation via ReportLab + Matplotlib.

Multi-page report: cover/summary, stress analysis, CF4 diagnostics,
CF5 sensitivity explorer, technical notes.
"""
from __future__ import annotations

import io
import math
from datetime import datetime
from typing import Any, Dict, List, Optional


# ── Chart helpers ────────────────────────────────────────────────────────────

def _to_png(fig) -> io.BytesIO:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


def _tau_histogram(tau_mags: List[float]) -> io.BytesIO:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    mags = np.array(tau_mags)
    fig, ax = plt.subplots(figsize=(6.5, 3))
    n, bins, patches = ax.hist(mags, bins=35, color="#14b8a6", alpha=0.85,
                                edgecolor="white", linewidth=0.4)
    # Colour bars by magnitude
    norm = plt.Normalize(bins[0], bins[-1])
    cmap = plt.cm.get_cmap("YlOrRd")
    for patch, left in zip(patches, bins):
        patch.set_facecolor(cmap(norm(left)))

    ax.axvline(float(np.mean(mags)), color="#0f766e", linewidth=1.4,
               linestyle="--", label=f"Mean = {np.mean(mags):.4f}")
    ax.axvline(float(np.max(mags)),  color="#dc2626", linewidth=1.4,
               linestyle="--", label=f"Peak = {np.max(mags):.4f}")
    ax.legend(fontsize=8)
    ax.set_xlabel("|τ|  (SGS stress magnitude)", fontsize=9)
    ax.set_ylabel("Node count", fontsize=9)
    ax.set_title("Distribution of SGS Stress Magnitude across Grid Nodes", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _to_png(fig)


def _uncertainty_histogram(unc_vals: List[float]) -> io.BytesIO:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.array(unc_vals)
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ax.hist(arr, bins=35, color="#818cf8", alpha=0.85, edgecolor="white", linewidth=0.4)
    p75 = float(np.percentile(arr, 75))
    ax.axvline(p75, color="#4f46e5", linewidth=1.4, linestyle="--",
               label=f"75th pct = {p75:.4f}  (high-unc threshold)")
    ax.legend(fontsize=8)
    ax.set_xlabel("Uncertainty (σ norm per node)", fontsize=9)
    ax.set_ylabel("Node count", fontsize=9)
    ax.set_title("Epistemic Uncertainty Distribution (V1 Ensemble Variance)", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _to_png(fig)


def _regime_chart(regime_labels: List[int], backscatter_frac: float) -> io.BytesIO:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.array(regime_labels)
    n   = max(len(arr), 1)
    labels = ["Strain-\ndominated", "Mixed\nregime", "Vortex-\ndominated"]
    colors = ["#0ea5e9", "#8b5cf6", "#f59e0b"]
    counts = [(arr == i).sum() for i in range(3)]
    pcts   = [c / n * 100 for c in counts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

    # Bar chart of regimes
    bars = ax1.bar(labels, pcts, color=colors, alpha=0.85, edgecolor="white", width=0.55)
    for bar, pct, cnt in zip(bars, pcts, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{pct:.1f}%\n({cnt} nodes)", ha="center", va="bottom", fontsize=7.5)
    ax1.set_ylabel("% of nodes", fontsize=9)
    ax1.set_title("Flow Regime Distribution", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, max(pcts) * 1.35 + 5)
    ax1.tick_params(labelsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Backscatter pie
    bs_fwd  = max(0.0, 1.0 - backscatter_frac)
    bs_back = max(0.0, backscatter_frac)
    wedges, texts, autotexts = ax2.pie(
        [bs_fwd, bs_back],
        labels=["Forward scatter", "Backscatter"],
        colors=["#22c55e", "#f43f5e"],
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 8},
    )
    ax2.set_title("SGS Energy Transfer\n(Π = −τ:S)", fontsize=10, fontweight="bold")

    fig.tight_layout()
    return _to_png(fig)


def _dissipation_chart(dissipation: List[float]) -> io.BytesIO:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.array(dissipation)
    fig, ax = plt.subplots(figsize=(6.5, 3))
    pos = arr[arr >= 0]
    neg = arr[arr < 0]
    ax.hist(pos, bins=25, color="#22c55e", alpha=0.75, label=f"Forward ({len(pos)} nodes)", edgecolor="white")
    ax.hist(neg, bins=15, color="#f43f5e", alpha=0.75, label=f"Backscatter ({len(neg)} nodes)", edgecolor="white")
    ax.axvline(0, color="#475569", linewidth=1, linestyle="-")
    ax.set_xlabel("SGS dissipation Π = −τ:S", fontsize=9)
    ax.set_ylabel("Node count", fontsize=9)
    ax.set_title("SGS Dissipation Distribution", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _to_png(fig)


def _sensitivity_chart(parameter_ranking: list, base_peak_tau: float) -> io.BytesIO:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    params = [r.get("param", "") for r in parameter_ranking]
    deltas = [r.get("max_abs_delta", 0) for r in parameter_ranking]
    pct_impact = [d / base_peak_tau * 100 if base_peak_tau else 0 for d in deltas]

    order  = np.argsort(deltas)  # ascending so top is at top of horizontal bar
    params  = [params[i] for i in order]
    deltas  = [deltas[i] for i in order]
    pct_impact = [pct_impact[i] for i in order]

    fig, ax = plt.subplots(figsize=(6.5, max(2.8, len(params) * 0.55 + 0.8)))
    y = np.arange(len(params))
    bars = ax.barh(y, deltas, color="#6366f1", alpha=0.85, edgecolor="white", height=0.55)
    for bar, pct in zip(bars, pct_impact):
        ax.text(bar.get_width() + max(deltas) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=8, color="#475569")
    ax.set_yticks(y)
    ax.set_yticklabels(params, fontsize=9)
    ax.set_xlabel("Max |Δ peak τ|  (absolute)", fontsize=9)
    ax.set_title("Parameter Sensitivity Ranking (±10% / ±20% sweep)", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, max(deltas) * 1.2 if deltas else 1)
    fig.tight_layout()
    return _to_png(fig)


def _interaction_heatmap(imap: dict) -> io.BytesIO:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    grid    = np.array(imap.get("grid", []))
    levels  = imap.get("levels", [])
    param_a = imap.get("param_a", "Parameter A")
    param_b = imap.get("param_b", "Parameter B")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn_r", origin="lower",
                   vmin=float(np.min(grid)), vmax=float(np.max(grid)))
    tick_labels = [f"{l:+.0%}" for l in levels]
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(tick_labels, fontsize=7.5)
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels(tick_labels, fontsize=7.5)
    ax.set_xlabel(f"{param_b} perturbation", fontsize=9)
    ax.set_ylabel(f"{param_a} perturbation", fontsize=9)
    ax.set_title(f"Interaction Map: {param_a} × {param_b}\n(colour = peak τ)", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Peak |τ|", shrink=0.85)
    fig.tight_layout()
    return _to_png(fig)


def _perturbation_fan(param_name: str, perturbations: list, base_value: float, base_peak_tau: float) -> io.BytesIO:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    levels     = [p.get("level", 0) for p in perturbations]
    peak_taus  = [p.get("peak_tau", base_peak_tau) for p in perturbations]
    new_values = [p.get("new_value", base_value) for p in perturbations]

    fig, ax = plt.subplots(figsize=(5, 2.8))
    ax.plot([base_value] + new_values, [base_peak_tau] + peak_taus,
            "o-", color="#6366f1", linewidth=1.5, markersize=5)
    ax.axhline(base_peak_tau, color="#94a3b8", linewidth=0.8, linestyle="--", label="Base")
    ax.axvline(base_value,    color="#94a3b8", linewidth=0.8, linestyle="--")
    ax.set_xlabel(param_name, fontsize=9)
    ax.set_ylabel("Peak |τ|", fontsize=9)
    ax.set_title(f"Sensitivity: {param_name}", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _to_png(fig)


# ── Main generate function ───────────────────────────────────────────────────

def generate_pdf(
    geometry_type: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    cf4_data: Optional[Dict[str, Any]] = None,
    cf5_data: Optional[Dict[str, Any]] = None,
    tau_magnitudes: Optional[List[float]] = None,
    uncertainty_values: Optional[List[float]] = None,
) -> bytes:
    """Generate a multi-page PDF report. Returns raw PDF bytes."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak, Image, KeepTogether,
        )
    except ImportError:
        raise RuntimeError("reportlab not installed — run: pip install reportlab")

    W, H = A4  # 595 x 842 pt
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm,
        topMargin=2.2 * cm, bottomMargin=2.2 * cm,
    )

    styles = getSampleStyleSheet()
    TEAL    = colors.HexColor("#14b8a6")
    INDIGO  = colors.HexColor("#6366f1")
    SLATE   = colors.HexColor("#334155")
    LIGHT   = colors.HexColor("#f1f5f9")
    MID     = colors.HexColor("#e2e8f0")
    RED     = colors.HexColor("#dc2626")
    GREEN   = colors.HexColor("#22c55e")

    h1    = ParagraphStyle("h1",  parent=styles["Heading1"],  textColor=TEAL,   fontSize=26, spaceAfter=4,  fontName="Helvetica-Bold")
    h2    = ParagraphStyle("h2",  parent=styles["Heading2"],  textColor=SLATE,  fontSize=14, spaceBefore=16, spaceAfter=6, fontName="Helvetica-Bold")
    h3    = ParagraphStyle("h3",  parent=styles["Heading3"],  textColor=INDIGO, fontSize=11, spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold")
    sub   = ParagraphStyle("sub", parent=styles["Normal"],    textColor=SLATE,  fontSize=10, spaceAfter=2)
    body  = ParagraphStyle("body",parent=styles["Normal"],    fontSize=9.5,     spaceAfter=4, leading=14)
    small = ParagraphStyle("sm",  parent=styles["Normal"],    fontSize=8,       textColor=colors.HexColor("#64748b"), leading=12)
    foot  = ParagraphStyle("ft",  parent=styles["Normal"],    fontSize=7.5,     textColor=colors.HexColor("#94a3b8"), leading=11)

    IMG_W = 14 * cm
    IMG_H = 6  * cm
    IMG_H_WIDE = 6.5 * cm

    story = []

    def hr(color=MID, space=6):
        story.append(HRFlowable(width="100%", color=color, spaceAfter=space, spaceBefore=space))

    def embed(png_buf: io.BytesIO, w=IMG_W, h=IMG_H):
        story.append(Image(png_buf, width=w, height=h))
        story.append(Spacer(1, 0.3 * cm))

    def two_col_table(rows, col_widths=None):
        col_widths = col_widths or [8 * cm, 6 * cm]
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), TEAL),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME",      (0, 1), (0, -1), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID",          (0, 0), (-1, -1), 0.4, MID),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ]))
        return t

    # ════════════════════════════════════════════════════════
    # PAGE 1 — Cover / Summary
    # ════════════════════════════════════════════════════════
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph("SHEAR", h1))
    story.append(Paragraph("Turbulence Flow Intelligence — Analysis Report", sub))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", small))
    hr(TEAL, space=10)

    # Geometry config
    story.append(Paragraph("Geometry Configuration", h2))
    geo_rows = [["Parameter", "Value"], ["type", geometry_type]]
    for k, v in params.items():
        geo_rows.append([str(k), str(v)])
    story.append(two_col_table(geo_rows))
    story.append(Spacer(1, 0.4 * cm))

    # Key metrics
    if metrics:
        hr()
        story.append(Paragraph("Key Metrics", h2))
        met_rows = [["Metric", "Value"]]
        for k, v in metrics.items():
            if isinstance(v, float):
                formatted = f"{v:.5f}" if abs(v) < 0.01 else f"{v:.4f}"
            else:
                formatted = str(v)
            met_rows.append([str(k), formatted])
        story.append(two_col_table(met_rows))
        story.append(Spacer(1, 0.3 * cm))

    # Tau histogram
    if tau_magnitudes and len(tau_magnitudes) > 2:
        hr()
        story.append(Paragraph("SGS Stress Field Overview", h2))
        try:
            embed(_tau_histogram(tau_magnitudes), IMG_W, IMG_H)
        except Exception:
            pass

    # ════════════════════════════════════════════════════════
    # PAGE 2 — CF4 Diagnostics
    # ════════════════════════════════════════════════════════
    if cf4_data:
        story.append(PageBreak())
        story.append(Paragraph("CF4 — Flow Regime & Turbulence Diagnostics", h2))

        rl  = cf4_data.get("regime_labels", [])
        bs  = cf4_data.get("backscatter_frac", 0.0)
        unc = cf4_data.get("uncertainty", [])
        dis = cf4_data.get("dissipation", [])
        thr = cf4_data.get("regime_thresholds", [0, 0])

        # Summary table
        n_nodes = len(rl) or 1
        n_strain  = sum(1 for r in rl if r == 0)
        n_mixed   = sum(1 for r in rl if r == 1)
        n_vortex  = sum(1 for r in rl if r == 2)
        mean_unc  = sum(unc) / len(unc) if unc else 0
        max_unc   = max(unc) if unc else 0
        n_high_unc = sum(1 for u in unc if u > (max_unc * 0.75)) if unc else 0

        cf4_rows = [["CF4 Metric", "Value"]]
        cf4_rows += [
            ["Strain-dominated nodes", f"{n_strain}  ({n_strain/n_nodes:.1%})"],
            ["Mixed regime nodes",     f"{n_mixed}   ({n_mixed/n_nodes:.1%})"],
            ["Vortex-dominated nodes", f"{n_vortex}  ({n_vortex/n_nodes:.1%})"],
            ["Q-criterion threshold (33rd pct)", f"{thr[0]:.5f}"],
            ["Q-criterion threshold (66th pct)", f"{thr[-1]:.5f}"],
            ["Backscatter fraction (Π < 0)",     f"{bs:.2%}"],
            ["Mean uncertainty σ",               f"{mean_unc:.5f}"],
            ["High-uncertainty nodes (top 25%)", f"{n_high_unc}  ({n_high_unc/n_nodes:.1%})"],
        ]
        story.append(two_col_table(cf4_rows, [9 * cm, 6 * cm]))
        story.append(Spacer(1, 0.4 * cm))

        story.append(Paragraph(
            "The Q-criterion partitions nodes into three regimes: "
            "<b>strain-dominated</b> (Q &lt; Q₃₃) where viscous dissipation dominates, "
            "<b>mixed</b> (Q₃₃ ≤ Q &lt; Q₆₆), and <b>vortex-dominated</b> (Q ≥ Q₆₆) "
            "where rotation dominates. "
            "The <b>backscatter fraction</b> measures nodes with negative SGS energy transfer "
            "(Π = −τ:S &lt; 0); Smagorinsky produces zero backscatter by construction whereas "
            "V1 captures this physically.", body
        ))

        # Regime + backscatter chart
        if rl:
            try:
                embed(_regime_chart(rl, bs), IMG_W, IMG_H_WIDE)
            except Exception:
                pass

        # Uncertainty histogram
        if uncertainty_values and len(uncertainty_values) > 2:
            story.append(Paragraph("Epistemic Uncertainty", h3))
            story.append(Paragraph(
                "Per-node uncertainty is the L₂ norm of the variance across V1 ensemble samples. "
                "The top-25% threshold (dashed line) flags structurally sensitive regions where "
                "additional mesh refinement or more samples would reduce prediction error.", body
            ))
            try:
                embed(_uncertainty_histogram(uncertainty_values), IMG_W, IMG_H)
            except Exception:
                pass

        # SGS dissipation chart
        if dis:
            story.append(Paragraph("SGS Dissipation", h3))
            story.append(Paragraph(
                "Positive Π: kinetic energy flows from resolved to sub-grid scales (forward scatter). "
                "Negative Π: energy returns from sub-grid to resolved scales (backscatter). "
                "Smagorinsky purely dissipates; V1 allows both.", body
            ))
            try:
                embed(_dissipation_chart(dis), IMG_W, IMG_H)
            except Exception:
                pass

    # ════════════════════════════════════════════════════════
    # PAGE 3 — CF5 Parametric Sensitivity
    # ════════════════════════════════════════════════════════
    if cf5_data and cf5_data.get("parameter_ranking"):
        story.append(PageBreak())
        story.append(Paragraph("CF5 — Parametric Stress Explorer", h2))

        ranking      = cf5_data.get("parameter_ranking", [])
        base_peak    = cf5_data.get("base_peak_tau", 0.0)
        n_calls      = cf5_data.get("n_inference_calls", "—")
        total_ms     = cf5_data.get("total_time_ms", 0)
        imap_data    = cf5_data.get("interaction_map")

        story.append(Paragraph(
            f"<b>Base peak τ:</b> {base_peak:.5f} — "
            f"computed via {n_calls} inference calls in {total_ms/1000:.1f}s. "
            "Each numeric parameter was perturbed ±10% and ±20%; the delta in peak stress "
            "ranks parameters by their influence on turbulent loading.", body
        ))

        # Ranking table
        rank_rows = [["Rank", "Parameter", "Base value", "Max |Δ peak τ|", "% impact"]]
        for i, r in enumerate(ranking, 1):
            delta = r.get("max_abs_delta", 0)
            pct   = delta / base_peak * 100 if base_peak else 0
            rank_rows.append([
                str(i),
                r.get("param", ""),
                f"{r.get('base_value', 0):.4f}",
                f"{delta:.5f}",
                f"{pct:.1f}%",
            ])
        rt = Table(rank_rows, colWidths=[1.2*cm, 5*cm, 3.5*cm, 3.5*cm, 2.8*cm])
        rt.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), INDIGO),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8.5),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT]),
            ("GRID",          (0, 0), (-1, -1), 0.4, MID),
            ("TOPPADDING",    (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("ALIGN",         (0, 0), (0, -1), "CENTER"),
        ]))
        story.append(rt)
        story.append(Spacer(1, 0.4 * cm))

        # Sensitivity bar chart
        try:
            embed(_sensitivity_chart(ranking, base_peak), IMG_W, IMG_H_WIDE)
        except Exception:
            pass

        # Top-1 parameter fan chart
        if ranking:
            top = ranking[0]
            story.append(Paragraph(f"Most Influential Parameter: {top.get('param', '')}", h3))
            story.append(Paragraph(
                f"A ±20% change in <b>{top.get('param', '')}</b> shifts peak τ by "
                f"{top.get('max_abs_delta', 0):.5f} "
                f"({top.get('max_abs_delta', 0)/base_peak*100:.1f}% of base). "
                "The fan chart below shows how peak stress changes across the full perturbation range.", body
            ))
            try:
                embed(
                    _perturbation_fan(
                        top.get("param", ""),
                        top.get("perturbations", []),
                        top.get("base_value", 0),
                        base_peak,
                    ),
                    11 * cm, 5 * cm
                )
            except Exception:
                pass

        # Interaction map heatmap
        if imap_data:
            story.append(Paragraph(
                f"2D Interaction Map: {imap_data.get('param_a', '?')} × {imap_data.get('param_b', '?')}", h3
            ))
            story.append(Paragraph(
                "Joint perturbation of the top-two parameters. "
                "Red cells indicate higher stress; green cells lower. "
                "Non-diagonal variation signals parameter interaction effects.", body
            ))
            try:
                embed(_interaction_heatmap(imap_data), 12 * cm, 9 * cm)
            except Exception:
                pass

    # ════════════════════════════════════════════════════════
    # Final page — Technical notes
    # ════════════════════════════════════════════════════════
    story.append(PageBreak())
    story.append(Paragraph("Technical Notes", h2))
    story.append(Paragraph("<b>Model:</b> V1 SE(3)-Equivariant Graph Neural Network with Conditional Flow Matching (CFM). "
        "Trained on JHTDB isotropic1024coarse dataset (ε = 1.8×10⁻⁵, Re_λ ≈ 433). "
        "Architecture: EGNN with e3nn irreps (1×0e + 1×2e), 4 interaction layers, hidden dim 64.", body))
    story.append(Paragraph("<b>Inference:</b> Stochastic ODE integration from Gaussian prior x₀ → stress field x₁ "
        "via the learned vector field v_θ. Multiple samples give epistemic uncertainty via ensemble variance.", body))
    story.append(Paragraph("<b>Panel method (CF1):</b> Velocity gradient ∇u computed analytically on a structured "
        "Cartesian grid using geometry-specific panel boundary conditions. Grid resolution per side is set "
        "by grid_size parameter.", body))
    story.append(Paragraph("<b>Limitations:</b> V1 is trained on isotropic turbulence; accuracy degrades for "
        "highly compressible flows, strong rotation-dominated regimes, or geometries far outside the training distribution. "
        "OOD confidence is estimated via Mahalanobis distance in AF1.", body))
    story.append(Spacer(1, 0.5 * cm))

    # Perturbation detail tables for each parameter (if CF5)
    if cf5_data and cf5_data.get("parameter_ranking"):
        story.append(Paragraph("CF5 Perturbation Detail", h3))
        for r in cf5_data["parameter_ranking"]:
            perts = r.get("perturbations", [])
            if not perts:
                continue
            story.append(Paragraph(f"<b>{r.get('param', '')}:</b>", body))
            pt_rows = [["Level", "New value", "Peak τ", "Δ peak τ"]]
            for p in perts:
                pt_rows.append([
                    f"{p.get('level', 0):+.0%}",
                    f"{p.get('new_value', 0):.4f}",
                    f"{p.get('peak_tau', 0):.5f}",
                    f"{p.get('delta_peak_tau', 0):+.5f}",
                ])
            pt = Table(pt_rows, colWidths=[2.5*cm, 3.5*cm, 4*cm, 4*cm])
            pt.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
                ("TEXTCOLOR",     (0, 0), (-1, 0), INDIGO),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID",          (0, 0), (-1, -1), 0.3, MID),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ]))
            story.append(pt)
            story.append(Spacer(1, 0.25 * cm))

    story.append(Spacer(1, 1.0 * cm))
    hr()
    story.append(Paragraph(
        "SHEAR v1.0 · SE(3)-Equivariant Turbulence Flow Intelligence · "
        "V1 CFM trained on JHTDB isotropic1024coarse · ε = 1.8×10⁻⁵ · "
        "Report generated by SHEAR AF4 PDF Engine · All values are direct V1 inference results.",
        foot,
    ))

    doc.build(story)
    return buf.getvalue()
