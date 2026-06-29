"""
AF4 — PDF Report generation via ReportLab.

Multi-page report: cover, stress field summary, CF4 diagnostics, CF5 explorer results.
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional


def generate_pdf(
    geometry_type: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    cf4_data: Optional[Dict[str, Any]] = None,
    cf5_data: Optional[Dict[str, Any]] = None,
) -> bytes:
    """
    Generate a multi-page PDF report.

    Returns raw PDF bytes ready to serve as a file download.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak,
        )
    except ImportError:
        raise RuntimeError("reportlab not installed — run: pip install reportlab")

    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
    )
    styles = getSampleStyleSheet()
    story  = []

    TEAL   = colors.HexColor("#14b8a6")
    SLATE  = colors.HexColor("#334155")
    LIGHT  = colors.HexColor("#e2e8f0")

    h1  = ParagraphStyle("h1",  parent=styles["Heading1"],  textColor=TEAL,  fontSize=22, spaceAfter=6)
    h2  = ParagraphStyle("h2",  parent=styles["Heading2"],  textColor=SLATE, fontSize=14, spaceBefore=14, spaceAfter=4)
    sub = ParagraphStyle("sub", parent=styles["Normal"],    textColor=SLATE, fontSize=10, spaceAfter=2)
    body= ParagraphStyle("body",parent=styles["Normal"],    fontSize=10, spaceAfter=4, leading=14)

    def hr():
        story.append(HRFlowable(width="100%", color=LIGHT, spaceAfter=8, spaceBefore=8))

    # ---- Cover page ----
    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("SHEAR", h1))
    story.append(Paragraph("Turbulence Flow Intelligence — Analysis Report", sub))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", sub))
    hr()

    # Geometry summary
    story.append(Paragraph("Geometry Configuration", h2))
    story.append(Paragraph(f"<b>Type:</b> {geometry_type}", body))
    for k, v in params.items():
        story.append(Paragraph(f"<b>{k}:</b> {v}", body))

    hr()

    # Key metrics
    story.append(Paragraph("Key Metrics", h2))
    metric_rows = [["Metric", "Value"]]
    for k, v in metrics.items():
        if isinstance(v, float):
            metric_rows.append([k, f"{v:.4f}"])
        else:
            metric_rows.append([k, str(v)])

    t = Table(metric_rows, colWidths=[8*cm, 6*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(t)

    # ---- CF4 diagnostics page ----
    if cf4_data:
        story.append(PageBreak())
        story.append(Paragraph("CF4 — Flow Regime Diagnostics", h2))

        story.append(Paragraph(
            f"<b>Backscatter fraction:</b> {cf4_data.get('backscatter_frac', 0):.1%} of nodes show "
            f"negative SGS dissipation (Π &lt; 0). Smagorinsky produces zero backscatter by construction; "
            f"V1 models this physically.", body
        ))
        story.append(Paragraph(
            f"<b>Regime thresholds:</b> Q₃₃ = {cf4_data.get('regime_thresholds', [0,0])[0]:.4f}, "
            f"Q₆₆ = {cf4_data.get('regime_thresholds', [0,0])[-1]:.4f}", body
        ))

        unc_list = cf4_data.get("uncertainty", [])
        if unc_list:
            import statistics
            story.append(Paragraph(
                f"<b>Uncertainty (mean):</b> {statistics.mean(unc_list):.4f} — "
                f"top 25% flagged as high-uncertainty zones.", body
            ))

    # ---- CF5 explorer page ----
    if cf5_data:
        story.append(PageBreak())
        story.append(Paragraph("CF5 — Parametric Stress Explorer", h2))
        story.append(Paragraph(
            f"<b>Base peak τ:</b> {cf5_data.get('base_peak_tau', 0):.4f} — "
            f"inference calls: {cf5_data.get('n_inference_calls', '—')}", body
        ))

        ranking = cf5_data.get("parameter_ranking", [])
        if ranking:
            story.append(Paragraph("<b>Parameter sensitivity ranking:</b>", body))
            rank_rows = [["Parameter", "Base value", "Max |Δ peak τ|"]]
            for r in ranking:
                rank_rows.append([
                    r.get("param", ""),
                    str(round(r.get("base_value", 0), 4)),
                    f"{r.get('max_abs_delta', 0):.4f}",
                ])
            t2 = Table(rank_rows, colWidths=[6*cm, 5*cm, 5*cm])
            t2.setStyle(TableStyle([
                ("BACKGROUND",    (0, 0), (-1, 0), TEAL),
                ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
                ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",      (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING",    (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t2)

        imap = cf5_data.get("interaction_map")
        if imap:
            story.append(Spacer(1, 0.3*cm))
            story.append(Paragraph(
                f"<b>2D Interaction Map</b> — {imap.get('param_a', '?')} × {imap.get('param_b', '?')}: "
                f"joint effect on peak τ across ±20% perturbations.", body
            ))

    # ---- Footer note ----
    story.append(Spacer(1, 1.5*cm))
    hr()
    story.append(Paragraph(
        "Language note: all values reported are exact V1 inference results. "
        "No extrapolation beyond tested perturbation levels. "
        "V1 SE(3)-Equivariant CFM, JHTDB isotropic1024coarse, ε = 1.8×10⁻⁵.",
        ParagraphStyle("footer", parent=styles["Normal"], fontSize=8, textColor=colors.HexColor("#94a3b8")),
    ))

    doc.build(story)
    return buf.getvalue()
