"""
Laboratory Formulation PDF Certificate & Report Generator.
Uses ReportLab to generate publication-ready formulation reports and SOP documents.
"""

import io
import os
from datetime import datetime
from typing import Dict, Any, List, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


def generate_formulation_pdf_report(
    drug_name: str,
    drug_properties: Dict[str, Any],
    top_formulations: List[Dict[str, Any]],
    lab_sop: Dict[str, Any],
    polymer_system: str = "PLGA"
) -> bytes:
    """
    Builds a professional, comprehensive PDF laboratory formulation report.
    Returns bytes buffer of the generated PDF.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1B4965'),
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'ReportSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor('#555555'),
        alignment=TA_CENTER
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1B4965'),
        fontName='Helvetica-Bold',
        spaceBefore=8,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#222222')
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        textColor=colors.white,
        fontName='Helvetica-Bold',
        alignment=TA_CENTER
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        alignment=TA_CENTER
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("🧬 NANOFORMULA AI - FORMULATION OPTIMIZATION REPORT", title_style))
    elements.append(Paragraph(f"Department of Pharmaceutical Engineering & Technology | IIT (BHU) Varanasi<br/>Generated on: {datetime.now().strftime('%B %d, %Y - %H:%M:%S UTC')}", subtitle_style))
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1B4965'), spaceAfter=12))

    # 2. Executive Summary & Drug Identity
    elements.append(Paragraph("1. ACTIVE PHARMACEUTICAL INGREDIENT (API) PROFILE", section_heading))
    
    drug_table_data = [
        [
            Paragraph("<b>Target Drug:</b>", body_style), Paragraph(str(drug_name), body_style),
            Paragraph("<b>Mol. Weight:</b>", body_style), Paragraph(f"{drug_properties.get('mol_MW', 'N/A')} g/mol", body_style)
        ],
        [
            Paragraph("<b>Lipophilicity (LogP):</b>", body_style), Paragraph(f"{drug_properties.get('mol_logP', 'N/A')}", body_style),
            Paragraph("<b>Polar Surface (TPSA):</b>", body_style), Paragraph(f"{drug_properties.get('mol_TPSA', 'N/A')} Å²", body_style)
        ],
        [
            Paragraph("<b>Melting Point:</b>", body_style), Paragraph(f"{drug_properties.get('mol_melting_point', 'N/A')} °C", body_style),
            Paragraph("<b>H-Acceptors / Donors:</b>", body_style), Paragraph(f"{drug_properties.get('mol_Hacceptors', 'N/A')} / {drug_properties.get('mol_Hdonors', 'N/A')}", body_style)
        ]
    ]
    
    t_drug = Table(drug_table_data, colWidths=[110, 150, 120, 160])
    t_drug.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8F9FA')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E0E0E0')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(t_drug)
    elements.append(Spacer(1, 10))

    # 3. Recommended Pareto Formulations Table
    elements.append(Paragraph(f"2. TOP RECOMMENDED {polymer_system.upper()} FORMULATIONS (PARETO-RANKED)", section_heading))
    
    if polymer_system.startswith("PLGA"):
        headers = ["Rank", "PLGA MW", "LA/GA", "D/P Ratio", "Surf. %", "Pred. Size", "Pred. EE%", "Pred. LC%", "AD Domain"]
        table_rows = [[Paragraph(f"<b>{h}</b>", table_header_style) for h in headers]]
        
        for i, row in enumerate(top_formulations):
            ci_size = f"{row.get('pred_size', 0):.1f} ± {row.get('pred_size_std', 0):.1f} nm"
            ci_ee = f"{row.get('pred_EE', 0):.1f} ± {row.get('pred_ee_std', 0):.1f}%"
            ad_status = row.get('ad_status', 'In Domain').split(' ')[0]
            
            table_rows.append([
                Paragraph(f"<b>#{i+1}</b>", table_cell_style),
                Paragraph(f"{row.get('polymer_MW', 0):.1f} kDa", table_cell_style),
                Paragraph(f"{row.get('LA/GA', 0):.2f}", table_cell_style),
                Paragraph(f"{row.get('drug/polymer', 0):.4f}", table_cell_style),
                Paragraph(f"{row.get('surfactant_concentration', 0):.2f}%", table_cell_style),
                Paragraph(ci_size, table_cell_style),
                Paragraph(ci_ee, table_cell_style),
                Paragraph(f"{row.get('pred_LC', 0):.1f}%", table_cell_style),
                Paragraph(ad_status, table_cell_style),
            ])
        col_widths = [30, 55, 45, 55, 45, 85, 75, 55, 95]
    else:
        headers = ["Rank", "Chitosan MW", "CS Conc", "TPP Conc", "CS:TPP", "Pred. Size", "Pred. PDI", "Zeta (mV)", "AD Domain"]
        table_rows = [[Paragraph(f"<b>{h}</b>", table_header_style) for h in headers]]
        for i, row in enumerate(top_formulations):
            table_rows.append([
                Paragraph(f"<b>#{i+1}</b>", table_cell_style),
                Paragraph(f"{row.get('chitosan_MW', 0):.0f} kDa", table_cell_style),
                Paragraph(f"{row.get('chitosan_conc', 0):.2f} mg/mL", table_cell_style),
                Paragraph(f"{row.get('TPP_conc', 0):.2f} mg/mL", table_cell_style),
                Paragraph(f"{row.get('chitosan_TPP_ratio', 0):.2f}", table_cell_style),
                Paragraph(f"{row.get('pred_size', 0):.1f} nm", table_cell_style),
                Paragraph(f"{row.get('pred_PDI', 0):.2f}", table_cell_style),
                Paragraph(f"+{row.get('pred_zeta', 0):.1f} mV", table_cell_style),
                Paragraph(row.get('ad_status', 'In Domain').split(' ')[0], table_cell_style),
            ])
        col_widths = [30, 65, 60, 60, 45, 70, 55, 60, 95]

    t_recs = Table(table_rows, colWidths=col_widths)
    t_recs.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1B4965')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#B0BEC5')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F4F6F7')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    elements.append(t_recs)
    elements.append(Spacer(1, 12))

    # 4. Standard Operating Procedure (SOP)
    elements.append(Paragraph(f"3. STEP-BY-STEP LABORATORY PROTOCOL (Batch Recipe: {lab_sop.get('recipe', {}).get('batch_volume_ml', 10)} mL)", section_heading))
    elements.append(Paragraph(f"<b>Fabrication Technique:</b> {lab_sop.get('method', 'Nanoprecipitation / Solvent Evaporation')}", body_style))
    elements.append(Spacer(1, 4))

    recipe_data = lab_sop.get('recipe', {})
    recipe_rows = []
    for k, v in recipe_data.items():
        k_clean = k.replace('_', ' ').title()
        recipe_rows.append([Paragraph(f"<b>{k_clean}:</b>", body_style), Paragraph(str(v), body_style)])

    # Split recipe into 2-column key-value pairs
    half = (len(recipe_rows) + 1) // 2
    rec_table_data = []
    for i in range(half):
        left = recipe_rows[i]
        right = recipe_rows[i + half] if (i + half) < len(recipe_rows) else [Paragraph("", body_style), Paragraph("", body_style)]
        rec_table_data.append([left[0], left[1], right[0], right[1]])

    t_recipe = Table(rec_table_data, colWidths=[120, 150, 120, 150])
    t_recipe.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#EBF5FB')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#AED6F1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    elements.append(t_recipe)
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("<b>Procedure Execution Steps:</b>", body_style))
    for step in lab_sop.get('steps', []):
        elements.append(Paragraph(step, body_style))
        elements.append(Spacer(1, 3))

    # Footer note
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#CCCCCC'), spaceAfter=6))
    elements.append(Paragraph(
        "<i>Disclaimer: NanoFormula AI predictions are derived from ensemble machine learning with 10-fold cross-validation. "
        "Laboratory users should observe standard Biosafety Level (BSL) and personal protective equipment protocols when handling organic solvents and APIs.</i>",
        subtitle_style
    ))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
