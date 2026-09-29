"""
Publication Figure Generator for Meta-Analysis, Parity, Bland-Altman, and Hansen Compatibility.
Generates 300 DPI high-resolution figures for submission to peer-reviewed Q1 journals.
"""

import os
import sys

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath('.'))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse

from nanoformula.validation.literature_validation import LiteratureValidator, PUBLISHED_LITERATURE_CASE_STUDIES
from nanoformula.thermodynamics.hsp_engine import HSPEngine, POLYMER_HSP_DATABASE

# Set high-quality publication styling
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9


def generate_publication_figures(output_dir: str = "paper_materials/figures"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model bundle
    bundle_path = "saved_models/nanoformula_models_bundle.pkl"
    if not os.path.exists(bundle_path):
        print(f"Error: {bundle_path} not found.")
        return

    bundle = joblib.load(bundle_path)
    validator = LiteratureValidator(bundle)
    df_val = validator.run_literature_validation()
    meta_stats = validator.compute_meta_analysis_statistics(df_val)

    # ----------------------------------------------------
    # Figure: 4-Panel Multi-Study Meta-Analysis & Validation
    # ----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.patch.set_facecolor('white')

    # Panel A: Particle Size Parity Plot
    ax1 = axes[0, 0]
    exp_s = df_val["Exp. Size (nm)"].values
    pred_s = df_val["AI Pred. Size (nm)"].values
    exp_err = df_val["Exp. Size Error (nm)"].values

    min_val = min(exp_s.min(), pred_s.min()) * 0.85
    max_val = max(exp_s.max(), pred_s.max()) * 1.10
    
    # Parity lines & error bounds
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='Ideal Parity (y = x)')
    ax1.fill_between([min_val, max_val], [min_val * 0.90, max_val * 0.90], [min_val * 1.10, max_val * 1.10],
                     color='#2b5c8f', alpha=0.12, label='±10% Error Envelope')

    ax1.errorbar(exp_s, pred_s, xerr=exp_err, fmt='o', color='#1f77b4', ecolor='#888888',
                 elinewidth=1.2, capsize=3, markersize=8, markeredgecolor='black', markeredgewidth=0.8,
                 label=f'Published Studies (N={len(df_val)})')

    ax1.set_xlim(min_val, max_val)
    ax1.set_ylim(min_val, max_val)
    ax1.set_xlabel("Experimental Particle Size (nm)")
    ax1.set_ylabel("NanoFormula AI Predicted Size (nm)")
    ax1.set_title(f"A. External Size Validation (R² = {meta_stats['size_metrics']['R2']:.3f}, MAPE = {meta_stats['size_metrics']['mape_percent']:.1f}%)", fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left', frameon=True)

    # Panel B: Bland-Altman Agreement Plot for Particle Size
    ax2 = axes[0, 1]
    diff_s = pred_s - exp_s
    mean_s = (pred_s + exp_s) / 2.0
    bias = meta_stats['size_metrics']['bland_altman_bias_nm']
    loa_up = meta_stats['size_metrics']['bland_altman_loa_upper']
    loa_lo = meta_stats['size_metrics']['bland_altman_loa_lower']

    ax2.scatter(mean_s, diff_s, color='#e6550d', s=60, edgecolors='black', linewidth=0.8, zorder=3)
    ax2.axhline(bias, color='#d95f02', linestyle='-', lw=2, label=f'Mean Bias: {bias:+.1f} nm')
    ax2.axhline(loa_up, color='#7570b3', linestyle='--', lw=1.5, label=f'+1.96 SD: {loa_up:+.1f} nm')
    ax2.axhline(loa_lo, color='#7570b3', linestyle='--', lw=1.5, label=f'-1.96 SD: {loa_lo:+.1f} nm')
    ax2.fill_between([mean_s.min() - 10, mean_s.max() + 10], loa_lo, loa_up, color='#7570b3', alpha=0.08)

    ax2.set_xlim(mean_s.min() - 10, mean_s.max() + 10)
    ax2.set_xlabel("Mean Particle Size (Experimental + AI) / 2 (nm)")
    ax2.set_ylabel("Difference: AI Predicted - Experimental (nm)")
    ax2.set_title("B. Bland-Altman 95% Agreement Analysis", fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', frameon=True)

    # Panel C: Entrapment Efficiency Parity Plot
    ax3 = axes[1, 0]
    df_ee = df_val.dropna(subset=["Exp. EE (%)", "AI Pred. EE (%)"])
    exp_ee = df_ee["Exp. EE (%)"].values
    pred_ee = df_ee["AI Pred. EE (%)"].values

    ax3.plot([30, 100], [30, 100], 'k--', lw=1.5, label='Ideal Parity (y = x)')
    ax3.fill_between([30, 100], [27, 90], [33, 110], color='#2ca02c', alpha=0.12, label='±10% Error Envelope')
    ax3.scatter(exp_ee, pred_ee, color='#2ca02c', s=70, edgecolors='black', linewidth=0.8,
                label=f'Drug-Loaded Formulations (N={len(df_ee)})', zorder=3)

    ax3.set_xlim(30, 100)
    ax3.set_ylim(30, 100)
    ax3.set_xlabel("Experimental Entrapment Efficiency (%)")
    ax3.set_ylabel("NanoFormula AI Predicted EE (%)")
    ax3.set_title(f"C. Entrapment Efficiency Generalization (R² = {meta_stats['ee_metrics']['R2']:.3f}, MAPE = {meta_stats['ee_metrics']['mape_percent']:.1f}%)", fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    ax3.legend(loc='upper left', frameon=True)

    # Panel D: Thermodynamic Hansen Miscibility Space
    ax4 = axes[1, 1]
    # Plot standard polymers in (delta_P vs delta_H) space
    polymers = list(POLYMER_HSP_DATABASE.keys())
    p_dp = [POLYMER_HSP_DATABASE[p]["delta_P"] for p in polymers]
    p_dh = [POLYMER_HSP_DATABASE[p]["delta_H"] for p in polymers]
    p_r0 = [POLYMER_HSP_DATABASE[p]["interaction_radius_R0"] for p in polymers]

    for p, dp, dh, r0 in zip(polymers, p_dp, p_dh, p_r0):
        circle = plt.Circle((dp, dh), r0 * 0.6, color='#3182bd', alpha=0.15, linestyle='--', ec='#08519c', lw=1.2)
        ax4.add_patch(circle)
        ax4.scatter(dp, dh, color='#08519c', s=80, marker='s', zorder=4)
        ax4.text(dp + 0.3, dh + 0.3, p, fontsize=8.5, fontweight='semibold', color='#08306b')

    # Add sample drug coordinates
    sample_drugs = [
        ("Paclitaxel", 7.9, 10.2, "#de2d26"),
        ("Curcumin", 8.4, 11.8, "#de2d26"),
        ("Doxorubicin", 12.1, 14.8, "#de2d26"),
        ("Dexamethasone", 6.8, 8.5, "#de2d26"),
        ("5-Fluorouracil", 9.5, 12.0, "#de2d26")
    ]
    for dname, d_p, d_h, col in sample_drugs:
        ax4.scatter(d_p, d_h, color=col, s=80, marker='^', edgecolors='black', zorder=5)
        ax4.text(d_p + 0.3, d_h - 0.4, dname, fontsize=8.5, color='#a50f15', fontstyle='italic')

    ax4.set_xlim(0, 18)
    ax4.set_ylim(2, 22)
    ax4.set_xlabel("Polar Hansen Parameter δP (MPa⁰˙⁵)")
    ax4.set_ylabel("Hydrogen Bonding Parameter δH (MPa⁰˙⁵)")
    ax4.set_title("D. Hansen Thermodynamic Miscibility Map", fontweight='bold')
    ax4.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "meta_analysis_validation_300dpi.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Successfully generated 300 DPI publication figure at: {fig_path}")
    return fig_path


if __name__ == "__main__":
    generate_publication_figures()
