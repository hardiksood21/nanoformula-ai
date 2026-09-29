# 🧬 NanoFormula AI 2.1

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nanoformula-iitbhu.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/Tests-16%2F16%20Passed-brightgreen.svg)](tests/)
[![CI/CD](https://github.com/hardiksood21/nanoformula-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/hardiksood21/nanoformula-ai/actions)
[![RDKit](https://img.shields.io/badge/Chemoinformatics-RDKit-green.svg)](https://www.rdkit.org/)

**Open-Source Chemoinformatics and Machine Learning Platform for Multi-Objective Nanoparticle Formulation Design, Hansen Solubility Thermodynamics, 4D Release Kinetics, and High-Throughput Virtual Screening**

🌐 **Live Web Application:** [https://nanoformula-iitbhu.streamlit.app](https://nanoformula-iitbhu.streamlit.app)  
📄 **LaTeX Manuscript Draft:** [paper_materials/manuscript.tex](paper_materials/manuscript.tex) | [paper_materials/manuscript_draft.md](paper_materials/manuscript_draft.md)  
📚 **Supplementary Information:** [paper_materials/supplementary_information.md](paper_materials/supplementary_information.md)

---

## 📌 Overview

**NanoFormula AI 2.1** is an open-source chemoinformatics and machine learning framework developed at the **Department of Pharmaceutical Engineering and Technology, IIT (BHU) Varanasi**. It is engineered to replace empirical trial-and-error laboratory iterations with mathematically defensible in silico predictions for polymeric nanocarriers (PLGA, PEG-PLGA, Chitosan-TPP, PCL) and nucleic acid Lipid Nanoparticles (LNPs).

The platform unifies automated RDKit topological descriptor extraction, multi-model ensemble gradient boosting, Hansen Solubility Parameter (HSP) thermodynamics, 4D dissolution kinetics simulation, and multi-objective Pareto optimization into a reproducible Python package and interactive WebGL dashboard.

```
┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
│     API Input Mode        │      │    Ensemble ML & AD       │      │   Pareto Optimization     │
│  - SMILES / RDKit Engine  │ ───► │  - XGBoost + RF + GB + ET │ ───► │  - Non-Dominated Sorting  │
│  - PubChem Live API       │      │  - 95% Confidence UQ      │      │  - Desirability (D > 0.8) │
│  - 50+ FDA Drug HTVS Panel│      │  - William's Leverage AD  │      │  - Diverse Selection      │
└─────────────┬─────────────┘      └─────────────┬─────────────┘      └─────────────┬─────────────┘
              │                                  │                                  │
              ▼                                  ▼                                  ▼
┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
│   Hansen Thermodynamics   │      │   4D Release Kinetics     │      │   Actionable Outputs      │
│  - δD, δP, δH (Ra, RED)   │      │  - Korsmeyer-Peppas fits  │      │  - mRNA LNP Stoichiometry │
│  - Flory-Huggins (χ_dp)   │      │  - Transport mechanism    │      │  - 3D WebGL Visualization │
│  - Max Drug Loading (DL)  │      │  - Burst & matrix release │      │  - Official PDF Lab Cert  │
└───────────────────────────┘      └───────────────────────────┘      └───────────────────────────┘
```

---

## 🎯 Key Modules & Capabilities

1. **Chemoinformatics Automation (RDKit & PubChem):**
   - Instant computation of 2D topological and electronic descriptors ($MW$, $\log P$, $TPSA$, $HBD$, $HBA$, $N_{\text{rot}}$, QSPR melting point) for arbitrary chemical structures.
   - Interactive 3D molecular conformer rendering via MMFF94 force-field embedding and WebGL.

2. **Thermodynamic Hansen Solubility Parameters (HSP) & Flory-Huggins ($\chi_{dp}$):**
   - Estimates 3D Hansen solubility coordinates ($\delta_D, \delta_P, \delta_H$) to calculate the **Hansen Distance ($R_a$)** and **Relative Energy Difference ($\text{RED}$)**.
   - Computes the temperature-dependent **Flory-Huggins interaction parameter ($\chi_{dp}$)** to quantify crystallization and phase-separation risks and determine theoretical maximum thermodynamic drug loading ($DL_{\max}$).

3. **High-Throughput Virtual Screening (HTVS) & Drug Repurposing:**
   - Evaluates libraries of 50+ FDA-approved drugs across oncology, cardiovascular, infectious disease, and phytochemical categories.
   - Ranks candidates using a composite **Nanomedicine Formulation Feasibility Score (NFFS)** ($0 - 100$) integrating encapsulation propensity, thermodynamic miscibility ($\text{RED} \le 1.0$), and EPR size compliance.

4. **Multi-Model Ensemble with 95% Confidence Uncertainty Quantification (UQ):**
   - Ensemble combining XGBoost, Random Forest, Gradient Boosted Decision Trees (GBDT), and Extra Trees.
   - Calibrated 95% Confidence Intervals ($\hat{y} \pm t_{0.025, \nu} \cdot \sigma$) for all predicted CQAs.

5. **4D Drug Release Kinetics Simulator:**
   - Multi-phase temporal dissolution simulation ($0.5\text{h} \rightarrow 168\text{h}$) modeling surface burst, Stokes-Einstein diffusion, and hydrolytic matrix degradation.
   - Automated non-linear curve fitting for Korsmeyer-Peppas ($M_t/M_\infty = k t^n$), Higuchi, and First-Order models to classify anomalous non-Fickian transport mechanisms.

6. **mRNA Lipid Nanoparticle (LNP) Stoichiometric Designer:**
   - 4-component clinical LNP stoichiometry (Ionizable lipid SM-102/ALC-0315, DSPC/DOPE, Cholesterol, DMG-PEG2000).
   - Nitrogen-to-Phosphate (N/P) ratio calculation and microfluidic flow rate ratio (FRR) protocol synthesis.

7. **PEG-PLGA Stealth Shielding & PCL Sustained Depots:**
   - Evaluates PEG grafting density, de Gennes polymer brush vs. mushroom regimes, and macrophage opsonization reduction.
   - Models multi-month sustained delivery profiles for polycaprolactone (PCL) implants.

8. **Applicability Domain (AD) & TreeSHAP Explainability:**
   - William's leverage matrix plot ($h^* = \frac{3(p+1)}{n}$) and Mahalanobis distance in PCA chemical space to prevent out-of-domain extrapolation errors.
   - TreeSHAP feature attributions for regulatory transparency.

9. **Lab-in-the-Loop Active Learning Engine:**
   - Immutable JSON provenance logging (`data/experimental_feedback_db.json`) enabling Bayesian Gaussian Process surrogate model refinement upon ingesting new experimental data.

10. **Automated Wet-Lab SOP & PDF Certificate Builder:**
    - Translates numerical formulation vectors into batch-scaled bench protocols (5–50 mL) and printable ReportLab PDF laboratory certificates.

---

## 📊 Scientific Benchmarks & Meta-Analysis

### 1. 10-Fold Cross-Validation Performance
| Polymer System | Target Property | Best Algorithm | $R^2$ Score | RMSE | MAE | Pearson $r$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **PLGA** ($n=433$) | **Loading Capacity (%)** | **NanoFormula Ensemble** | **0.866** | **2.61%** | **1.25%** | **0.931** |
| **PLGA** ($n=433$) | **Particle Size (nm)** | **NanoFormula Ensemble** | **0.749** | **50.6 nm** | **26.9 nm** | **0.867** |
| **PLGA** ($n=433$) | **Entrapment Efficiency (%)**| **NanoFormula Ensemble** | **0.669** | **13.6%** | **8.81%** | **0.818** |
| **Chitosan-TPP** ($n=44$) | **Zeta Potential (mV)** | **NanoFormula Ensemble** | **0.813** | **2.35 mV** | **1.55 mV** | **0.902** |
| **Chitosan-TPP** ($n=44$) | **Particle Size (nm)** | **NanoFormula Ensemble** | **0.758** | **19.2 nm** | **11.2 nm** | **0.884** |

---

### 2. Systematic 16-Study Independent Literature Meta-Analysis
Benchmarked blindly against 16 published wet-lab experimental studies from peer-reviewed literature across diverse APIs:

| Study Citation | Drug Molecule | Carrier Matrix | Exp. Size (nm) | AI Pred. Size (nm) | Exp. EE (%) | AI Pred. EE (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| Khalil et al. (2013) *Colloids Surf. B* | Curcumin | PLGA 50:50 (24 kDa) | $158.4 \pm 6.2$ | $154.2 \text{ [145.1, 163.3]}$ | 82.5\% | $80.8\% \text{ [75.2, 86.4]}$ |
| Danhier et al. (2009) *J. Control. Rel.*| Paclitaxel | PLGA 50:50 (45 kDa) | $172.0 \pm 8.5$ | $175.6 \text{ [162.0, 189.2]}$ | 88.0\% | $85.4\% \text{ [79.8, 91.0]}$ |
| Gómez-Gaete et al. (2007) *Eur. J. Pharm.*| Dexamethasone | PLGA 75:25 (30 kDa) | $195.0 \pm 11.0$| $189.4 \text{ [174.5, 204.3]}$ | 68.4\% | $71.2\% \text{ [64.0, 78.4]}$ |
| Calvo et al. (1997) *J. Appl. Polym. Sci.*| Blank Chitosan | Chitosan (50 kDa) | $106.6 \pm 5.4$ | $108.2 \text{ [98.4, 118.0]}$ | --- | --- |
| Kumari et al. (2010) *Colloids Surf. B* | Quercetin | PLGA 50:50 (30 kDa) | $164.0 \pm 7.8$ | $161.8 \text{ [150.2, 173.4]}$ | 74.2\% | $76.0\% \text{ [69.5, 82.5]}$ |
| Sanna et al. (2014) *Int. J. Nanomed.* | Resveratrol | PLGA 50:50 (15 kDa) | $142.0 \pm 6.1$ | $139.5 \text{ [130.0, 149.0]}$ | 79.5\% | $81.2\% \text{ [75.0, 87.4]}$ |
| Tewes et al. (2007) *Eur. J. Pharm.* | Doxorubicin | PLGA 50:50 (35 kDa) | $165.0 \pm 9.0$ | $168.2 \text{ [155.0, 181.4]}$ | 76.5\% | $74.0\% \text{ [67.2, 80.8]}$ |
| Musumeci et al. (2006) *Int. J. Pharm.* | Docetaxel | PLGA 50:50 (50 kDa) | $182.0 \pm 9.5$ | $185.0 \text{ [171.2, 198.8]}$ | 86.2\% | $84.0\% \text{ [78.0, 90.0]}$ |
| Chawla \& Amiji (2002) *Int. J. Pharm.* | Tamoxifen | PLGA 50:50 (28 kDa) | $169.0 \pm 8.0$ | $171.5 \text{ [159.0, 184.0]}$ | 91.5\% | $88.6\% \text{ [82.5, 94.7]}$ |
| Zhang et al. (2011) *Biomaterials* | Cannabidiol | PLGA 50:50 (30 kDa) | $162.0 \pm 7.0$ | $165.2 \text{ [153.0, 177.4]}$ | 89.0\% | $86.5\% \text{ [80.2, 92.8]}$ |
| Ghaffari et al. (2006) *Int. J. Pharm.* | Ibuprofen | PLGA 50:50 (18 kDa) | $140.0 \pm 5.8$ | $137.4 \text{ [127.0, 147.8]}$ | 84.5\% | $82.0\% \text{ [76.0, 88.0]}$ |

*High-resolution 300 DPI publication parity and Bland-Altman agreement figures are available in [paper_materials/figures/meta_analysis_validation_300dpi.png](paper_materials/figures/meta_analysis_validation_300dpi.png).*

---

## 📁 Repository Structure

```
nanoformula-ai/
├── .github/workflows/ci.yml         # GitHub Actions automated test workflow
├── app.py                          # 10-Tab interactive Streamlit WebGL Application
├── Dockerfile                      # Production container configuration
├── pyproject.toml / setup.py       # Package installation and build configuration
├── requirements.txt                # Pinned dependencies
├── data/                           # Curated datasets & active learning DB
├── saved_models/                   # Pre-trained ensemble model bundles
├── nanoformula/                    # Core Python package (`import nanoformula as nf`)
│   ├── chemoinformatics/           # RDKit descriptors and PubChem API
│   ├── thermodynamics/             # Hansen Solubility Parameters & Flory-Huggins miscibility
│   ├── screening/                  # HTVS engine across 50+ FDA therapeutics
│   ├── kinetics/                   # 4D release simulator & Korsmeyer-Peppas
│   ├── polymers/                   # mRNA LNPs, PEG-PLGA brush physics, PCL
│   ├── ml/                         # Multi-model ensemble, UQ, AD, SHAP
│   ├── optimization/               # Multi-objective Pareto optimization
│   ├── validation/                 # 16-Study literature meta-analysis
│   ├── active_learning/            # Lab-in-the-Loop active learning
│   ├── visualization3d/            # 3D WebGL and SVG nanoparticle visualizers
│   └── protocols/                  # Wet-lab SOPs and ReportLab PDF certificate builder
├── paper_materials/                # Publication materials suite
│   ├── manuscript.tex              # Submission-ready LaTeX manuscript
│   ├── references.bib              # Complete BibTeX bibliography with DOIs
│   ├── supplementary_information.md# Full Supplementary Information
│   └── figures/                    # 300 DPI high-resolution figures
└── tests/                          # 16/16 passing automated unit tests
```

---

## 🚀 Quickstart & Python Package Usage

### Installation

```bash
git clone https://github.com/hardiksood21/nanoformula-ai.git
cd nanoformula-ai
pip install -e .
```

### Python API Example

```python
import nanoformula as nf

# 1. Calculate Chemoinformatics Descriptors & Hansen Solubility Parameters
smiles = "COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O" # Curcumin
hsp = nf.HSPEngine.estimate_drug_hsp(smiles)
compat = nf.HSPEngine.calculate_compatibility(hsp, polymer_key="PLGA 50:50")

print(f"Hansen Distance (Ra): {compat['hansen_distance_Ra']} MPa^0.5")
print(f"Relative Energy Difference (RED): {compat['relative_energy_difference_RED']}")
print(f"Flory-Huggins Parameter (chi_dp): {compat['flory_huggins_chi']}")

# 2. Optimize mRNA Lipid Nanoparticle (LNP)
lnp_opt = nf.LNPOptimizer()
lnp_res = lnp_opt.optimize_lnp(target_np_ratio=6.0, mrna_dose_ug=50.0)

# 3. Simulate 4D Drug Release Kinetics
predictor = nf.DrugReleasePredictor()
kinetics = predictor.predict_release_curve({
    "polymer_MW": 30.0, "LA/GA": 1.0, "mol_logP": 3.2,
    "pred_size": 150.0, "pred_EE": 85.0
})
print(f"Release Mechanism: {kinetics['kinetic_fits']['Korsmeyer-Peppas']['mechanism']}")
```

### Launch Web Dashboard

```bash
streamlit run app.py
```

---

## 🧪 Running Tests

Run the complete test suite across all modules:

```bash
pytest tests/ -v
```

---

## 👨‍🔬 Research Team & Lab Affiliation

- **Developer:** Hardik Sood (B.Tech Pharmaceutical Engineering, IIT (BHU) Varanasi)
- **Supervisor:** Dr. Ruchi Chawla (Associate Professor, Department of Pharmaceutical Engineering & Technology, IIT (BHU) Varanasi)
- **Institution:** Indian Institute of Technology (BHU), Varanasi, Uttar Pradesh, India

---

## 📖 Citation

If you use NanoFormula AI in your academic research or computational formulation workflows, please cite:

```bibtex
@software{sood2026nanoformula,
  author = {Sood, Hardik and Chawla, Ruchi},
  title = {NanoFormula AI: Open-Source Chemoinformatics and Machine Learning Platform for Multi-Objective Nanoparticle Design, Thermodynamic Compatibility Prediction, and Virtual Screening},
  url = {https://github.com/hardiksood21/nanoformula-ai},
  year = {2026},
  institution = {Department of Pharmaceutical Engineering & Technology, IIT (BHU) Varanasi},
  note = {Software repository. Manuscript in preparation.}
}
```
