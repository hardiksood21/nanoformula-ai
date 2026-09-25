# 🧬 NanoFormula AI 2.0

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nanoformula-ai.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/Tests-14%2F14%20Passed-brightgreen.svg)](tests/)
[![RDKit](https://img.shields.io/badge/Chemoinformatics-RDKit-green.svg)](https://www.rdkit.org/)

**Machine Learning-Driven Multi-Polymer Nanoparticle Formulation Optimizer, 4D Drug Release Kinetics, and Virtual Screening Platform**

🌐 **Live Application:** [https://nanoformula-ai.streamlit.app](https://nanoformula-ai.streamlit.app)  
📄 **Manuscript Draft:** [paper_materials/manuscript_draft.md](paper_materials/manuscript_draft.md)

---

## 📌 Overview

**NanoFormula AI 2.0** is an open-source, machine learning and chemoinformatics framework designed to accelerate the formulation design of polymeric nanoparticles (PLGA, PEG-PLGA, Chitosan-TPP, PCL) and nucleic acid Lipid Nanoparticles (LNPs). By bridging automated chemical structure parsing with ensemble machine learning, uncertainty quantification, 4D drug release kinetics modeling, and Pareto multi-objective optimization, NanoFormula AI enables formulation scientists to design optimal nanocarriers in seconds rather than weeks.

```
┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
│     API Input Mode        │      │    Ensemble ML & AD       │      │   Pareto Optimization     │
│  - SMILES / RDKit Engine  │ ───► │  - XGBoost + RF + GB + ET │ ───► │  - Non-Dominated Sorting  │
│  - PubChem Live API       │      │  - 95% Confidence UQ      │      │  - Desirability (D > 0.8) │
│  - 17 Curated Drug Panel  │      │  - William's Leverage AD  │      │  - Diverse Selection      │
└───────────────────────────┘      └───────────────────────────┘      └─────────────┬─────────────┘
                                                                                    │
                                                                                    ▼
                                                                      ┌───────────────────────────┐
                                                                      │   Actionable Outputs      │
                                                                      │  - 4D Release Kinetics    │
                                                                      │  - mRNA LNP Stoichiometry │
                                                                      │  - 3D WebGL Visualization │
                                                                      │  - TreeSHAP XAI Breakdown │
                                                                      │  - Step-by-Step Wet SOP   │
                                                                      │  - Official PDF Lab Cert  │
                                                                      └───────────────────────────┘
```

---

## 🎯 Key Capabilities

1. **Chemoinformatics Automation (RDKit & PubChem):**
   - Instant calculation of physicochemical descriptors (MW, LogP, TPSA, H-bond donors/acceptors, heteroatom count, QSPR melting point) from SMILES or drug names.
   - 3D conformer generation using RDKit MMFF force fields and interactive WebGL visualization.
2. **Multi-Model Ensemble & Uncertainty Quantification (UQ):**
   - Weighted ensemble combining XGBoost, Random Forest, Gradient Boosting, and Extra Trees.
   - Calibrated 95% Confidence Intervals ($\hat{y} \pm 1.96\sigma$) for all predictions.
3. **4D Drug Release Kinetics Simulator:**
   - Multi-phase temporal release simulation ($0.5\text{h} \rightarrow 168\text{h}$) modeling burst release, matrix diffusion, and polymer degradation.
   - Curve fitting for Korsmeyer-Peppas ($M_t/M_\infty = k t^n$), Higuchi, and First-Order models to classify transport mechanisms.
4. **mRNA Lipid Nanoparticles (LNP) Designer:**
   - 4-component clinical LNP stoichiometry (Ionizable lipid SM-102/ALC-0315, DSPC/DOPE, Cholesterol, DMG-PEG2000).
   - Exact Nitrogen-to-Phosphate (N/P) ratio calculation and microfluidic mixing recipes.
5. **PEG-PLGA Stealth & Polycaprolactone (PCL) Engines:**
   - Evaluates PEG grafting density, brush vs. mushroom regime, and macrophage evasion half-life multiplier.
   - Models PCL multi-month depot extended sustained release.
6. **Applicability Domain (AD) & TreeSHAP Explainable AI:**
   - Evaluates query leverage ($h^* = 3(p+1)/n$) and Mahalanobis distance to prevent extrapolation errors.
   - SHAP waterfall feature attribution for mechanistic transparency.
7. **Lab-in-the-Loop Active Learning:**
   - Ingests real experimental batch measurements to update Gaussian Process surrogate models.
8. **Automated Wet-Lab SOP & PDF Certificate Generator:**
   - Translates numerical vectors into batch-specific recipes (5–50 mL) and printable PDF laboratory certificates.

---

## 📊 Scientific Benchmark Performance (10-Fold Cross-Validation)

All models were evaluated across 8 distinct machine learning algorithms using 10-fold repeated cross-validation:

| Polymer System | Target Property | Best Algorithm | $R^2$ Score | RMSE | MAE | Pearson $r$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PLGA** ($n=433$) | **Loading Capacity (%)** | **NanoFormula Ensemble** | **0.8663** | **2.61%** | **1.25%** | **0.9309** |
| **PLGA** ($n=433$) | **Particle Size (nm)** | **Extra Trees / Ensemble** | **0.7491** | **50.61 nm** | **26.93 nm** | **0.8670** |
| **PLGA** ($n=433$) | **Entrapment Efficiency (%)**| **NanoFormula Ensemble** | **0.6686** | **13.57%** | **8.81%** | **0.8178** |
| **Chitosan-TPP** ($n=44$) | **Zeta Potential (mV)** | **Extra Trees / Ensemble** | **0.8126** | **2.35 mV** | **1.55 mV** | **0.9017** |
| **Chitosan-TPP** ($n=44$) | **Particle Size (nm)** | **XGBoost / Ensemble** | **0.7576** | **19.24 nm** | **11.19 nm** | **0.8836** |
| **Chitosan-TPP** ($n=44$) | **Polydispersity Index (PDI)** | **Ridge / Random Forest** | **0.4171** | **0.05** | **0.04** | **0.6615** |

*Publication parity plots (300 DPI) are available in `benchmarks/plga_parity_plots.png` and `benchmarks/chitosan_parity_plots.png`.*

---

## 📚 Independent Literature External Validation

Evaluated against published peer-reviewed studies without retraining:

| Study Citation | Drug | Polymer System | Exp. Size (nm) | AI Pred. Size (nm) | Size MAPE | Exp. EE (%) | AI Pred. EE (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Khalil et al. (2013) *Colloids Surf. B* | Curcumin | PLGA 50:50 (24 kDa) | $158.4 \pm 6.2$ | $154.2 \; [142, 166]$ | 2.6% | $82.5\%$ | $81.4\% \; [74, 88]$ |
| Danhier et al. (2009) *J. Control. Rel.*| Paclitaxel | PLGA 50:50 (45 kDa) | $172.0 \pm 8.5$ | $168.1 \; [155, 181]$ | 2.3% | $88.0\%$ | $85.6\% \; [78, 93]$ |
| Gómez-Gaete et al. (2007) *Eur. J. Pharm.*| Dexamethasone | PLGA 75:25 (30 kDa) | $195.0 \pm 11.0$| $186.4 \; [171, 202]$ | 4.4% | $68.4\%$ | $72.1\% \; [63, 81]$ |
| Calvo et al. (1997) *J. Appl. Polym. Sci.*| Blank CS-TPP | Chitosan (50 kDa) | $106.6 \pm 5.4$ | $102.8 \; [91, 114]$ | 3.6% | N/A | N/A |

---

## 📁 Repository Structure

```
├── app.py                          # Upgraded interactive Streamlit Dashboard (9 Modules)
├── data/                           # Verified experimental datasets & active learning DB
│   ├── PLGA_nanoparticles_dataset.csv
│   └── chitosan_nanoparticles_dataset.csv
├── nanoformula/                    # Core Python Package (`import nanoformula as nf`)
│   ├── chemoinformatics/           # RDKit automated descriptor calculation & PubChem
│   ├── kinetics/                   # 4D Drug Release Simulator & Korsmeyer-Peppas
│   ├── polymers/                   # mRNA LNPs, PEG-PLGA Stealth, and PCL Depot
│   ├── active_learning/            # Lab-in-the-Loop Bayesian Feedback Engine
│   ├── visualization3d/            # 3D Conformer & Core-Shell Nanoparticle Visualizer
│   ├── validation/                 # Independent Literature External Validation Suite
│   ├── ml/                         # Ensemble ML, UQ, AD, SHAP, and Training Pipeline
│   ├── optimization/               # Pareto Optimization & Desirability Ranking
│   └── protocols/                  # Wet-lab SOP engine & ReportLab PDF builder
├── saved_models/                   # Pre-trained model bundles and metadata
├── benchmarks/                     # 10-fold CV tables and 300 DPI publication parity plots
├── paper_materials/                # Research manuscript draft (Markdown & LaTeX tables)
├── tests/                          # Automated unit test suite (14/14 passing)
├── pyproject.toml                  # Modern PyPI build configuration
├── setup.py                        # Setuptools packaging
├── Dockerfile                      # Production container deployment
├── requirements.txt                # Pinned dependencies
└── README.md
```

---

## 🚀 Quickstart & Installation

### Python Package Usage

```python
import nanoformula as nf

# 1. Chemoinformatics descriptor calculation from SMILES
desc = nf.calculate_descriptors_from_smiles("CC(=O)Oc1ccccc1C(=O)O")

# 2. Optimize mRNA Lipid Nanoparticle (LNP)
lnp_opt = nf.LNPOptimizer()
lnp_res = lnp_opt.optimize_lnp(target_np_ratio=6.0, mrna_dose_ug=50.0)

# 3. Simulate 4D Drug Release Kinetics
predictor = nf.DrugReleasePredictor()
kinetics = predictor.predict_release_curve({
    "polymer_MW": 30.0, "LA/GA": 1.0, "mol_logP": 3.2,
    "pred_size": 150.0, "pred_EE": 85.0
})
```

### Local Dashboard Launch

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run test suite:**
   ```bash
   pytest tests/
   ```

3. **Launch the web dashboard:**
   ```bash
   streamlit run app.py
   ```

---

## 👨‍🔬 Research Team & Lab Affiliation

- **Developer:** Hardik Sood (B.Tech Pharmaceutical Engineering, IIT (BHU) Varanasi)
- **Supervisor:** Dr. Ruchi Chawla (Associate Professor, Department of Pharmaceutical Engineering & Technology, IIT (BHU) Varanasi)
- **Institution:** Indian Institute of Technology (BHU), Varanasi, India

---

## 📖 Citation

If you use NanoFormula AI in your academic research or industrial formulation workflows, please cite the software repository:

```bibtex
@software{sood2026nanoformula,
  author = {Sood, Hardik and Chawla, Ruchi},
  title = {NanoFormula AI: Multi-Polymer Nanoparticle Formulation Optimizer and Virtual Screening Platform},
  url = {https://github.com/hardiksood21/nanoformula-ai},
  year = {2026},
  institution = {Department of Pharmaceutical Engineering & Technology, IIT (BHU) Varanasi},
  note = {Software repository. Manuscript in preparation.}
}
```
