# 🧬 NanoFormula AI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://nanoformula-ai.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Passing](https://img.shields.io/badge/Tests-6%2F6%20Passed-brightgreen.svg)](tests/)
[![RDKit](https://img.shields.io/badge/Chemoinformatics-RDKit-green.svg)](https://www.rdkit.org/)

**Machine Learning-Driven Multi-Objective Nanoparticle Formulation Optimization and High-Throughput Virtual Screening Platform**

🌐 **Live Application:** [https://nanoformula-ai.streamlit.app](https://nanoformula-ai.streamlit.app)  
📄 **Manuscript Draft:** [paper_materials/manuscript_draft.md](paper_materials/manuscript_draft.md)

---

## 📌 Overview

**NanoFormula AI** is an open-source, machine learning and chemoinformatics framework designed to accelerate the formulation development of polymeric nanoparticles (PLGA and Chitosan-TPP). By bridging automated chemical structure parsing with ensemble machine learning, uncertainty quantification, and Pareto multi-objective optimization, NanoFormula AI enables formulation scientists to design optimal nanocarriers with target Critical Quality Attributes (CQAs) in seconds rather than weeks.

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
                                                                      │  - Interactive Plotly 3D  │
                                                                      │  - TreeSHAP XAI Breakdown │
                                                                      │  - Step-by-Step Wet SOP   │
                                                                      │  - Official PDF Lab Cert  │
                                                                      └───────────────────────────┘
```

---

## 🎯 Key Capabilities

1. **Chemoinformatics Automation (RDKit & PubChem):**
   - Instant calculation of physicochemical descriptors (MW, LogP, TPSA, H-bond donors/acceptors, heteroatom count, QSPR melting point) from SMILES or drug names.
   - Real-time 2D vector structure rendering.
2. **Multi-Model Ensemble & Uncertainty Quantification (UQ):**
   - Weighted ensemble combining XGBoost, Random Forest, Gradient Boosting, and Extra Trees.
   - Calibrated 95% Confidence Intervals ($\hat{y} \pm 1.96\sigma$) for all predictions.
3. **Applicability Domain (AD) Verification:**
   - Evaluates query leverage (Hat matrix diagonal $h^* = 3(p+1)/n$) and Mahalanobis distance to protect against extrapolation errors.
4. **Pareto Multi-Objective Optimization:**
   - Non-dominated sorting and Derringer-Suich desirability functions targeting particle size, entrapment efficiency (% EE), and loading capacity (% LC).
5. **Explainable AI (TreeSHAP):**
   - Local and global feature attribution waterfall charts explaining *why* specific formulation parameters impact hydrodynamic size or encapsulation.
6. **Automated Wet-Lab SOP & PDF Certificate Generator:**
   - Translates numerical formulation vectors into batch-specific recipes (5–50 mL) and printable PDF laboratory certificates.
7. **High-Throughput Virtual Screening:**
   - Batch screening tab to evaluate drug libraries against PLGA and Chitosan systems.

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

## 📁 Repository Structure

```
├── app.py                          # Upgraded, modern interactive Streamlit Dashboard
├── data/
│   ├── PLGA_nanoparticles_dataset.csv
│   └── chitosan_nanoparticles_dataset.csv
├── nanoformula/
│   ├── chemoinformatics/           # RDKit automated descriptor calculation & PubChem API
│   │   ├── descriptors.py
│   │   └── drug_database.py        # Curated library of 17+ high-value nanomedicine APIs
│   ├── ml/                         # ML Ensemble, UQ, AD, SHAP, and Training Pipeline
│   │   ├── ensemble_model.py
│   │   ├── applicability_domain.py
│   │   ├── explainability.py
│   │   └── trainer.py
│   ├── optimization/               # Pareto optimization & Desirability ranking
│   │   └── pareto_optimizer.py
│   └── protocols/                  # Wet-lab SOP engine & ReportLab PDF builder
│       ├── lab_protocol_engine.py
│       └── pdf_report_builder.py
├── saved_models/                   # Pre-trained model bundles and metadata
├── benchmarks/                     # 10-fold CV tables and 300 DPI publication parity plots
├── paper_materials/                # Research manuscript draft (Markdown & LaTeX tables)
├── tests/                          # Automated unit test suite (pytest)
├── Dockerfile                      # Production container deployment
├── requirements.txt                # Pinned dependencies
└── README.md
```

---

## 🚀 Quickstart & Installation

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hardiksood21/nanoformula-ai.git
   cd nanoformula-ai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run unit tests:**
   ```bash
   pytest tests/
   ```

4. **Launch the web dashboard:**
   ```bash
   streamlit run app.py
   ```

### Docker Deployment

```bash
docker build -t nanoformula-ai .
docker run -p 8501:8501 nanoformula-ai
```

---

## 👨‍🔬 Research Team & Lab Affiliation

- **Developer:** Hardik Sood (B.Tech Pharmaceutical Engineering, IIT (BHU) Varanasi)
- **Supervisor:** Dr. Ruchi Chawla (Associate Professor, Department of Pharmaceutical Engineering & Technology, IIT (BHU) Varanasi)
- **Institution:** Indian Institute of Technology (BHU), Varanasi, India

---

## 📖 Citation

If you use NanoFormula AI in your academic research or industrial formulation workflows, please cite:

```bibtex
@article{sood2026nanoformula,
  title={NanoFormula AI: Machine Learning-Driven Multi-Objective Nanoparticle Formulation Optimization and High-Throughput Virtual Screening},
  author={Sood, Hardik and Chawla, Ruchi},
  journal={Journal of Controlled Release},
  year={2026},
  publisher={Elsevier}
}
```
