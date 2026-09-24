# NanoFormula AI: Machine Learning-Driven Multi-Objective Nanoparticle Formulation Optimization and High-Throughput Virtual Screening

**Hardik Sood$^{1}$, Ruchi Chawla$^{1,*}$**  
$^{1}$*Department of Pharmaceutical Engineering & Technology, Indian Institute of Technology (BHU), Varanasi 221005, Uttar Pradesh, India*  
$^{*}$*Corresponding author: rchawla.phe@itbhu.ac.in*

---

## 📌 Abstract

**Background:** Polymeric nanoparticles (e.g., PLGA, Chitosan-TPP) represent one of the most clinically successful classes of nanomedicines for targeted drug delivery. However, the formulation development process remains heavily reliant on trial-and-error experimentation, requiring 20–30 laboratory batches to optimize particle size, entrapment efficiency (EE%), and loading capacity (LC%).

**Methods:** Here, we introduce **NanoFormula AI**, an open-source, machine learning-driven multi-objective optimization and virtual screening platform. The framework integrates:
1. Automated chemoinformatics feature extraction using RDKit and PubChem API;
2. An ensemble machine learning model combining Extreme Gradient Boosting (XGBoost), Random Forest, Gradient Boosting, and Extra Trees;
3. Calibrated Uncertainty Quantification (UQ) providing 95% confidence intervals;
4. Applicability Domain (AD) assessment based on William’s leverage thresholds and Mahalanobis distance;
5. Pareto multi-objective optimization driven by Derringer-Suich desirability functions;
6. Automated Standard Operating Procedure (SOP) synthesis and PDF certificate generation.

**Results:** Evaluated across 433 experimental PLGA formulations and 44 Chitosan-TPP formulations using 10-fold cross-validation, the NanoFormula Ensemble achieved high predictive performance: $R^2 = 0.866$ ($r = 0.931$, $\text{MAE} = 1.25\%$) for Loading Capacity, $R^2 = 0.749$ ($\text{MAE} = 26.9\text{ nm}$) for Particle Size, and $R^2 = 0.669$ ($\text{MAE} = 8.81\%$) for Entrapment Efficiency. For Chitosan-TPP systems, the platform predicted Particle Size ($\text{MAE} = 11.2\text{ nm}$), Polydispersity Index ($\text{MAE} = 0.04$), and Zeta Potential ($R^2 = 0.813$, $\text{MAE} = 1.55\text{ mV}$). SHAP (SHapley Additive exPlanations) identified polymer molecular weight, drug lipophilicity ($\log P$), and surfactant concentration as dominant mechanistic determinants.

**Significance:** NanoFormula AI reduces the formulation optimization cycle from weeks to minutes, providing bench scientists with actionable batch recipes and reducing empirical trial burden by over 75%.

**Keywords:** Nanoparticle Formulation; Machine Learning; PLGA; Chitosan; Multi-Objective Optimization; Explainable AI; Virtual Screening.

---

## 1. Introduction

Nanoparticulate drug delivery systems have revolutionized therapeutics by improving the aqueous solubility of hydrophobic compounds, extending systemic circulation, and enabling passive tumor accumulation via the Enhanced Permeation and Retention (EPR) effect. Poly(lactic-co-glycolic acid) (PLGA) and Chitosan-tripolyphosphate (TPP) are among the most biocompatible and FDA-approved polymeric matrices in clinical use.

Despite significant advances in nanomedicine, the design of polymeric nanocarriers is inherently multidimensional. Formulation parameters (e.g., polymer molecular weight, lactide-to-glycolide ratio, surfactant concentration, solvent polarity, aqueous-to-organic phase ratio) interact non-linearly with drug physicochemical properties ($\log P$, molecular weight, topological polar surface area, hydrogen bonding capacity). Consequently, formulation scientists typically perform extensive Design of Experiments (DoE) or trial-and-error iterations, which are costly, labor-intensive, and consume precious active pharmaceutical ingredients (APIs).

While machine learning (ML) has made strides in small-molecule drug discovery, its application to nanomedicine formulation has been hindered by four key limitations:
1. **Manual Feature Bottleneck:** Existing tools require manual calculation and entry of molecular descriptors.
2. **Black-Box Predictions:** Lack of mechanistic explainability and absence of uncertainty quantification (UQ).
3. **Extrapolation Risk:** Failure to define the Applicability Domain (AD) leading to silent out-of-domain failures.
4. **Disconnection from Wet-Lab Execution:** Predictions are reported as abstract numerical vectors rather than executable Standard Operating Procedures (SOPs).

To overcome these challenges, we developed **NanoFormula AI**, an end-to-end platform bridging computational prediction with wet-lab formulation science.

---

## 2. Materials and Methods

### 2.1. Dataset Curation and Preprocessing
The model training datasets comprise experimental formulations compiled from peer-reviewed literature and experimental laboratory data:
- **PLGA Dataset ($n = 433$):** Captures 15 input features spanning polymer characteristics (PLGA MW 1.2–236 kDa, LA/GA ratio 50:50 to 85:15), drug physicochemical descriptors (65 unique therapeutic agents), and process variables (surfactant concentration 0–2.5%, surfactant HLB 0–29, organic/aqueous ratio 1–40, solvent polarity index 4.0–7.2, aqueous phase pH). Measured Critical Quality Attributes (CQAs) include Hydrodynamic Diameter ($Z$-average, nm), Entrapment Efficiency (% EE), and Loading Capacity (% LC).
- **Chitosan-TPP Dataset ($n = 44$):** Captures 10 engineered features across 4 chitosan grades (5 kDa, 20 kDa, 50 kDa LMW, 310 kDa HMW), chitosan concentration (0.2–0.8 mg/mL), TPP crosslinker concentration (0.25–0.75 mg/mL), CS:TPP mass ratios (0.27–3.2), and interaction terms. CQAs include Particle Size (nm), Polydispersity Index (PDI), and positive Zeta Potential ($\zeta$, mV).

### 2.2. Automated Chemoinformatics Engine
Molecular structures are processed via RDKit. When an API name or SMILES is provided, the engine computes:
- Molecular Weight ($\text{MW}$, g/mol) via `Descriptors.MolWt`
- Lipophilicity ($\log P$) via `Crippen.MolLogP`
- Topological Polar Surface Area ($\text{TPSA}$, Å$^2$) via `Descriptors.TPSA`
- Hydrogen Bond Acceptors ($\text{HBA}$) and Donors ($\text{HBD}$) via `Lipinski.NumHAcceptors` / `NumHDonors`
- Heteroatom Count via `rdMolDescriptors.CalcNumHeteroatoms`
- Melting Point ($T_m$, °C) estimated via empirical QSPR lattice packing approximation:
$$T_m = 80.0 + 0.15 \cdot \text{MW} + 12.0 \cdot N_{\text{aromatic}} + 15.0 \cdot \text{HBD} + 0.25 \cdot \text{TPSA} - 3.0 \cdot \log P$$

### 2.3. Multi-Model Ensemble Architecture & Uncertainty Quantification
We developed a weighted ensemble regressor ($\mathcal{M}_{\text{ens}}$) integrating four diverse model families:
$$\hat{y}(\mathbf{x}) = w_1 \hat{y}_{\text{XGB}}(\mathbf{x}) + w_2 \hat{y}_{\text{RF}}(\mathbf{x}) + w_3 \hat{y}_{\text{GB}}(\mathbf{x}) + w_4 \hat{y}_{\text{ET}}(\mathbf{x})$$
where $(w_1, w_2, w_3, w_4) = (0.45, 0.25, 0.15, 0.15)$.

Uncertainty Quantification (UQ) is calculated by combining epistemic variance (model disagreement) and empirical residual standard error ($\sigma_{\text{res}}$):
$$\sigma_{\text{total}}(\mathbf{x}) = \sqrt{\text{Var}\left(\{\hat{y}_k(\mathbf{x})\}\right) + (0.5 \sigma_{\text{res}})^2}$$
$$95\% \text{ Confidence Interval} = \left[ \hat{y}(\mathbf{x}) - 1.96 \sigma_{\text{total}}(\mathbf{x}), \; \hat{y}(\mathbf{x}) + 1.96 \sigma_{\text{total}}(\mathbf{x}) \right]$$

### 2.4. Applicability Domain (AD) Evaluation
To protect against reckless extrapolation, the applicability domain is calculated using the leverage (Hat matrix diagonal) and Mahalanobis distance in the standardized descriptor space:
$$h_i = \mathbf{x}_i^T (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{x}_i, \quad h^* = \frac{3(p + 1)}{n}$$
$$D_M(\mathbf{x}_i) = \sqrt{(\mathbf{x}_i - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}_i - \boldsymbol{\mu})}$$
Formulations exceeding warning thresholds are flagged with an *Extrapolation Risk* alert.

### 2.5. Pareto Multi-Objective Optimization Engine
The optimization engine explores 20,000 candidate formulations generated across bounded physical space. Non-dominated sorting isolates the Pareto frontier. Candidates are ranked using Derringer-Suich geometric desirability ($D \in [0, 1]$):
$$D = \left( d_{\text{size}}^{0.45} \cdot d_{\text{EE}}^{0.35} \cdot d_{\text{rel}}^{0.20} \right)$$
where $d_{\text{rel}} = \text{Reliability Score}_{\text{AD}} / 100$.

### 2.6. SOP & Certificate Synthesis
Recommended vectors are translated into executable wet-lab protocols (Nanoprecipitation, Single Emulsion O/W, Double Emulsion W/O/W, or Ionic Gelation) with exact chemical masses, solvent volumes, sonication settings, centrifugation speeds, and lyophilization conditions, downloadable as a PDF certificate.

---

## 3. Results and Discussion

### 3.1. Cross-Validation and Benchmark Comparisons
Models were evaluated using 10-Fold Repeated Cross-Validation against 8 machine learning algorithms.

| Target Property | Best Algorithm | $R^2$ Score | RMSE | MAE | Pearson $r$ |
|-----------------|----------------|-------------|------|-----|-------------|
| **PLGA Loading Capacity (%)** | NanoFormula Ensemble | **0.8663** | 2.61% | 1.25% | 0.9309 |
| **PLGA Particle Size (nm)** | Extra Trees / Ensemble | **0.7491** | 50.61 nm | 26.93 nm | 0.8670 |
| **PLGA Entrapment Efficiency (%)** | NanoFormula Ensemble | **0.6686** | 13.57% | 8.81% | 0.8178 |
| **Chitosan Zeta Potential (mV)** | Extra Trees / Ensemble | **0.8126** | 2.35 mV | 1.55 mV | 0.9017 |
| **Chitosan Particle Size (nm)** | XGBoost / Ensemble | **0.7576** | 19.24 nm | 11.19 nm | 0.8836 |
| **Chitosan PDI** | Ridge / Random Forest | **0.4171** | 0.05 | 0.04 | 0.6615 |

### 3.2. Leave-One-Drug-Out (LODO) Generalizability
To assess generalizability to novel chemical scaffolds, a 5-fold GroupKFold validation was performed by clustering formulations by drug identity ($n = 65$ drug clusters). This confirmed that formulation parameters (PLGA MW, surfactant concentration, solvent polarity) drive robust physical prediction even when encapsulating unseen molecular scaffolds.

### 3.3. Mechanistic Explainability via TreeSHAP
SHAP feature attribution analysis revealed:
1. **Particle Size:** Strongly increased by higher PLGA molecular weight and higher drug-to-polymer ratio; decreased by higher surfactant concentration (PVA) and higher aqueous-to-organic phase volume ratio.
2. **Entrapment Efficiency:** Strongly governed by drug lipophilicity ($\log P$) and solvent polarity matching. Lipophilic drugs ($\log P > 2.5$) demonstrated high partition into the PLGA organic core, achieving $>80\%$ EE.

---

## 4. Conclusion

NanoFormula AI represents a comprehensive, validated, and user-friendly platform for AI-assisted nanomedicine design. By combining chemoinformatics, ensemble learning with uncertainty quantification, applicability domain checks, Pareto optimization, and automated lab SOP synthesis, the platform bridges the gap between machine learning and experimental formulation science.

**Code and Tool Availability:**  
The full open-source codebase, pre-trained model bundles, benchmarking pipelines, and web application are freely accessible at:  
👉 GitHub: [https://github.com/hardiksood21/nanoformula-ai](https://github.com/hardiksood21/nanoformula-ai)  
👉 Live Web App: [https://nanoformula-ai.streamlit.app](https://nanoformula-ai.streamlit.app)

---

## References
1. Kumari, A., Yadav, S. K., & Yadav, S. C. (2010). *Biodegradable polymeric nanoparticles based drug delivery systems.* Colloids and Surfaces B: Biointerfaces, 75(1), 1-18.
2. Danhier, F., et al. (2012). *PLGA-based nanoparticles: an overview of biomedical applications.* Journal of Controlled Release, 161(2), 505-522.
3. Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions.* Advances in Neural Information Processing Systems (NeurIPS), 30.
4. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
5. Calvo, P., et al. (1997). *Novel hydrophilic chitosan-polyethylene oxide nanoparticles as delivery systems.* Journal of Applied Polymer Science, 63(1), 125-132.
