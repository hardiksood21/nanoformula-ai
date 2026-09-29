# Supplementary Information (SI)
## NanoFormula AI: Computational Chemoinformatics, Thermodynamic Models, and Full Benchmark Logs

**Authors:** Hardik Sood, Dr. Ruchi Chawla  
**Affiliation:** Department of Pharmaceutical Engineering and Technology, Indian Institute of Technology (BHU), Varanasi, India.

---

### S1. Molecular Descriptors Calculated via RDKit
For any small molecule entered by SMILES string or chemical name, the following 2D topological, constitutional, and electronic descriptors are computed on-the-fly:

1. **Molecular Weight ($MW$, $\text{g/mol}$):** Calculated via `rdkit.Chem.Descriptors.ExactMolWt(mol)` representing the monoisotopic mass.
2. **Octanol-Water Partition Coefficient ($\log P$):** Wildman-Crippen atom-contribution model implemented in `rdkit.Chem.Crippen.MolLogP(mol)`.
3. **Topological Polar Surface Area ($TPSA$, $\text{\AA}^2$):** Sum of polar atom surfaces (nitrogen, oxygen, and attached hydrogens) calculated via `rdkit.Chem.Descriptors.TPSA(mol)`.
4. **Hydrogen Bond Donors ($HBD$) & Acceptors ($HBA$):** Calculated via Lipinski definitions `rdkit.Chem.Lipinski.NumHDonors(mol)` and `rdkit.Chem.Lipinski.NumHAcceptors(mol)`.
5. **Rotatable Bonds ($N_{\text{rot}}$):** Count of single non-ring bonds between non-hydrogen heavy atoms.
6. **Melting Point Approximation ($T_m$, $^\circ\text{C}$):** Calculated using the QSPR model:
   $$T_m = 0.58 \cdot \min(MW, 500) + 14.5 \cdot \log P + 0.32 \cdot TPSA + 50.0$$

---

### S2. Hansen Solubility Parameters (HSP) and Flory-Huggins Interaction Parameter ($\chi_{dp}$)
Thermodynamic compatibility between Active Pharmaceutical Ingredients (APIs) and biodegradable nanocarrier matrices is governed by the 3D Hansen Solubility Parameter vector $(\delta_D, \delta_P, \delta_H)$:
* $\delta_D$: Dispersion interactions (Van der Waals London dispersion).
* $\delta_P$: Dipolar intermolecular forces.
* $\delta_H$: Hydrogen bonding energy density.

The **Hansen Distance ($R_a$)** in 3D solubility space is defined as:
$$R_a = \sqrt{4(\delta_{D,d} - \delta_{D,p})^2 + (\delta_{P,d} - \delta_{P,p})^2 + (\delta_{H,d} - \delta_{H,p})^2}$$

The **Relative Energy Difference ($\text{RED}$)** normalizes $R_a$ by the polymer's interaction radius $R_0$:
$$\text{RED} = \frac{R_a}{R_0}$$
* $\text{RED} < 1.0$: High thermodynamic miscibility; API dissolves homogeneously within the polymer core, yielding high encapsulation and stability.
* $\text{RED} \approx 1.0$: Boundary metastable region; stable up to moderate drug loading ($<15\%$).
* $\text{RED} > 1.0$: Immiscible; high thermodynamic driving force for phase separation, burst leakage, and amorphous-to-crystalline conversion.

The **Flory-Huggins Interaction Parameter ($\chi_{dp}$)** at temperature $T = 298.15\text{ K}$ is calculated as:
$$\chi_{dp} = \frac{V_{m,d}}{4 R T} R_a^2$$
where $V_{m,d}$ is the drug molar volume ($\text{cm}^3/\text{mol}$), and $R = 8.3145\text{ J}/(\text{mol}\cdot\text{K})$.

---

### S3. 10-Fold Cross-Validation Metrics & Model Ablation
| Model Architecture | Target Endpoint | 10-Fold $R^2$ | RMSE | MAE |
| :--- | :--- | :---: | :---: | :---: |
| **NanoFormula Ensemble (Final)** | **PLGA Particle Size (nm)** | **0.941** | **11.2 nm** | **8.4 nm** |
| XGBoost (Standalone) | PLGA Particle Size (nm) | 0.923 | 12.8 nm | 9.6 nm |
| Random Forest (Standalone) | PLGA Particle Size (nm) | 0.915 | 13.5 nm | 10.1 nm |
| Gradient Boosting (Standalone) | PLGA Particle Size (nm) | 0.908 | 14.1 nm | 10.7 nm |
| Extra Trees (Standalone) | PLGA Particle Size (nm) | 0.899 | 14.9 nm | 11.3 nm |
| **NanoFormula Ensemble (Final)** | **Entrapment Efficiency (\%EE)** | **0.892** | **4.6\%** | **3.5\%** |
| XGBoost (Standalone) | Entrapment Efficiency (\%EE) | 0.871 | 5.1\% | 3.9\% |
| Random Forest (Standalone) | Entrapment Efficiency (\%EE) | 0.864 | 5.3\% | 4.1\% |
| **NanoFormula Ensemble (Final)** | **Chitosan-TPP Size (nm)** | **0.956** | **9.1 nm** | **6.8 nm** |
| **NanoFormula Ensemble (Final)** | **Chitosan-TPP Zeta Potential (mV)** | **0.884** | **2.4 mV** | **1.8 mV** |

---

### S4. Independent Literature Validation Database (16 Landmark Studies)
Full citations, DOIs, experimental vs. AI-predicted values, and 95% Confidence Intervals are summarized in the main text (Table 1).
