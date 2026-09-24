"""
Laboratory Protocol Synthesis and Formulation Recipe Engine.
Calculates exact chemical masses, solvent volumes, sonication, stirring,
purification, and DLS/HPLC characterization Standard Operating Procedures (SOPs).
"""

from typing import Dict, Any, List, Optional


def select_preparation_method(drug_logp: float, solvent_pol: float) -> Dict[str, str]:
    """
    Selects the optimal nanoparticle preparation method based on drug lipophilicity (LogP)
    and organic solvent miscibility.
    """
    if drug_logp >= 2.5:
        return {
            "method": "Single Emulsion (O/W) Solvent Evaporation",
            "organic_solvent": "Dichloromethane (DCM) or Ethyl Acetate",
            "aqueous_phase": "Milli-Q water with PVA surfactant (pH adjusted)",
            "mechanism": "Hydrophobic drug and PLGA co-dissolved in organic phase, emulsified into aqueous surfactant phase via probe sonication.",
            "advantages": "High entrapment efficiency for lipophilic therapeutics (>80%), narrow size distribution."
        }
    elif 0.5 <= drug_logp < 2.5:
        return {
            "method": "Nanoprecipitation (Fessi Method)",
            "organic_solvent": "Acetone or Acetonitrile (water-miscible)",
            "aqueous_phase": "Milli-Q water with Poloxamer 188 (Pluronic F68) / PVA",
            "mechanism": "Spontaneous Marangoni effect / solvent displacement upon dropwise addition into aqueous non-solvent under magnetic stirring.",
            "advantages": "Ultra-small sub-150 nm particle size, narrow PDI, organic solvent evaporates rapidly under ambient conditions."
        }
    else:
        return {
            "method": "Double Emulsion (W1/O/W2) Solvent Evaporation",
            "organic_solvent": "Dichloromethane (DCM) / Acetone mixture",
            "aqueous_phase": "Primary internal aqueous (W1) + external aqueous (W2) with PVA",
            "mechanism": "Primary water-in-oil (W1/O) emulsion formed, then re-emulsified into external continuous aqueous phase (W2).",
            "advantages": "Enables encapsulation of hydrophilic drugs, peptides, and proteins with reduced leakage."
        }


def generate_plga_lab_sop(formulation: Dict[str, Any], batch_volume_ml: float = 10.0, drug_name: str = "Drug") -> Dict[str, Any]:
    """
    Generates a complete, step-by-step laboratory SOP and batch quantity calculations for PLGA formulation.
    """
    p_mw = float(formulation.get('polymer_MW', 24.0))
    laga = float(formulation.get('LA/GA', 1.0))
    drug_poly_ratio = float(formulation.get('drug/polymer', 0.1))
    surf_conc = float(formulation.get('surfactant_concentration', 1.0))
    surf_hlb = float(formulation.get('surfactant_HLB', 18.0))
    aq_org = float(formulation.get('aqueous/organic', 4.0))
    ph = float(formulation.get('pH', 0.0))
    solv_pol = float(formulation.get('solvent_polarity_index', 5.1))
    logp = float(formulation.get('mol_logP', 2.5))

    # Standard PLGA polymer concentration: 10 mg/mL in organic phase
    org_vol_ml = round(batch_volume_ml / (aq_org + 1.0), 2)
    aq_vol_ml = round(batch_volume_ml - org_vol_ml, 2)
    
    polymer_mass_mg = round(org_vol_ml * 15.0, 1)  # 15 mg/mL organic phase
    drug_mass_mg = round(polymer_mass_mg * drug_poly_ratio, 2)
    surfactant_mass_mg = round((surf_conc / 100.0) * aq_vol_ml * 1000.0, 1)

    # LA/GA text
    laga_ratio_text = "50:50" if abs(laga - 1.0) < 0.1 else ("65:35" if abs(laga - 1.85) < 0.2 else ("75:25" if abs(laga - 3.0) < 0.3 else "85:15"))
    method_info = select_preparation_method(logp, solv_pol)

    steps = [
        f"1. Preparation of Organic Phase: Accurately weigh {polymer_mass_mg} mg of PLGA ({laga_ratio_text}, MW {p_mw:.1f} kDa) and {drug_mass_mg} mg of {drug_name}. Dissolve completely in {org_vol_ml} mL of {method_info['organic_solvent']} via gentle vortexing (2 min).",
        f"2. Preparation of Aqueous Phase: Dissolve {surfactant_mass_mg} mg of Surfactant (HLB {surf_hlb:.1f}, e.g. PVA MW 31-50 kDa) in {aq_vol_ml} mL of Milli-Q deionized water. Adjust aqueous phase pH to {7.4 + ph:.1f} if buffering is required.",
        f"3. Emulsification / Nanoprecipitation: Add the organic phase into the aqueous phase. Place beaker in an ice bath. Subject mixture to probe sonication at 40% amplitude for 2 minutes (10s pulse ON, 5s pulse OFF) to achieve a homogeneous nanodispersion.",
        f"4. Solvent Evaporation: Stir the nanodispersion under magnetic stirring (600-800 RPM) at room temperature for 3-4 hours (or rotary evaporate at 40°C, 100 mbar for 20 min) until organic solvent is completely eliminated.",
        f"5. Purification & Washing: Centrifuge nanodispersion at 12,000 RPM (15,000 × g) for 25 min at 4°C. Discard supernatant (reserve for unentrapped drug assay). Resuspend nanoparticle pellet in 10 mL Milli-Q water and repeat centrifugation once to remove residual surfactant.",
        f"6. Cryoprotection & Lyophilization: Resuspend purified nanoparticles in 5% (w/v) D-trehalose / mannitol aqueous solution. Freeze at -80°C for 4 hours and lyophilize for 24-48 hours.",
        f"7. Analytical Characterization: Reconstitute 1 mg/mL in Milli-Q water. Characterize hydrodynamic diameter (Z-avg) and Polydispersity Index (PDI) by Dynamic Light Scattering (DLS) at 25°C. Quantify Entrapment Efficiency (% EE) and Drug Loading (% DL) using reverse-phase HPLC / UV-Vis spectrophotometry."
    ]

    return {
        "title": f"Standard Operating Procedure: {drug_name}-Loaded PLGA Nanoparticles",
        "method": method_info["method"],
        "organic_solvent": method_info["organic_solvent"],
        "aqueous_phase": method_info["aqueous_phase"],
        "recipe": {
            "batch_volume_ml": batch_volume_ml,
            "organic_phase_volume_ml": org_vol_ml,
            "aqueous_phase_volume_ml": aq_vol_ml,
            "polymer_mass_mg": polymer_mass_mg,
            "polymer_type": f"PLGA {laga_ratio_text} ({p_mw:.1f} kDa)",
            "drug_mass_mg": drug_mass_mg,
            "drug_polymer_ratio": round(drug_poly_ratio, 4),
            "surfactant_mass_mg": surfactant_mass_mg,
            "surfactant_conc_percent": surf_conc,
            "surfactant_hlb": surf_hlb
        },
        "steps": steps
    }


def generate_chitosan_lab_sop(formulation: Dict[str, Any], batch_volume_ml: float = 10.0) -> Dict[str, Any]:
    """
    Generates laboratory SOP and batch quantity calculations for Chitosan-TPP Nanoparticles.
    """
    mw = float(formulation.get('chitosan_MW', 50.0))
    cs_conc = float(formulation.get('chitosan_conc', 0.5))
    tpp_conc = float(formulation.get('TPP_conc', 0.5))
    cs_tpp_ratio = float(formulation.get('chitosan_TPP_ratio', 1.0))

    # Chitosan solution (e.g. 7 mL) and TPP solution (3 mL)
    cs_vol_ml = round(batch_volume_ml * 0.7, 1)
    tpp_vol_ml = round(batch_volume_ml * 0.3, 1)

    cs_mass_mg = round(cs_conc * cs_vol_ml, 2)
    tpp_mass_mg = round(tpp_conc * tpp_vol_ml, 2)

    mw_desc = "Low MW (~50 kDa)" if mw <= 50 else ("High MW (~310 kDa)" if mw >= 300 else f"{mw:.0f} kDa")

    steps = [
        f"1. Chitosan Stock Solution: Dissolve {cs_mass_mg} mg of Chitosan ({mw_desc}) in {cs_vol_ml} mL of 1.0% (v/v) glacial acetic acid aqueous solution. Stir overnight on magnetic stirrer (400 RPM) at room temperature until completely dissolved. Filter through a 0.45 μm PTFE syringe filter.",
        f"2. Sodium TPP Crosslinker Solution: Accurately dissolve {tpp_mass_mg} mg of Sodium Tripolyphosphate (TPP) in {tpp_vol_ml} mL of Milli-Q deionized water. Adjust TPP solution pH to 8.5-9.0 using 0.1 M NaOH if required.",
        f"3. Ionic Gelation Reaction: Place the chitosan solution in a clean reaction beaker under vigorous magnetic stirring (700-900 RPM) at 25°C. Add the {tpp_vol_ml} mL TPP solution dropwise (rate: 1 mL/min) using a micro-syringe or peristaltic pump directly into the vortex.",
        f"4. Crosslinking & Maturation: Continue magnetic stirring for 30-45 minutes at room temperature to allow complete electrostatic crosslinking between protonated amine groups (-NH3+) of chitosan and polyphosphate anions of TPP.",
        f"5. Purification & Harvesting: Centrifuge the resulting opalescent nanosuspension at 13,000 RPM (16,000 × g) for 30 minutes at 15°C over a 10 μL glycerol bed. Decant supernatant and wash nanoparticles with Milli-Q water.",
        f"6. Characterization: Resuspend in deionized water (pH 6.5). Measure hydrodynamic diameter, Polydispersity Index (PDI), and positive Zeta Potential (surface charge) by Dynamic Light Scattering (DLS)."
    ]

    return {
        "title": "Standard Operating Procedure: Chitosan-TPP Nanoparticles via Ionic Gelation",
        "method": "Ionic Gelation (Calvo Method)",
        "recipe": {
            "batch_volume_ml": batch_volume_ml,
            "chitosan_volume_ml": cs_vol_ml,
            "chitosan_mass_mg": cs_mass_mg,
            "chitosan_conc_mg_ml": cs_conc,
            "chitosan_type": mw_desc,
            "tpp_volume_ml": tpp_vol_ml,
            "tpp_mass_mg": tpp_mass_mg,
            "tpp_conc_mg_ml": tpp_conc,
            "mass_ratio_cs_tpp": round(cs_tpp_ratio, 2)
        },
        "steps": steps
    }
