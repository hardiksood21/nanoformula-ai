"""
Unit tests for NanoFormula AI modules.
Tests chemoinformatics, ML ensemble models, UQ, applicability domain,
Pareto optimization, and PDF protocol generation.
"""

import pytest
import numpy as np
import pandas as pd
from nanoformula.chemoinformatics import (
    calculate_descriptors_from_smiles,
    get_drug_by_name,
    get_drug_names,
    estimate_melting_point,
    RDKIT_AVAILABLE
)
from nanoformula.ml import (
    NanoparticleEnsembleRegressor,
    ApplicabilityDomain,
    ModelExplainer,
    PLGA_FEATURES,
    CHITOSAN_FEATURES
)
from nanoformula.optimization import (
    PLGAFormulationOptimizer,
    ChitosanFormulationOptimizer,
    find_pareto_front,
    calculate_desirability_size,
    calculate_desirability_ee
)
from nanoformula.protocols import (
    generate_plga_lab_sop,
    generate_chitosan_lab_sop,
    generate_formulation_pdf_report
)


def test_drug_database():
    drug_names = get_drug_names()
    assert len(drug_names) >= 15
    assert "Paclitaxel" in drug_names
    assert "Curcumin" in drug_names
    
    curc = get_drug_by_name("Curcumin")
    assert curc["mol_MW"] == 368.38
    assert curc["mol_logP"] == 3.20


def test_smiles_descriptors():
    if RDKIT_AVAILABLE:
        # Aspirin: CC(=O)Oc1ccccc1C(=O)O
        res = calculate_descriptors_from_smiles("CC(=O)Oc1ccccc1C(=O)O")
        assert res["success"] is True
        assert abs(res["mol_MW"] - 180.16) < 0.1
        assert res["mol_Hacceptors"] == 3
        assert res["mol_Hdonors"] == 1


def test_desirability_functions():
    d_size = calculate_desirability_size(150.0, target_size=160.0)
    assert 0.7 <= d_size <= 1.0
    
    d_ee = calculate_desirability_ee(85.0, min_ee=70.0)
    assert 0.7 <= d_ee <= 1.0


def test_pareto_front():
    df = pd.DataFrame({
        'pred_size': [120, 150, 180, 200],
        'pred_EE': [70, 85, 90, 80]
    })
    res_df = find_pareto_front(df, size_col='pred_size', ee_col='pred_EE')
    assert 'is_pareto' in res_df.columns
    assert bool(res_df.loc[0, 'is_pareto']) is True  # 120 nm, 70% EE is on front


def test_ensemble_and_applicability_domain():
    # Create synthetic dataset
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(50, len(PLGA_FEATURES)), columns=PLGA_FEATURES)
    y = np.random.uniform(100, 300, 50)
    
    ens = NanoparticleEnsembleRegressor("Size (nm)", PLGA_FEATURES)
    ens.fit(X, y)
    
    # Test prediction with uncertainty
    y_mean, y_std, ci_l, ci_u = ens.predict(X, return_std=True)
    assert len(y_mean) == 50
    assert np.all(ci_l <= y_mean)
    assert np.all(y_mean <= ci_u)
    
    # Test applicability domain
    ad = ApplicabilityDomain(PLGA_FEATURES)
    ad.fit(X)
    check_res = ad.check(X.iloc[0].to_dict())
    assert "status" in check_res
    assert check_res["status"] == "In Domain (High Reliability)"


def test_pdf_report_generation():
    drug_props = get_drug_by_name("Curcumin")
    top_recs = [{
        'polymer_MW': 50.0,
        'LA/GA': 1.0,
        'drug/polymer': 0.1,
        'surfactant_concentration': 1.0,
        'aqueous/organic': 4.0,
        'pred_size': 150.0,
        'pred_size_std': 10.0,
        'pred_EE': 80.0,
        'pred_ee_std': 4.0,
        'pred_LC': 8.0,
        'ad_status': 'In Domain'
    }]
    sop = generate_plga_lab_sop(top_recs[0], 10.0, "Curcumin")
    pdf_bytes = generate_formulation_pdf_report("Curcumin", drug_props, top_recs, sop, "PLGA")
    assert len(pdf_bytes) > 2000
    assert pdf_bytes.startswith(b"%PDF")
