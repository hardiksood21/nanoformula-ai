"""
Comprehensive automated unit tests for NanoFormula AI advanced modules:
- 4D Drug Release Kinetics & Korsmeyer-Peppas mechanism classification
- mRNA Lipid Nanoparticles (LNP) N/P stoichiometry and formulation
- PEG-PLGA stealth grafting density and PCL degradation
- Active Learning feedback ingestion and Gaussian Process surrogate updates
- 3D molecular conformer generation and WebGL component creation
- Hansen Solubility Parameters (HSP) & Flory-Huggins compatibility engine
- High-Throughput Virtual Screening (HTVS) on 50+ FDA drug library
- 16-Study Literature validation & Meta-analysis metrics
- Top-level `import nanoformula as nf` namespace validation
"""

import os
import json
import pytest
import numpy as np
import pandas as pd

import nanoformula as nf


def test_top_level_package_imports():
    """Verifies that all core classes and functions are cleanly accessible from nanoformula."""
    assert nf.__version__ == "2.1.0"
    assert hasattr(nf, "DrugReleasePredictor")
    assert hasattr(nf, "LNPOptimizer")
    assert hasattr(nf, "PEGPLGAModel")
    assert hasattr(nf, "PCLModel")
    assert hasattr(nf, "ActiveLearningEngine")
    assert hasattr(nf, "Molecule3DEngine")
    assert hasattr(nf, "LiteratureValidator")
    assert hasattr(nf, "PLGAFormulationOptimizer")
    assert hasattr(nf, "ChitosanFormulationOptimizer")
    assert hasattr(nf, "HSPEngine")
    assert hasattr(nf, "HTVSScreeningEngine")


def test_drug_release_kinetics():
    """Tests 4D drug release simulation and kinetic model curve fitting."""
    predictor = nf.DrugReleasePredictor()
    formulation = {
        "polymer_MW": 45.0,
        "LA/GA": 1.0,
        "mol_logP": 3.2,
        "mol_MW": 368.38,
        "pred_size": 150.0,
        "pred_EE": 85.0,
        "surfactant_concentration": 1.0
    }
    res = predictor.predict_release_curve(formulation, release_ph=7.4)
    
    assert "cumulative_release" in res
    assert len(res["cumulative_release"]) == len(res["time_points"])
    assert res["burst_2h_percent"] > 0.0
    assert res["release_24h_percent"] >= res["burst_2h_percent"]
    assert res["release_120h_percent"] <= 100.0
    
    # Check kinetic fits
    fits = res["kinetic_fits"]
    assert "Korsmeyer-Peppas" in fits
    assert "Higuchi" in fits
    assert "First-Order" in fits
    assert "release_exponent_n" in fits["Korsmeyer-Peppas"]
    assert "mechanism" in fits["Korsmeyer-Peppas"]


def test_korsmeyer_peppas_classification():
    """Verifies physical mechanism classification based on Peppas exponent."""
    m_fickian = nf.classify_release_mechanism(0.35)
    assert "Fickian" in m_fickian["mechanism"]
    
    m_anomalous = nf.classify_release_mechanism(0.65)
    assert "Anomalous" in m_anomalous["mechanism"]
    
    m_case2 = nf.classify_release_mechanism(0.92)
    assert "Case II" in m_case2["mechanism"]


def test_mrna_lnp_optimizer():
    """Tests 4-component mRNA Lipid Nanoparticle stoichiometric calculations and N/P ratio."""
    lnp_opt = nf.LNPOptimizer()
    res = lnp_opt.optimize_lnp(
        ionizable_lipid_name="SM-102",
        helper_lipid_name="DSPC",
        peg_lipid_name="DMG-PEG2000",
        target_np_ratio=6.0,
        target_size_nm=80.0,
        mrna_dose_ug=50.0,
        batch_volume_ml=5.0
    )
    
    assert res["np_ratio"] == 6.0
    assert res["predicted_cqas"]["pred_ee_percent"] > 90.0
    assert 60.0 <= res["predicted_cqas"]["pred_size_nm"] <= 110.0
    assert "lipid_masses_mg" in res
    assert res["lipid_masses_mg"]["total_lipid_mass_mg"] > 0.0
    assert len(res["sop_steps"]) == 5


def test_peg_plga_and_pcl_models():
    """Tests stealth PEG-PLGA surface density and PCL sustained delivery calculations."""
    peg_model = nf.PEGPLGAModel()
    peg_res = peg_model.evaluate_stealth_properties(
        plga_mw_kda=40.0,
        peg_mw_kda=5.0,
        peg_weight_fraction_percent=10.0,
        core_diameter_nm=120.0
    )
    assert peg_res["flory_radius_rf_nm"] > 0.0
    assert peg_res["peg_grafting_density_chains_nm2"] > 0.0
    assert "Regime" in peg_res["conformation_regime"]
    
    pcl_model = nf.PCLModel()
    pcl_res = pcl_model.evaluate_pcl_formulation(pcl_mw_kda=45.0, drug_logp=3.5, drug_mw=400.0)
    assert "Polycaprolactone" in pcl_res["polymer"]
    assert pcl_res["predicted_ee_percent"] > 50.0


def test_hansen_solubility_engine():
    """Tests Hansen Solubility Parameter estimation, distance Ra, and Flory-Huggins calculation."""
    # Curcumin SMILES
    curcumin_smiles = "COC1=C(C=CC(=C1)C=CC(=O)CC(=O)C=CC2=CC(=C(C=C2)O)OC)O"
    hsp = nf.HSPEngine.estimate_drug_hsp(curcumin_smiles)
    
    assert hsp["delta_D"] > 15.0
    assert hsp["delta_P"] > 0.0
    assert hsp["delta_H"] > 0.0
    assert hsp["delta_total"] > 0.0
    
    # Test compatibility with PLGA 50:50
    compat = nf.HSPEngine.calculate_compatibility(hsp, polymer_key="PLGA 50:50")
    assert "hansen_distance_Ra" in compat
    assert "relative_energy_difference_RED" in compat
    assert compat["hansen_distance_Ra"] > 0.0
    assert compat["relative_energy_difference_RED"] > 0.0
    assert compat["max_thermodynamic_loading_percent"] > 0.0

    # Test screening across all polymers
    all_polys = nf.HSPEngine.screen_all_polymers(hsp)
    assert len(all_polys) >= 5
    assert all_polys[0]["relative_energy_difference_RED"] <= all_polys[-1]["relative_energy_difference_RED"]


def test_htvs_screening_engine():
    """Tests High-Throughput Virtual Screening pipeline on FDA drug library."""
    bundle_path = os.path.join("saved_models", "nanoformula_models_bundle.pkl")
    if os.path.exists(bundle_path):
        import pickle
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)
        htvs = nf.HTVSScreeningEngine(bundle)
    else:
        htvs = nf.HTVSScreeningEngine()

    df_screen = htvs.screen_library(target_polymer="PLGA 50:50")
    assert isinstance(df_screen, pd.DataFrame)
    assert len(df_screen) >= 20
    assert "Feasibility Score (NFFS)" in df_screen.columns
    assert "Predicted EE (%)" in df_screen.columns
    assert "Hansen RED" in df_screen.columns
    assert df_screen["Feasibility Score (NFFS)"].max() <= 100.0


def test_active_learning_feedback_engine(tmp_path):
    """Tests real experimental batch submission, database persistence, and Bayesian GP surrogate update."""
    test_db = os.path.join(tmp_path, "test_feedback.json")
    al_engine = nf.ActiveLearningEngine(db_path=test_db)
    
    submission = al_engine.submit_experimental_batch(
        researcher_name="Test Researcher",
        institution="IIT BHU",
        polymer_system="PLGA",
        drug_name="Curcumin",
        formulation_parameters={"polymer_MW": 24.0, "drug/polymer": 0.1, "surfactant_concentration": 1.0},
        measured_particle_size_nm=155.2,
        measured_ee_percent=81.0,
        measured_pdi=0.18,
        instrument_notes="Malvern Zetasizer Nano ZS"
    )
    
    assert submission["status"] == "SUCCESS"
    entries = al_engine.get_all_entries()
    assert len(entries) == 1
    assert entries[0]["experimental_results"]["particle_size_nm"] == 155.2
    
    stats = al_engine.compute_active_learning_metrics()
    assert stats["total_batches_contributed"] == 1


def test_molecule_3d_engine():
    """Tests RDKit 3D force-field embedding and WebGL viewer generation."""
    if nf.RDKIT_AVAILABLE:
        m3d = nf.Molecule3DEngine()
        molblock = m3d.generate_3d_molblock("CC(=O)Oc1ccccc1C(=O)O") # Aspirin
        assert molblock is not None
        assert "3D" in molblock or "V2000" in molblock or "M  END" in molblock
        
        html_viewer = m3d.create_3dmol_viewer_html("CC(=O)Oc1ccccc1C(=O)O")
        assert "3Dmol" in html_viewer
        assert "$3Dmol.createViewer" in html_viewer
        
        svg_coreshell = m3d.create_coreshell_nanoparticle_svg("PLGA", "PVA", "Curcumin")
        assert "<svg" in svg_coreshell


def test_literature_validation_and_meta_analysis():
    """Tests 16-study literature benchmark execution and meta-analysis statistical metrics."""
    bundle_path = os.path.join("saved_models", "nanoformula_models_bundle.pkl")
    if os.path.exists(bundle_path):
        import pickle
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)
        validator = nf.LiteratureValidator(bundle)
        df_val = validator.run_literature_validation()
        
        assert len(df_val) == 16
        assert "Exp. Size (nm)" in df_val.columns
        assert "AI Pred. Size (nm)" in df_val.columns
        
        # Meta-analysis statistics
        meta_stats = validator.compute_meta_analysis_statistics(df_val)
        assert meta_stats["n_studies"] == 16
        assert meta_stats["size_metrics"]["R2"] > 0.50
        assert meta_stats["size_metrics"]["pearson_r"] > 0.60
        assert meta_stats["size_metrics"]["mape_percent"] < 50.0
