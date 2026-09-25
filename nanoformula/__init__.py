"""
NanoFormula AI: Multi-Polymer Nanoparticle Formulation Optimizer & Virtual Screening Platform.
Department of Pharmaceutical Engineering & Technology, IIT (BHU) Varanasi.
"""

__version__ = "2.0.0"
__author__ = "Hardik Sood & Dr. Ruchi Chawla"

# Chemoinformatics
from .chemoinformatics import (
    calculate_descriptors_from_smiles,
    fetch_drug_from_pubchem,
    generate_structure_svg,
    get_drug_names,
    get_drug_by_name,
    CURATED_DRUG_DATABASE,
    RDKIT_AVAILABLE
)

# Core Optimization
from .optimization import (
    PLGAFormulationOptimizer,
    ChitosanFormulationOptimizer,
    find_pareto_front,
    calculate_desirability_size,
    calculate_desirability_ee
)

# ML & Uncertainty
from .ml import (
    NanoparticleEnsembleRegressor,
    ApplicabilityDomain,
    ModelExplainer,
    PLGA_FEATURES,
    CHITOSAN_FEATURES,
    FEATURE_DISPLAY_NAMES
)

# 4D Drug Release Kinetics
from .kinetics import (
    DrugReleasePredictor,
    fit_all_kinetic_models,
    classify_release_mechanism,
    zero_order,
    first_order,
    higuchi,
    korsmeyer_peppas,
    hixson_crowell
)

# Multi-Polymer & mRNA LNPs
from .polymers import (
    LNPOptimizer,
    PEGPLGAModel,
    PCLModel,
    LIPID_COMPONENTS_DATABASE
)

# Active Learning & Lab-in-the-Loop
from .active_learning import (
    ActiveLearningEngine,
    FEEDBACK_DB_PATH
)

# 3D Visualization
from .visualization3d import (
    Molecule3DEngine
)

# Literature Validation
from .validation import (
    LiteratureValidator,
    PUBLISHED_LITERATURE_CASE_STUDIES
)

# Protocols & Reports
from .protocols import (
    generate_plga_lab_sop,
    generate_chitosan_lab_sop,
    generate_formulation_pdf_report
)

__all__ = [
    "__version__",
    "calculate_descriptors_from_smiles",
    "fetch_drug_from_pubchem",
    "generate_structure_svg",
    "get_drug_names",
    "get_drug_by_name",
    "CURATED_DRUG_DATABASE",
    "RDKIT_AVAILABLE",
    "PLGAFormulationOptimizer",
    "ChitosanFormulationOptimizer",
    "find_pareto_front",
    "calculate_desirability_size",
    "calculate_desirability_ee",
    "NanoparticleEnsembleRegressor",
    "ApplicabilityDomain",
    "ModelExplainer",
    "PLGA_FEATURES",
    "CHITOSAN_FEATURES",
    "FEATURE_DISPLAY_NAMES",
    "DrugReleasePredictor",
    "fit_all_kinetic_models",
    "classify_release_mechanism",
    "zero_order",
    "first_order",
    "higuchi",
    "korsmeyer_peppas",
    "hixson_crowell",
    "LNPOptimizer",
    "PEGPLGAModel",
    "PCLModel",
    "LIPID_COMPONENTS_DATABASE",
    "ActiveLearningEngine",
    "FEEDBACK_DB_PATH",
    "Molecule3DEngine",
    "LiteratureValidator",
    "PUBLISHED_LITERATURE_CASE_STUDIES",
    "generate_plga_lab_sop",
    "generate_chitosan_lab_sop",
    "generate_formulation_pdf_report"
]
