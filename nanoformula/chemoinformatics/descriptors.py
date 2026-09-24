"""
Molecular Descriptors Engine and Chemoinformatics Utilities using RDKit & PubChem API.
Calculates physicochemical descriptors required for Nanoparticle Formulations.
"""

import requests
import numpy as np
from typing import Dict, Any, Optional, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Draw
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def estimate_melting_point(mw: float, logp: float, tpsa: float, hbd: int, aromatic_rings: int) -> float:
    """
    Empirical QSPR estimation of melting point (°C) based on molecular weight,
    hydrogen bonding, polarity, and crystal lattice packing parameters (Yalkowsky approximation).
    """
    # Base estimate for organic drug-like small molecules
    base_mp = 80.0 + (0.15 * mw) + (12.0 * aromatic_rings) + (15.0 * hbd) + (0.25 * tpsa) - (3.0 * logp)
    return float(np.clip(base_mp, 40.0, 380.0))


def calculate_descriptors_from_smiles(smiles: str) -> Dict[str, Any]:
    """
    Calculates exact physicochemical descriptors from a SMILES string using RDKit.
    Returns dictionary with all 7 core PLGA formulation features and additional structural metrics.
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit is not installed in the environment.")

    clean_smiles = smiles.strip()
    mol = Chem.MolFromSmiles(clean_smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: '{smiles}'. Please verify the chemical structure.")

    # Core PLGA model descriptors
    mol_mw = float(Descriptors.MolWt(mol))
    mol_logp = float(Descriptors.MolLogP(mol))
    mol_tpsa = float(Descriptors.TPSA(mol))
    mol_hacc = int(Lipinski.NumHAcceptors(mol))
    mol_hdon = int(Lipinski.NumHDonors(mol))
    mol_het = int(rdMolDescriptors.CalcNumHeteroatoms(mol))
    
    # Structural features for advanced screening & MP estimation
    num_rotatable = int(Lipinski.NumRotatableBonds(mol))
    num_aromatic = int(Lipinski.NumAromaticRings(mol))
    num_rings = int(Lipinski.RingCount(mol))
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    fraction_csp3 = float(Descriptors.FractionCSP3(mol))
    
    # Estimate melting point
    est_mp = round(estimate_melting_point(mol_mw, mol_logp, mol_tpsa, mol_hdon, num_aromatic), 1)

    return {
        "success": True,
        "smiles": Chem.MolToSmiles(mol),
        "mol_MW": round(mol_mw, 2),
        "mol_logP": round(mol_logp, 2),
        "mol_TPSA": round(mol_tpsa, 2),
        "mol_melting_point": est_mp,
        "mol_Hacceptors": mol_hacc,
        "mol_Hdonors": mol_hdon,
        "mol_heteroatoms": mol_het,
        "num_rotatable_bonds": num_rotatable,
        "num_aromatic_rings": num_aromatic,
        "num_rings": num_rings,
        "heavy_atom_count": heavy_atoms,
        "fraction_csp3": round(fraction_csp3, 2),
        "formula": rdMolDescriptors.CalcMolFormula(mol)
    }


def fetch_drug_from_pubchem(drug_name: str) -> Optional[Dict[str, Any]]:
    """
    Fetches chemical information and canonical SMILES from PubChem REST PUG API.
    """
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(drug_name)}/JSON"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            compound = data['PC_Compounds'][0]
            
            # Extract SMILES
            canonical_smiles = None
            for prop in compound.get('props', []):
                urn = prop.get('urn', {})
                if urn.get('label') == 'SMILES' and urn.get('name') == 'Canonical':
                    canonical_smiles = prop.get('value', {}).get('sval')
                    break
            
            if canonical_smiles:
                desc = calculate_descriptors_from_smiles(canonical_smiles)
                desc['name'] = drug_name.title()
                desc['cid'] = compound.get('id', {}).get('id', {}).get('cid')
                return desc
    except Exception:
        pass
    return None


def generate_structure_svg(smiles: str, width: int = 350, height: int = 250) -> Optional[str]:
    """
    Generates a crisp 2D chemical structure SVG rendering from a SMILES string.
    """
    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = False
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return None
