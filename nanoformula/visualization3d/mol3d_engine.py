"""
3D Molecular Conformer Generation & Core-Shell Nanoparticle 3D Visualizer.
Uses RDKit MMFF force-field optimization and generates 3Dmol.js WebGL rendering components.
"""

from typing import Dict, Any, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class Molecule3DEngine:
    """
    Generates optimized 3D molecular conformers using RDKit MMFF force fields
    and creates interactive WebGL visualization widgets.
    """
    def __init__(self):
        pass

    def generate_3d_molblock(self, smiles: str) -> Optional[str]:
        """
        Generates energy-minimized 3D conformer coordinates (MolBlock) from SMILES.
        """
        if not RDKIT_AVAILABLE:
            return None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol_h = Chem.AddHs(mol)
            res = AllChem.EmbedMolecule(mol_h, randomSeed=42)
            if res != 0:
                AllChem.EmbedMolecule(mol_h, useRandomCoords=True, randomSeed=42)
            
            # MMFF94 force field energy minimization
            try:
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=500)
            except Exception:
                pass

            return Chem.MolToMolBlock(mol_h)
        except Exception:
            return None

    def create_3dmol_viewer_html(
        self,
        smiles: str,
        style: str = "stick",
        bg_color: str = "#1A1A24",
        width: str = "100%",
        height: int = 380
    ) -> str:
        """
        Builds self-contained HTML/JS snippet integrating 3Dmol.js WebGL viewer.
        """
        molblock = self.generate_3d_molblock(smiles)
        if not molblock:
            return "<div style='color:red;text-align:center;padding:20px;'>Could not generate 3D conformer from SMILES.</div>"

        # Escape backticks and newlines for JS string template
        molblock_escaped = molblock.replace('\\', '\\\\').replace('`', '\\`').replace('\n', '\\n')

        style_config = "{stick: {radius: 0.15, colorscheme: 'Jmol'}}" if style == "stick" else (
            "{sphere: {scale: 0.28, colorscheme: 'Jmol'}, stick: {radius: 0.12}}" if style == "ball_and_stick" else
            "{sphere: {colorscheme: 'Jmol'}}"
        )

        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"></script>
            <style>
                body {{ margin: 0; padding: 0; background-color: {bg_color}; font-family: sans-serif; }}
                #viewport {{ width: {width}; height: {height}px; position: relative; border-radius: 8px; overflow: hidden; }}
                .ctrl-badge {{ position: absolute; bottom: 8px; right: 10px; color: #888; font-size: 11px; z-index: 10; pointer-events: none; }}
            </style>
        </head>
        <body>
            <div id="viewport"></div>
            <div class="ctrl-badge">🖱️ Rotate: Left Click | Zoom: Scroll | Pan: Right Click</div>
            <script>
                document.addEventListener("DOMContentLoaded", function() {{
                    var element = document.getElementById('viewport');
                    var config = {{ backgroundColor: '{bg_color}' }};
                    var viewer = $3Dmol.createViewer(element, config);
                    var molData = `{molblock_escaped}`;
                    
                    viewer.addModel(molData, "mol");
                    viewer.setStyle({{}}, {style_config});
                    viewer.zoomTo();
                    viewer.render();
                    viewer.spin(true);
                }});
            </script>
        </body>
        </html>
        """
        return html_code

    def create_coreshell_nanoparticle_svg(
        self,
        core_polymer: str = "PLGA Hydrophobic Matrix",
        corona_polymer: str = "PVA Surfactant / PEG Corona",
        drug_name: str = "Entrapped Drug Molecules",
        core_color: str = "#2E86AB",
        corona_color: str = "#27AE60"
    ) -> str:
        """
        Generates a cross-section schematic SVG of a drug-loaded core-shell nanoparticle micelle.
        """
        svg = f"""
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 450 320" width="100%" height="300">
            <defs>
                <radialGradient id="coreGrad" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stop-color="#1B4965" />
                    <stop offset="85%" stop-color="{core_color}" />
                    <stop offset="100%" stop-color="#143642" />
                </radialGradient>
                <linearGradient id="coronaGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="{corona_color}" stop-opacity="0.8" />
                    <stop offset="100%" stop-color="{corona_color}" stop-opacity="0.2" />
                </linearGradient>
            </defs>

            <!-- Background -->
            <rect width="450" height="320" fill="#F8F9FA" rx="10" />

            <!-- Hydrophilic Corona / PEG Brush Layer -->
            <circle cx="180" cy="160" r="125" fill="url(#coronaGrad)" stroke="{corona_color}" stroke-width="2.5" stroke-dasharray="6,4" />

            <!-- Hydrophobic Polymer Core -->
            <circle cx="180" cy="160" r="85" fill="url(#coreGrad)" stroke="#1B4965" stroke-width="3" />

            <!-- Entrapped Drug Molecules -->
            <circle cx="150" cy="140" r="7" fill="#E74C3C" stroke="#fff" stroke-width="1.5" />
            <circle cx="210" cy="155" r="8" fill="#E74C3C" stroke="#fff" stroke-width="1.5" />
            <circle cx="175" cy="185" r="7" fill="#E74C3C" stroke="#fff" stroke-width="1.5" />
            <circle cx="185" cy="120" r="6" fill="#E74C3C" stroke="#fff" stroke-width="1.5" />
            <circle cx="140" cy="175" r="8" fill="#E74C3C" stroke="#fff" stroke-width="1.5" />

            <!-- Surfactant / PEG Tails -->
            <path d="M 180 35 Q 170 20 180 5" stroke="{corona_color}" stroke-width="2" fill="none" />
            <path d="M 270 75 Q 285 65 295 55" stroke="{corona_color}" stroke-width="2" fill="none" />
            <path d="M 305 160 Q 320 160 335 155" stroke="{corona_color}" stroke-width="2" fill="none" />
            <path d="M 270 245 Q 285 255 295 265" stroke="{corona_color}" stroke-width="2" fill="none" />
            <path d="M 180 285 Q 170 300 180 315" stroke="{corona_color}" stroke-width="2" fill="none" />

            <!-- Legend & Labels -->
            <rect x="300" y="30" width="135" height="150" fill="#ffffff" stroke="#CCCCCC" rx="6" />
            <text x="310" y="50" font-family="sans-serif" font-size="11" font-weight="bold" fill="#1B4965">Architecture</text>
            
            <circle cx="318" cy="72" r="6" fill="#E74C3C" />
            <text x="330" y="76" font-family="sans-serif" font-size="10" fill="#333333">{drug_name[:14]}</text>

            <circle cx="318" cy="98" r="6" fill="{core_color}" />
            <text x="330" y="102" font-family="sans-serif" font-size="10" fill="#333333">Core: {core_polymer[:10]}</text>

            <circle cx="318" cy="124" r="6" fill="{corona_color}" />
            <text x="330" y="128" font-family="sans-serif" font-size="10" fill="#333333">Corona: Surfactant</text>

            <text x="310" y="160" font-family="sans-serif" font-size="9" fill="#777777">DLS Size: Core + Shell</text>
        </svg>
        """
        return svg
