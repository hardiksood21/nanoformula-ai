"""
NanoFormula AI 2.0: Multi-Polymer Nanoparticle Formulation Optimizer & Virtual Screening Platform
Department of Pharmaceutical Engineering & Technology, IIT (BHU) Varanasi
Supervisor: Dr. Ruchi Chawla | Developer: Hardik Sood
"""

import os
import io
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Unified Package Imports
import nanoformula as nf

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==============================================================================
st.set_page_config(
    page_title="NanoFormula AI | Nanoparticle Formulation Optimizer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 32px;
        font-weight: 800;
        color: #1B4965;
        text-align: center;
        padding-top: 5px;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 15px;
        color: #555;
        text-align: center;
        margin-bottom: 18px;
        font-weight: 500;
    }
    .metric-card {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        justify-content: center;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 15px;
        font-weight: 600;
        padding: 6px 14px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. MODEL BUNDLE LOADER
# ==============================================================================
@st.cache_resource
def load_models_bundle():
    bundle_path = os.path.join("saved_models", "nanoformula_models_bundle.pkl")
    if not os.path.exists(bundle_path):
        from nanoformula.ml.trainer import train_and_save_all
        bundle = train_and_save_all()
    else:
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)
    return bundle

bundle = load_models_bundle()
plga_optimizer = nf.PLGAFormulationOptimizer(bundle)
chitosan_optimizer = nf.ChitosanFormulationOptimizer(bundle)
release_predictor = nf.DrugReleasePredictor()
lnp_optimizer = nf.LNPOptimizer()
peg_plga_model = nf.PEGPLGAModel()
pcl_model = nf.PCLModel()
active_learning_engine = nf.ActiveLearningEngine()
mol3d_engine = nf.Molecule3DEngine()
lit_validator = nf.LiteratureValidator(bundle)

# ==============================================================================
# 3. HEADER
# ==============================================================================
st.markdown('<div class="main-header">🧬 NanoFormula AI 2.0</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Polymer Nanoparticle Formulation Optimizer & 4D Release Kinetics Engine<br/><b>Dr. Ruchi Chawla\'s Lab</b> | Department of Pharmaceutical Engineering & Technology, IIT (BHU) Varanasi</div>', unsafe_allow_html=True)

# 9 Comprehensive Feature Tabs
tab_plga, tab_chitosan, tab_lnp, tab_kinetics, tab_stealth, tab_screening, tab_active, tab_benchmarks, tab_about = st.tabs([
    "🎯 PLGA Optimizer",
    "🧪 Chitosan-TPP",
    "🧬 mRNA Lipid Nanoparticles (LNP)",
    "📈 4D Release Kinetics",
    "🛡️ PEG-PLGA & PCL",
    "⚡ Virtual Screening",
    "🔄 Lab-in-the-Loop AI",
    "🔬 Benchmarks & Validation",
    "📄 Research & Citations"
])


# ==============================================================================
# TAB 1: PLGA NANOPARTICLE OPTIMIZER
# ==============================================================================
with tab_plga:
    col_input, col_results = st.columns([1, 2.2], gap="large")

    with col_input:
        st.markdown("### 💊 1. Drug (API) Input")
        
        input_mode = st.radio(
            "Input Method:",
            ["📚 Curated Drug Library", "🔍 PubChem Live Search", "🧪 SMILES Structure", "⚙️ Manual Descriptors"],
            horizontal=False
        )

        drug_name = "Target API"
        drug_smiles = None
        drug_props = {}

        if input_mode == "📚 Curated Drug Library":
            drug_list = nf.get_drug_names()
            selected_name = st.selectbox("Select Drug Compound:", drug_list, index=drug_list.index("Curcumin") if "Curcumin" in drug_list else 0)
            drug_info = nf.get_drug_by_name(selected_name)
            drug_name = drug_info["name"]
            drug_smiles = drug_info["smiles"]
            drug_props = {
                'mol_MW': drug_info['mol_MW'],
                'mol_logP': drug_info['mol_logP'],
                'mol_TPSA': drug_info['mol_TPSA'],
                'mol_melting_point': drug_info['mol_melting_point'],
                'mol_Hacceptors': drug_info['mol_Hacceptors'],
                'mol_Hdonors': drug_info['mol_Hdonors'],
                'mol_heteroatoms': drug_info['mol_heteroatoms']
            }
            st.caption(f"**Class:** {drug_info.get('therapeutic_class', '')} | **Status:** {drug_info.get('clinical_stage', '')}")

        elif input_mode == "🔍 PubChem Live Search":
            query_name = st.text_input("Enter Drug / Chemical Name:", "Paclitaxel")
            if query_name:
                with st.spinner("Searching PubChem & calculating RDKit descriptors..."):
                    fetched = nf.fetch_drug_from_pubchem(query_name)
                    if fetched:
                        drug_name = fetched.get('name', query_name)
                        drug_smiles = fetched.get('smiles')
                        drug_props = {k: fetched[k] for k in ['mol_MW', 'mol_logP', 'mol_TPSA', 'mol_melting_point', 'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms']}
                        st.success(f"Found **{drug_name}** in PubChem!")
                    else:
                        st.error(f"Could not retrieve '{query_name}' from PubChem. Try entering SMILES directly.")

        elif input_mode == "🧪 SMILES Structure":
            smiles_input = st.text_area("Paste SMILES string:", "COC1=C(C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC)O", height=60)
            drug_custom_name = st.text_input("Compound Name / Identifier:", "Curcumin Analog")
            if smiles_input:
                try:
                    calc_res = nf.calculate_descriptors_from_smiles(smiles_input)
                    drug_name = drug_custom_name
                    drug_smiles = calc_res['smiles']
                    drug_props = {k: calc_res[k] for k in ['mol_MW', 'mol_logP', 'mol_TPSA', 'mol_melting_point', 'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms']}
                    st.success(f"RDKit calculated descriptors for formula **{calc_res.get('formula')}**")
                except Exception as e:
                    st.error(f"Error parsing SMILES: {str(e)}")

        elif input_mode == "⚙️ Manual Descriptors":
            drug_name = st.text_input("Compound Identifier:", "Custom Molecule")
            c1, c2 = st.columns(2)
            with c1:
                mw = st.number_input("Mol. Weight (g/mol)", 50.0, 1500.0, 368.38, 5.0)
                logp = st.number_input("LogP (Lipophilicity)", -5.0, 10.0, 3.20, 0.1)
                tpsa = st.number_input("TPSA (Å²)", 0.0, 400.0, 93.06, 5.0)
                mp = st.number_input("Melting Point (°C)", 20.0, 500.0, 183.0, 5.0)
            with c2:
                hacc = st.number_input("H-Acceptors", 0, 30, 6, 1)
                hdon = st.number_input("H-Donors", 0, 20, 2, 1)
                het = st.number_input("Heteroatoms", 0, 40, 6, 1)
            drug_props = {
                'mol_MW': mw, 'mol_logP': logp, 'mol_TPSA': tpsa,
                'mol_melting_point': mp, 'mol_Hacceptors': hacc,
                'mol_Hdonors': hdon, 'mol_heteroatoms': het
            }

        # 3D/2D Molecular Depiction
        if drug_smiles and nf.RDKIT_AVAILABLE:
            st.markdown("#### 🔬 3D Molecular Conformer (WebGL)")
            html_3d = mol3d_engine.create_3dmol_viewer_html(drug_smiles, style="ball_and_stick", height=240)
            components.html(html_3d, height=250)

        st.markdown("---")
        st.markdown("### ⚙️ 2. Formulation Objectives")
        target_size = st.slider("Target Hydrodynamic Size (nm)", 50, 300, 160, 10)
        min_ee = st.slider("Minimum Entrapment Efficiency (% EE)", 40, 95, 70, 5)
        
        polymer_choice = st.selectbox(
            "PLGA Molecular Weight Constraints:",
            ["Auto-Explore Full Space", "Constrain to Low MW (5-15 kDa)", "Constrain to Med MW (24-50 kDa)", "Constrain to High MW (65-110 kDa)"]
        )
        p_mw_constraint = 10.0 if "Low" in polymer_choice else (30.0 if "Med" in polymer_choice else (80.0 if "High" in polymer_choice else None))
        n_recs = st.slider("Number of Diverse Candidates", 3, 8, 5)
        run_opt = st.button("🚀 OPTIMIZE PLGA FORMULATION", type="primary", use_container_width=True)

    with col_results:
        if run_opt and drug_props:
            with st.spinner("Running Pareto Candidate Search & Ensemble ML Inference..."):
                opt_results = plga_optimizer.optimize(
                    drug_properties=drug_props,
                    target_size=target_size,
                    min_ee=min_ee,
                    n_recommendations=n_recs,
                    constrained_polymer_mw=p_mw_constraint,
                    n_candidates=20000
                )
                st.session_state['plga_results'] = opt_results
                st.session_state['current_drug_name'] = drug_name
                st.session_state['current_drug_props'] = drug_props

        if 'plga_results' in st.session_state:
            results = st.session_state['plga_results']
            recs = results['recommendations']
            curr_drug = st.session_state.get('current_drug_name', 'Drug')
            curr_props = st.session_state.get('current_drug_props', drug_props)

            st.success(f"✅ Optimization Complete for **{curr_drug}**!")

            # Summary Metrics Row
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Formulations Evaluated", f"{results['n_candidates']:,}")
            m2.metric("Pareto Optimal Set", f"{len(results['pareto_candidates']):,}")
            m3.metric("Best Size", f"{recs['pred_size'].min():.1f} nm")
            m4.metric("Best EE%", f"{recs['pred_EE'].max():.1f}%")
            m5.metric("Best Loading", f"{recs['pred_LC'].max():.1f}%")

            # Recommendations Table
            st.markdown(f"### 🎯 Top Recommended Formulations for {curr_drug}")
            display_df = recs[['drug/polymer', 'polymer_MW', 'LA/GA', 'surfactant_concentration', 'aqueous/organic',
                               'pred_size', 'pred_size_std', 'pred_EE', 'pred_ee_std', 'pred_LC', 'ad_status']].copy()
            
            display_df['Predicted Size (nm)'] = [f"{s:.1f} ± {std:.1f}" for s, std in zip(display_df['pred_size'], display_df['pred_size_std'])]
            display_df['Predicted EE (%)'] = [f"{ee:.1f} ± {std:.1f}" for ee, std in zip(display_df['pred_EE'], display_df['pred_ee_std'])]
            display_df['Loading Cap. (%)'] = [f"{lc:.1f}%" for lc in display_df['pred_LC']]
            
            table_show = display_df[['drug/polymer', 'polymer_MW', 'LA/GA', 'surfactant_concentration',
                                     'aqueous/organic', 'Predicted Size (nm)', 'Predicted EE (%)', 'Loading Cap. (%)', 'ad_status']]
            table_show.columns = ['Drug/Polymer', 'PLGA MW (kDa)', 'LA/GA', 'Surf. Conc (%)', 'Aq/Org',
                                  'Predicted Size (nm)', 'Predicted EE (%)', 'Loading Cap. (%)', 'Applicability Domain']

            st.dataframe(table_show, use_container_width=True, hide_index=False)

            # Interactive Plotly Charts
            t_chart1, t_chart2, t_shap, t_coreshell = st.tabs(["📈 Pareto Frontier Trade-Off", "📊 Size Distribution", "🧠 SHAP Explainability", "🔬 Core-Shell Architecture"])

            with t_chart1:
                cand_df = results['all_candidates']
                fig_pareto = px.scatter(
                    cand_df.sample(min(3000, len(cand_df)), random_state=42),
                    x='pred_size', y='pred_EE', color='pred_LC', size='desirability',
                    hover_data=['polymer_MW', 'drug/polymer', 'surfactant_concentration', 'ad_status'],
                    labels={'pred_size': 'Predicted Particle Size (nm)', 'pred_EE': 'Predicted EE (%)', 'pred_LC': 'Loading Capacity (%)'},
                    title="Formulation Space & Pareto Optimal Frontier", color_continuous_scale="Viridis"
                )
                fig_pareto.add_vline(x=target_size, line_dash="dash", line_color="red", annotation_text="Target Size")
                fig_pareto.add_hline(y=min_ee, line_dash="dash", line_color="blue", annotation_text="Min EE%")
                fig_pareto.add_trace(go.Scatter(
                    x=recs['pred_size'], y=recs['pred_EE'], mode='markers+text',
                    marker=dict(size=14, color='#E74C3C', symbol='star', line=dict(width=1, color='black')),
                    text=[f"#{i+1}" for i in range(len(recs))], textposition="top center", name="Top Recommendations"
                ))
                st.plotly_chart(fig_pareto, use_container_width=True)

            with t_chart2:
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=[f"Form #{i+1}" for i in range(len(recs))], y=recs['pred_size'],
                    error_y=dict(type='data', array=recs['pred_size_std'], visible=True),
                    name="Particle Size (nm)", marker_color='#2E86AB'
                ))
                fig_bar.add_hline(y=target_size, line_dash="dash", line_color="red", annotation_text=f"Target: {target_size} nm")
                fig_bar.update_layout(title="Predicted Hydrodynamic Diameter (± 95% Confidence Interval)", yaxis_title="Size (nm)")
                st.plotly_chart(fig_bar, use_container_width=True)

            with t_shap:
                st.markdown("#### 🧠 TreeSHAP Feature Attribution Breakdown")
                explainer = nf.ModelExplainer(bundle['plga_size'], nf.PLGA_FEATURES, "Particle Size (nm)")
                fig_shap = explainer.plot_instance_breakdown(recs.iloc[0].to_dict(), f"SHAP Feature Attribution (Formulation #1 Size)")
                if fig_shap:
                    st.pyplot(fig_shap)

            with t_coreshell:
                svg_cs = mol3d_engine.create_coreshell_nanoparticle_svg("PLGA Matrix", "PVA Corona", curr_drug)
                st.markdown(svg_cs, unsafe_allow_html=True)

            # Laboratory SOP & Batch Recipe
            st.markdown("---")
            st.markdown("### 🧪 Suggested Laboratory Recipe & SOP")
            batch_vol = st.selectbox("Select Target Batch Volume:", [5.0, 10.0, 25.0, 50.0], index=1)
            
            best_rec = recs.iloc[0].to_dict()
            sop = nf.generate_plga_lab_sop(best_rec, batch_volume_ml=batch_vol, drug_name=curr_drug)
            
            recipe = sop['recipe']
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("PLGA Mass", f"{recipe['polymer_mass_mg']} mg", f"{recipe['polymer_type']}")
            r2.metric(f"{curr_drug} Mass", f"{recipe['drug_mass_mg']} mg", f"D/P: {recipe['drug_polymer_ratio']:.4f}")
            r3.metric("Organic Volume", f"{recipe['organic_phase_volume_ml']} mL", f"{sop['organic_solvent']}")
            r4.metric("Aqueous Volume", f"{recipe['aqueous_phase_volume_ml']} mL", f"Surfactant: {recipe['surfactant_mass_mg']} mg")

            with st.expander("📋 View Detailed Step-by-Step Wet Lab SOP", expanded=True):
                st.markdown(f"**Fabrication Method:** {sop['method']}")
                for step in sop['steps']:
                    st.markdown(f"- {step}")

            # Download Actions
            st.markdown("---")
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                pdf_bytes = nf.generate_formulation_pdf_report(
                    drug_name=curr_drug,
                    drug_properties=curr_props,
                    top_formulations=recs.to_dict(orient='records'),
                    lab_sop=sop,
                    polymer_system="PLGA"
                )
                st.download_button(
                    label="📄 Download Official Laboratory PDF Certificate",
                    data=pdf_bytes,
                    file_name=f"NanoFormula_{curr_drug}_PLGA_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            with d_col2:
                csv_bytes = table_show.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Recommendations CSV",
                    data=csv_bytes,
                    file_name=f"NanoFormula_{curr_drug}_recs.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        else:
            st.info("👈 Select or input your drug properties on the left sidebar and click **'OPTIMIZE PLGA FORMULATION'**.")


# ==============================================================================
# TAB 2: CHITOSAN-TPP NANOPARTICLES
# ==============================================================================
with tab_chitosan:
    st.markdown("### 🧪 Chitosan-TPP Nanoparticle Optimizer")
    st.info("**Predicts:** Hydrodynamic Size (nm), Polydispersity Index (PDI), and Zeta Potential (+mV) using Ionic Gelation modeling.")

    c_left, c_right = st.columns([1, 2.2], gap="large")

    with c_left:
        st.markdown("#### ⚙️ Parameters & Constraints")
        cs_mw_mode = st.radio("Chitosan Selection:", ["Auto-Explore All MW", "5 kDa", "20 kDa", "Low MW (~50 kDa)", "High MW (~310 kDa)"])
        cs_mw_fixed = 5.0 if "5 kDa" in cs_mw_mode else (20.0 if "20 kDa" in cs_mw_mode else (50.0 if "50 kDa" in cs_mw_mode else (310.0 if "310 kDa" in cs_mw_mode else None)))

        cs_target_size = st.slider("Target Size (nm)", 60, 250, 130, 5, key="cs_ts")
        cs_max_pdi = st.slider("Max Desired PDI", 0.15, 0.40, 0.25, 0.01, key="cs_mpdi")
        cs_min_zeta = st.slider("Min Zeta Potential (+mV)", 10.0, 35.0, 20.0, 1.0, key="cs_mzeta")
        cs_n_recs = st.slider("Number of Recommendations", 3, 8, 5, key="cs_nr")

        run_cs_opt = st.button("🚀 OPTIMIZE CHITOSAN", type="primary", use_container_width=True)

    with c_right:
        if run_cs_opt:
            with st.spinner("Optimizing Chitosan-TPP ionic gelation parameters..."):
                cs_results = chitosan_optimizer.optimize(
                    target_size=cs_target_size,
                    max_pdi=cs_max_pdi,
                    min_zeta=cs_min_zeta,
                    n_recommendations=cs_n_recs,
                    constrained_mw=cs_mw_fixed,
                    n_candidates=10000
                )
                st.session_state['cs_results'] = cs_results

        if 'cs_results' in st.session_state:
            cs_res = st.session_state['cs_results']
            recs_cs = cs_res['recommendations']

            st.success("✅ Chitosan-TPP Optimization Complete!")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Candidates Evaluated", f"{cs_res['n_candidates']:,}")
            c2.metric("Best Predicted Size", f"{recs_cs['pred_size'].min():.1f} nm")
            c3.metric("Best PDI", f"{recs_cs['pred_PDI'].min():.3f}")
            c4.metric("Best Zeta Potential", f"+{recs_cs['pred_zeta'].max():.1f} mV")

            display_cs = recs_cs[['chitosan_MW', 'chitosan_conc', 'TPP_conc', 'chitosan_TPP_ratio',
                                  'pred_size', 'pred_size_std', 'pred_PDI', 'pred_zeta', 'ad_status']].copy()
            display_cs['Predicted Size (nm)'] = [f"{s:.1f} ± {std:.1f}" for s, std in zip(display_cs['pred_size'], display_cs['pred_size_std'])]
            display_cs['PDI'] = [f"{p:.3f}" for p in display_cs['pred_PDI']]
            display_cs['Zeta Potential (mV)'] = [f"+{z:.1f} mV" for z in display_cs['pred_zeta']]

            cs_table_show = display_cs[['chitosan_MW', 'chitosan_conc', 'TPP_conc', 'chitosan_TPP_ratio',
                                        'Predicted Size (nm)', 'PDI', 'Zeta Potential (mV)', 'ad_status']]
            cs_table_show.columns = ['Chitosan MW (kDa)', 'CS Conc (mg/mL)', 'TPP Conc (mg/mL)', 'CS:TPP Ratio',
                                     'Predicted Size (nm)', 'PDI', 'Zeta Potential (mV)', 'Applicability Domain']

            st.dataframe(cs_table_show, use_container_width=True)

            # Plots
            fig_cs_scatter = px.scatter(
                recs_cs, x='pred_size', y='pred_zeta', size='chitosan_conc', color='chitosan_MW',
                labels={'pred_size': 'Predicted Particle Size (nm)', 'pred_zeta': 'Zeta Potential (+mV)'},
                title="Chitosan Size vs. Zeta Potential Profile", text=[f"#{i+1}" for i in range(len(recs_cs))]
            )
            st.plotly_chart(fig_cs_scatter, use_container_width=True)

            best_cs = recs_cs.iloc[0].to_dict()
            cs_sop = nf.generate_chitosan_lab_sop(best_cs, batch_volume_ml=10.0)
            
            with st.expander("📋 Suggested Ionic Gelation Lab Protocol", expanded=True):
                for step in cs_sop['steps']:
                    st.markdown(f"- {step}")

            pdf_cs_bytes = nf.generate_formulation_pdf_report(
                drug_name="Blank Chitosan-TPP Nanoparticles",
                drug_properties={'mol_MW': 'N/A', 'mol_logP': 'N/A', 'mol_TPSA': 'N/A', 'mol_melting_point': 'N/A', 'mol_Hacceptors': 'N/A', 'mol_Hdonors': 'N/A'},
                top_formulations=recs_cs.to_dict(orient='records'),
                lab_sop=cs_sop,
                polymer_system="Chitosan"
            )
            st.download_button("📄 Download Chitosan PDF Certificate", pdf_cs_bytes, "NanoFormula_Chitosan_TPP_Report.pdf", "application/pdf")
        else:
            st.info("👈 Set your target size, PDI, and Zeta constraints and click **'OPTIMIZE CHITOSAN'**.")


# ==============================================================================
# TAB 3: mRNA LIPID NANOPARTICLES (LNP) DESIGNER
# ==============================================================================
with tab_lnp:
    st.markdown("### 🧬 mRNA / siRNA Lipid Nanoparticle (LNP) Optimizer")
    st.info("**Clinical Standard:** Microfluidic impingement formulation based on FDA-approved COVID-19 mRNA vaccines (Comirnaty/Spikevax) and RNAi therapeutics (Onpattro).")

    lnp_col1, lnp_col2 = st.columns([1, 2.2], gap="large")

    with lnp_col1:
        st.markdown("#### 🧪 1. Lipid Components")
        ionizable_choice = st.selectbox("Ionizable Cationic Lipid:", ["SM-102 (Moderna standard)", "ALC-0315 (Pfizer standard)", "DLin-MC3-DMA (Alnylam standard)"])
        ionizable_key = "SM-102" if "SM-102" in ionizable_choice else ("ALC-0315" if "ALC-0315" in ionizable_choice else "DLin-MC3-DMA")

        helper_choice = st.selectbox("Helper Phospholipid:", ["DSPC (Structural Bilayer)", "DOPE (Fusogenic Endosomal Escape)"])
        helper_key = "DSPC" if "DSPC" in helper_choice else "DOPE"

        peg_choice = st.selectbox("PEGylated Lipid:", ["DMG-PEG2000 (1.5 mol%)", "ALC-0159 (1.6 mol%)"])
        peg_key = "DMG-PEG2000" if "DMG-PEG2000" in peg_choice else "ALC-0159"

        st.markdown("#### ⚙️ 2. Nucleic Acid & Stoichiometry")
        target_np = st.slider("Target N/P Molar Ratio:", 3.0, 10.0, 6.0, 0.5, help="Molar ratio of positive amine in ionizable lipid to negative phosphate in mRNA. Optimal = 6.0.")
        mrna_dose = st.number_input("Target mRNA Dose (µg):", 5.0, 1000.0, 50.0, 5.0)
        target_lnp_size = st.slider("Target LNP Diameter (nm):", 60, 120, 80, 5)
        batch_vol_lnp = st.selectbox("Total Formulation Volume (mL):", [2.0, 5.0, 10.0, 25.0], index=1)

        calc_lnp = st.button("🚀 DESIGN LNP FORMULATION", type="primary", use_container_width=True)

    with lnp_col2:
        if calc_lnp:
            lnp_res = lnp_optimizer.optimize_lnp(
                ionizable_lipid_name=ionizable_key,
                helper_lipid_name=helper_key,
                peg_lipid_name=peg_key,
                target_np_ratio=target_np,
                target_size_nm=target_lnp_size,
                mrna_dose_ug=mrna_dose,
                batch_volume_ml=batch_vol_lnp
            )
            st.session_state['lnp_results'] = lnp_res

        if 'lnp_results' in st.session_state:
            lnp_data = st.session_state['lnp_results']
            cqas = lnp_data['predicted_cqas']
            mol_ratios = lnp_data['molar_ratios']
            masses = lnp_data['lipid_masses_mg']

            st.success("✅ mRNA Lipid Nanoparticle Design Complete!")

            l1, l2, l3, l4 = st.columns(4)
            l1.metric("Predicted Diameter", f"{cqas['pred_size_nm']} nm", f"PDI: {cqas['pred_pdi']}")
            l2.metric("mRNA Encapsulation", f"{cqas['pred_ee_percent']}%", f"N/P Ratio: {lnp_data['np_ratio']:.1f}")
            l3.metric("Total Lipid Mass", f"{masses['total_lipid_mass_mg']} mg", f"Lipid:mRNA = {lnp_data['lipid_to_mrna_mass_ratio']}:1")
            l4.metric("Zeta Potential", f"{cqas['zeta_potential_ph74_mv']} mV (pH 7.4)", f"+{cqas['zeta_potential_ph55_mv']} mV (pH 5.5 Endosomal)")

            st.markdown("#### 📊 4-Component Molar Composition")
            df_mols = pd.DataFrame({
                "Lipid Component": list(mol_ratios.keys()),
                "Molar Fraction (mol%)": list(mol_ratios.values()),
                "Required Mass (mg)": [masses.get(k, 0.0) for k in mol_ratios.keys()]
            })
            st.dataframe(df_mols, use_container_width=True)

            fig_donut = px.pie(
                df_mols, values='Molar Fraction (mol%)', names='Lipid Component',
                title="Lipid Nanoparticle Molar Stoichiometry", hole=0.45,
                color_discrete_sequence=['#1B4965', '#2E86AB', '#27AE60', '#E74C3C']
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            with st.expander("📋 Microfluidic Manufacturing SOP & Recipe", expanded=True):
                st.markdown(f"**Organic Phase (Ethanol):** {lnp_data['microfluidics_recipe']['ethanol_phase_vol_ml']} mL | **Aqueous Phase (Citrate pH 4.0):** {lnp_data['microfluidics_recipe']['aqueous_phase_vol_ml']} mL")
                st.markdown(f"**Flow Rate Ratio (FRR):** {lnp_data['microfluidics_recipe']['flow_rate_ratio_org_aq']}")
                for s in lnp_data['sop_steps']:
                    st.markdown(f"- {s}")
        else:
            st.info("👈 Set your mRNA dose, N/P ratio, and lipid choices and click **'DESIGN LNP FORMULATION'**.")


# ==============================================================================
# TAB 4: 4D DRUG RELEASE KINETICS SIMULATOR
# ==============================================================================
with tab_kinetics:
    st.markdown("### 📈 4D Temporal Drug Release Kinetics Simulator")
    st.markdown("Simulate multi-day in vitro sustained drug release curves ($0.5\\text{h} \\rightarrow 168\\text{h}$) and fit Korsmeyer-Peppas, Higuchi, and First-Order biopharmaceutical equations.")

    k_col1, k_col2 = st.columns([1, 2.2], gap="large")

    with k_col1:
        st.markdown("#### 🧪 Formulation Parameters")
        k_pmw = st.slider("PLGA MW (kDa):", 5.0, 120.0, 30.0, 5.0)
        k_laga = st.selectbox("LA/GA Ratio:", [1.0, 1.5, 2.33, 3.0, 5.67], format_func=lambda x: "50:50" if x==1.0 else ("65:35" if x==1.5 else ("75:25" if x==3.0 else "85:15")))
        k_logp = st.number_input("Drug Lipophilicity (LogP):", -2.0, 8.0, 3.2, 0.2)
        k_size = st.number_input("Particle Diameter (nm):", 50.0, 350.0, 150.0, 10.0)
        k_ee = st.slider("Entrapment Efficiency (% EE):", 30.0, 99.0, 80.0, 1.0)
        k_ph = st.radio("Release Medium pH:", [7.4, 5.0], format_func=lambda x: "pH 7.4 (Physiological / Blood)" if x==7.4 else "pH 5.0 (Endosomal / Tumor Acidic)")

        calc_kinetics = st.button("🚀 SIMULATE 4D RELEASE CURVE", type="primary", use_container_width=True)

    with k_col2:
        if calc_kinetics:
            sim_formulation = {
                "polymer_MW": k_pmw, "LA/GA": k_laga, "mol_logP": k_logp,
                "mol_MW": 380.0, "pred_size": k_size, "pred_EE": k_ee,
                "surfactant_concentration": 1.0
            }
            k_res = release_predictor.predict_release_curve(sim_formulation, release_ph=k_ph)
            st.session_state['kinetics_results'] = k_res

        if 'kinetics_results' in st.session_state:
            k_data = st.session_state['kinetics_results']
            fits = k_data['kinetic_fits']
            kp_info = fits.get('Korsmeyer-Peppas', {})

            st.success(f"✅ Kinetics Simulation Complete (Release Medium: pH {k_data['release_ph']})")

            km1, km2, km3, km4 = st.columns(4)
            km1.metric("Burst Release (2h)", f"{k_data['burst_2h_percent']}%", k_data['burst_risk_level'])
            km2.metric("24h Sustained Release", f"{k_data['release_24h_percent']}%")
            km3.metric("72h Cumulative Release", f"{k_data['release_72h_percent']}%")
            km4.metric("Peppas Exponent (n)", f"{kp_info.get('release_exponent_n', 0.45)}", f"R² = {kp_info.get('R2', 0.0)}")

            # Plot Cumulative Release Curve
            df_k = k_data['profile_df']
            fig_kinetics = px.line(
                df_k, x='Time (hours)', y='Cumulative Release (%)',
                markers=True, title="In Vitro Sustained Drug Release Profile",
                labels={'Time (hours)': 'Time (h)', 'Cumulative Release (%)': 'Cumulative Drug Release (%)'}
            )
            fig_kinetics.add_hline(y=k_data['burst_2h_percent'], line_dash="dash", line_color="orange", annotation_text=f"2h Burst ({k_data['burst_2h_percent']}%)")
            fig_kinetics.update_traces(line=dict(color='#1B4965', width=3), marker=dict(size=8, color='#E74C3C'))
            st.plotly_chart(fig_kinetics, use_container_width=True)

            # Mathematical Models Fit Table
            st.markdown("#### 📊 Mathematical Kinetic Model Fits")
            models_table = [
                {"Model": "Korsmeyer-Peppas", "Formula": "Q(t) = k * t^n", "R² Score": fits['Korsmeyer-Peppas'].get('R2'), "Rate Constant": f"k = {fits['Korsmeyer-Peppas'].get('k_kp')}", "Exponent / Parameter": f"n = {fits['Korsmeyer-Peppas'].get('release_exponent_n')}"},
                {"Model": "Higuchi", "Formula": "Q(t) = kh * t^0.5", "R² Score": fits['Higuchi'].get('R2'), "Rate Constant": f"kh = {fits['Higuchi'].get('rate_constant_kh')} %/h^0.5", "Exponent / Parameter": "n = 0.5 (Fixed)"},
                {"Model": "First-Order", "Formula": "Q(t) = 100*(1-exp(-k1*t))", "R² Score": fits['First-Order'].get('R2'), "Rate Constant": f"k1 = {fits['First-Order'].get('rate_constant_k1')} h^-1", "Exponent / Parameter": "Concentration-dependent"},
                {"Model": "Zero-Order", "Formula": "Q(t) = k0 * t", "R² Score": fits['Zero-Order'].get('R2'), "Rate Constant": f"k0 = {fits['Zero-Order'].get('rate_constant_k0')} %/h", "Exponent / Parameter": "Constant rate"}
            ]
            st.dataframe(pd.DataFrame(models_table), use_container_width=True)

            st.info(f"**Identified Mechanism:** **{kp_info.get('mechanism')}**  \n*{kp_info.get('description')}*")
        else:
            st.info("👈 Set your formulation parameters and click **'SIMULATE 4D RELEASE CURVE'**.")


# ==============================================================================
# TAB 5: PEG-PLGA & PCL POLYMERS
# ==============================================================================
with tab_stealth:
    st.markdown("### 🛡️ PEG-PLGA Stealth Nanoparticles & Polycaprolactone (PCL)")
    
    subtab1, subtab2 = st.tabs(["🛡️ PEG-PLGA Stealth Shielding", "⏳ PCL Extended Multi-Month Depot"])

    with subtab1:
        st.markdown("#### 🛡️ PEG-PLGA Stealth Conformation & Macrophage Evasion")
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            core_d = st.slider("PLGA Core Diameter (nm):", 60.0, 200.0, 110.0, 5.0)
            peg_mw = st.selectbox("PEG Chain Molecular Weight:", [2.0, 3.4, 5.0, 10.0], index=2, format_func=lambda x: f"{x} kDa")
            peg_wt = st.slider("PEG Weight Fraction (% w/w):", 2.0, 20.0, 10.0, 1.0)
        
        peg_eval = peg_plga_model.evaluate_stealth_properties(core_diameter_nm=core_d, peg_mw_kda=peg_mw, peg_weight_fraction_percent=peg_wt)
        
        with p_c2:
            st.markdown("##### Stealth Performance Metrics")
            st.metric("Conformation Regime", peg_eval["conformation_regime"], f"Flory Radius: {peg_eval['flory_radius_rf_nm']} nm")
            st.metric("Macrophage Opsonin Reduction", f"{peg_eval['opsonization_reduction_percent']}%", f"Density: {peg_eval['peg_grafting_density_chains_nm2']} chains/nm²")
            st.metric("Blood Circulation Half-Life", peg_eval["circulation_half_life_multiplier"])
            st.metric("Effective Hydrodynamic Size", f"{peg_eval['effective_hydrodynamic_diameter_nm']} nm")

    with subtab2:
        st.markdown("#### ⏳ Polycaprolactone (PCL) Long-Term Sustained Delivery")
        pcl_c1, pcl_c2 = st.columns(2)
        with pcl_c1:
            pcl_mw = st.slider("PCL Molecular Weight (kDa):", 14.0, 80.0, 45.0, 5.0)
            pcl_logp = st.number_input("Drug LogP:", -1.0, 8.0, 3.5, 0.2, key="pcl_logp")
            pcl_mw_d = st.number_input("Drug MW (g/mol):", 100.0, 1000.0, 380.0, 10.0, key="pcl_mwd")
        
        pcl_eval = pcl_model.evaluate_pcl_formulation(pcl_mw_kda=pcl_mw, drug_logp=pcl_logp, drug_mw=pcl_mw_d)
        
        with pcl_c2:
            st.markdown("##### PCL Depot Performance")
            st.metric("Polymer Matrix", pcl_eval["polymer"], f"Degradation: {pcl_eval['degradation_half_life']}")
            st.metric("Predicted Encapsulation Efficiency", f"{pcl_eval['predicted_ee_percent']}%", f"Core Compatibility: {pcl_eval['drug_compatibility_score']}")
            st.markdown(f"**Multi-Month Release Milestones:**  \n- **30 Days:** {pcl_eval['release_milestones']['30_days']}  \n- **90 Days:** {pcl_eval['release_milestones']['90_days']}  \n- **180 Days:** {pcl_eval['release_milestones']['180_days']}")


# ==============================================================================
# TAB 6: BATCH VIRTUAL SCREENING
# ==============================================================================
with tab_screening:
    st.markdown("### ⚡ High-Throughput Batch Virtual Screening")
    st.markdown("Screen compound libraries to evaluate formulation feasibility, particle size, and encapsulation efficiency across multiple candidates.")

    if st.button("🚀 RUN HIGH-THROUGHPUT SCREENING ON CURATED PANEL", type="primary"):
        drug_names = nf.get_drug_names()
        results_list = []
        progress_bar = st.progress(0.0)

        for idx, name in enumerate(drug_names):
            d_info = nf.get_drug_by_name(name)
            d_props = {k: d_info[k] for k in ['mol_MW', 'mol_logP', 'mol_TPSA', 'mol_melting_point', 'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms']}
            opt_res = plga_optimizer.optimize(d_props, target_size=160, min_ee=70, n_recommendations=1, n_candidates=5000)
            best_cand = opt_res['recommendations'].iloc[0]
            
            results_list.append({
                "Drug": name, "Class": d_info.get('therapeutic_class', ''),
                "MW (g/mol)": d_info['mol_MW'], "LogP": d_info['mol_logP'],
                "Opt. PLGA MW (kDa)": best_cand['polymer_MW'], "Opt. D/P Ratio": best_cand['drug/polymer'],
                "Pred. Size (nm)": round(best_cand['pred_size'], 1), "Pred. EE (%)": round(best_cand['pred_EE'], 1),
                "Pred. LC (%)": round(best_cand['pred_LC'], 1), "AD Status": best_cand['ad_status'].split(' ')[0]
            })
            progress_bar.progress((idx + 1) / len(drug_names))

        df_screen = pd.DataFrame(results_list)
        st.session_state['screen_df'] = df_screen

    if 'screen_df' in st.session_state:
        df_screen = st.session_state['screen_df']
        st.markdown("#### 📊 Screening Results (Ranked by Predicted EE%)")
        st.dataframe(df_screen.sort_values("Pred. EE (%)", ascending=False), use_container_width=True)

        fig_screen = px.scatter(
            df_screen, x='LogP', y='Pred. EE (%)', size='Pred. LC (%)', color='Pred. Size (nm)',
            text='Drug', title="Drug Lipophilicity (LogP) vs. Predicted Encapsulation Efficiency",
            labels={'LogP': 'Drug LogP', 'Pred. EE (%)': 'Predicted EE (%)'}, color_continuous_scale='Viridis'
        )
        fig_screen.update_traces(textposition='top center')
        st.plotly_chart(fig_screen, use_container_width=True)

        csv_screen = df_screen.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Virtual Screening CSV", csv_screen, "nanoformula_screening_results.csv", "text/csv")


# ==============================================================================
# TAB 7: LAB-IN-THE-LOOP ACTIVE LEARNING
# ==============================================================================
with tab_active:
    st.markdown("### 🔄 Lab-in-the-Loop Active Learning & Experimental Feedback")
    st.markdown("Contribute wet-lab formulation batch results to the active learning repository to improve model precision.")

    al_col1, al_col2 = st.columns([1.2, 1.8], gap="large")

    with al_col1:
        st.markdown("#### 📝 Submit Experimental Batch")
        with st.form("feedback_form"):
            res_name = st.text_input("Researcher Name:", "Hardik Sood")
            inst_name = st.text_input("Institution / Lab:", "IIT (BHU) Varanasi")
            poly_sys = st.selectbox("Polymer System:", ["PLGA", "Chitosan-TPP", "mRNA Lipid Nanoparticles (LNP)", "PEG-PLGA", "PCL"])
            d_name = st.text_input("Drug / API Tested:", "Curcumin")
            
            st.markdown("**Formulation Parameters:**")
            fb_mw = st.number_input("Polymer MW (kDa):", 1.0, 300.0, 24.0, 1.0)
            fb_dp = st.number_input("Drug/Polymer Ratio:", 0.001, 1.0, 0.10, 0.01)
            fb_surf = st.number_input("Surfactant Conc (% w/v):", 0.0, 5.0, 1.0, 0.1)
            
            st.markdown("**Experimental Measured Outcomes:**")
            meas_size = st.number_input("Measured Particle Size (nm):", 10.0, 1000.0, 158.4, 0.5)
            meas_ee = st.number_input("Measured Entrapment Efficiency (% EE):", 0.0, 100.0, 82.5, 0.5)
            meas_pdi = st.number_input("Measured PDI:", 0.01, 1.0, 0.18, 0.01)
            meas_zeta = st.number_input("Measured Zeta Potential (mV):", -60.0, 60.0, -18.2, 0.5)
            inst_notes = st.text_input("Instrument / Method:", "Malvern Zetasizer Nano ZS / RP-HPLC")

            submit_fb = st.form_submit_button("📤 SUBMIT TO ACTIVE LEARNING REPOSITORY", type="primary")

            if submit_fb:
                fb_res = active_learning_engine.submit_experimental_batch(
                    researcher_name=res_name, institution=inst_name, polymer_system=poly_sys,
                    drug_name=d_name, formulation_parameters={"polymer_MW": fb_mw, "drug/polymer": fb_dp, "surfactant_concentration": fb_surf},
                    measured_particle_size_nm=meas_size, measured_ee_percent=meas_ee,
                    measured_pdi=meas_pdi, measured_zeta_potential_mv=meas_zeta, instrument_notes=inst_notes
                )
                st.success(fb_res["message"])

    with al_col2:
        st.markdown("#### 📊 Active Learning Provenance & Database")
        al_stats = active_learning_engine.compute_active_learning_metrics()
        
        a1, a2 = st.columns(2)
        a1.metric("Batches Registered", al_stats["total_batches_contributed"])
        a2.metric("Unique Drugs Tested", al_stats["unique_drugs_tested"])

        entries = active_learning_engine.get_all_entries()
        if entries:
            table_entries = []
            for e in entries:
                table_entries.append({
                    "Batch ID": e.get("entry_id"),
                    "Date": e.get("timestamp")[:10],
                    "Researcher": e.get("researcher"),
                    "Polymer": e.get("polymer_system"),
                    "Drug": e.get("drug_name"),
                    "Size (nm)": e.get("experimental_results", {}).get("particle_size_nm"),
                    "EE (%)": e.get("experimental_results", {}).get("ee_percent"),
                    "PDI": e.get("experimental_results", {}).get("pdi")
                })
            st.dataframe(pd.DataFrame(table_entries), use_container_width=True)
        else:
            st.info("No user batches submitted yet. Be the first to register an experimental batch!")


# ==============================================================================
# TAB 8: BENCHMARKS & LITERATURE VALIDATION
# ==============================================================================
with tab_benchmarks:
    st.markdown("### 🔬 Scientific Rigor & Benchmark Transparency")
    
    b_tab1, b_tab2, b_tab3 = st.tabs(["📚 Literature External Validation", "📊 10-Fold CV Benchmarks", "📈 300 DPI Parity Plots"])

    with b_tab1:
        st.markdown("#### 📚 Independent Literature Validation (Published Peer-Reviewed Papers)")
        st.markdown("Predictions compared against independent experimental studies from peer-reviewed literature.")
        val_df = lit_validator.run_literature_validation()
        st.dataframe(val_df, use_container_width=True)

    with b_tab2:
        st.markdown("#### 📊 10-Fold Cross-Validation Model Comparisons")
        b1, b2 = st.columns(2)
        with b1:
            st.markdown("##### PLGA Models Suite")
            bm_plga_path = os.path.join("benchmarks", "plga_algorithm_benchmarks.csv")
            if os.path.exists(bm_plga_path):
                st.dataframe(pd.read_csv(bm_plga_path), use_container_width=True)
        with b2:
            st.markdown("##### Chitosan Models Suite")
            bm_cs_path = os.path.join("benchmarks", "chitosan_algorithm_benchmarks.csv")
            if os.path.exists(bm_cs_path):
                st.dataframe(pd.read_csv(bm_cs_path), use_container_width=True)

    with b_tab3:
        st.markdown("#### 📈 Publication Parity Plots (300 DPI Actual vs. Predicted)")
        p1, p2 = st.columns(2)
        with p1:
            if os.path.exists("benchmarks/plga_parity_plots.png"):
                st.image("benchmarks/plga_parity_plots.png", caption="Figure 1: PLGA Particle Size, EE%, and Loading Capacity Parity Plots")
        with p2:
            if os.path.exists("benchmarks/chitosan_parity_plots.png"):
                st.image("benchmarks/chitosan_parity_plots.png", caption="Figure 2: Chitosan Particle Size, PDI, and Zeta Potential Parity Plots")


# ==============================================================================
# TAB 9: RESEARCH & CITATION
# ==============================================================================
with tab_about:
    st.markdown("### 📄 Research Methodology & Citations")
    st.markdown("""
    #### 💡 Theoretical Framework
    NanoFormula AI 2.0 addresses formulation bottlenecks across nanomedicine platforms:
    1. **Chemoinformatics**: Automated RDKit descriptor calculation and PubChem API integration.
    2. **Multi-Model Ensemble Learning**: Blending XGBoost, Random Forest, Gradient Boosting, and Extra Trees.
    3. **Uncertainty Quantification (UQ)**: Predicting mean and 95% confidence intervals to guide lab trials.
    4. **Applicability Domain (AD)**: Leverage calculation (William's Plot) and Mahalanobis distance to prevent extrapolation errors.
    5. **Pareto Multi-Objective Optimization**: Derringer-Suich desirability functions for simultaneous targeting of size, EE%, and loading capacity.
    6. **4D Drug Release Kinetics**: Modeling multi-phase surface burst, Stokes-Einstein diffusion, and Korsmeyer-Peppas mechanism classification.
    7. **mRNA Lipid Nanoparticles (LNPs)**: 4-Component stoichiometric optimization and N/P ratio calculations.
    8. **Automated Wet-Lab SOP Synthesis**: Translating numerical vectors into actionable laboratory recipes and printable PDF certificates.

    ---
    #### 👨‍🔬 Research Team & Lab Affiliation
    - **Developer:** Hardik Sood (B.Tech Pharmaceutical Engineering, IIT (BHU) Varanasi)
    - **Supervisor:** Dr. Ruchi Chawla (Associate Professor, Department of Pharmaceutical Engineering & Technology, IIT (BHU) Varanasi)
    - **Institution:** Indian Institute of Technology (BHU), Varanasi, India

    ---
    #### 📖 Suggested Citation
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
    """)
