"""
NanoFormula AI: Multi-Polymer Nanoparticle Formulation Optimizer powered by Machine Learning
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
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Chemoinformatics & Core Modules
from nanoformula.chemoinformatics import (
    calculate_descriptors_from_smiles,
    fetch_drug_from_pubchem,
    generate_structure_svg,
    get_drug_names,
    get_drug_by_name,
    RDKIT_AVAILABLE
)
from nanoformula.optimization import PLGAFormulationOptimizer, ChitosanFormulationOptimizer
from nanoformula.ml import (
    PLGA_FEATURES,
    CHITOSAN_FEATURES,
    FEATURE_DISPLAY_NAMES,
    ModelExplainer
)
from nanoformula.protocols import (
    generate_plga_lab_sop,
    generate_chitosan_lab_sop,
    generate_formulation_pdf_report
)

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
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
        font-size: 30px;
        font-weight: 800;
        color: #1B4965;
        text-align: center;
        padding-top: 10px;
        margin-bottom: 2px;
    }
    .sub-header {
        font-size: 15px;
        color: #555;
        text-align: center;
        margin-bottom: 20px;
        font-weight: 500;
    }
    .metric-card {
        background-color: #F8F9FA;
        border: 1px solid #E9ECEF;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
        padding: 8px 18px;
    }
    .status-badge-high {
        background-color: #D4EDDA;
        color: #155724;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .status-badge-mod {
        background-color: #FFF3CD;
        color: #856404;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
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
plga_optimizer = PLGAFormulationOptimizer(bundle)
chitosan_optimizer = ChitosanFormulationOptimizer(bundle)

# ==============================================================================
# 3. APPLICATION HEADER
# ==============================================================================
st.markdown('<div class="main-header">🧬 NanoFormula AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Polymer Nanoparticle Formulation Optimizer & Virtual Screening Engine<br/><b>Dr. Ruchi Chawla\'s Lab</b> | Department of Pharmaceutical Engineering & Technology, IIT (BHU) Varanasi</div>', unsafe_allow_html=True)

# Main Navigation Tabs
tab_plga, tab_chitosan, tab_screening, tab_benchmarks, tab_about = st.tabs([
    "🎯 PLGA Nanoparticle Optimizer",
    "🧪 Chitosan-TPP Nanoparticles",
    "⚡ Batch Virtual Screening",
    "🔬 Benchmarks & Transparency",
    "📄 Research & Citation"
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
            drug_list = get_drug_names()
            selected_name = st.selectbox("Select Drug Compound:", drug_list, index=drug_list.index("Curcumin") if "Curcumin" in drug_list else 0)
            drug_info = get_drug_by_name(selected_name)
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
            query_name = st.text_input("Enter Drug / Chemical Name:", "Doxorubicin")
            if query_name:
                with st.spinner("Searching PubChem & calculating RDKit descriptors..."):
                    fetched = fetch_drug_from_pubchem(query_name)
                    if fetched:
                        drug_name = fetched.get('name', query_name)
                        drug_smiles = fetched.get('smiles')
                        drug_props = {k: fetched[k] for k in ['mol_MW', 'mol_logP', 'mol_TPSA', 'mol_melting_point', 'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms']}
                        st.success(f"Found **{drug_name}** in PubChem!")
                    else:
                        st.error(f"Could not retrieve '{query_name}' from PubChem. Try entering SMILES directly.")

        elif input_mode == "🧪 SMILES Structure":
            smiles_input = st.text_area("Paste SMILES string:", "COC1=C(C=CC(=C1)/C=C/C(=O)CC(=O)/C=C/C2=CC(=C(C=C2)O)OC)O", height=70)
            drug_custom_name = st.text_input("Compound Name / Identifier:", "Curcumin Analog")
            if smiles_input:
                try:
                    calc_res = calculate_descriptors_from_smiles(smiles_input)
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

        # 2D Structure Rendering
        if drug_smiles and RDKIT_AVAILABLE:
            st.markdown("#### 🔬 2D Chemical Structure")
            svg_data = generate_structure_svg(drug_smiles, width=320, height=180)
            if svg_data:
                st.image(svg_data, use_container_width=True)

        st.markdown("---")
        st.markdown("### ⚙️ 2. Formulation Objectives")
        target_size = st.slider("Target Hydrodynamic Size (nm)", 50, 300, 160, 10, help="Tumor EPR: 100-160 nm | Oral/Mucosal: 150-250 nm")
        min_ee = st.slider("Minimum Entrapment Efficiency (% EE)", 40, 95, 70, 5)
        
        polymer_choice = st.selectbox(
            "PLGA Molecular Weight Constraints:",
            ["Auto-Explore Full Space", "Constrain to Low MW (5-15 kDa)", "Constrain to Med MW (24-50 kDa)", "Constrain to High MW (65-110 kDa)"]
        )
        p_mw_constraint = None
        if "Low" in polymer_choice:
            p_mw_constraint = 10.0
        elif "Med" in polymer_choice:
            p_mw_constraint = 30.0
        elif "High" in polymer_choice:
            p_mw_constraint = 80.0

        n_recs = st.slider("Number of Diverse Candidates", 3, 8, 5)
        run_opt = st.button("🚀 OPTIMIZE FORMULATION", type="primary", use_container_width=True)

    with col_results:
        if run_opt and drug_props:
            with st.spinner("Running Monte Carlo Candidate Generation & ML Ensemble Inference..."):
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
            
            display_df = recs[['drug/polymer', 'polymer_MW', 'LA/GA',
                               'surfactant_concentration', 'aqueous/organic',
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
            st.markdown("### 📊 Interactive Formulation Visualizer")
            t_chart1, t_chart2, t_shap = st.tabs(["📈 Pareto Frontier Trade-Off", "📊 Size & EE Distribution", "🧠 SHAP Explainability"])

            with t_chart1:
                cand_df = results['all_candidates']
                fig_pareto = px.scatter(
                    cand_df.sample(min(3000, len(cand_df)), random_state=42),
                    x='pred_size',
                    y='pred_EE',
                    color='pred_LC',
                    size='desirability',
                    hover_data=['polymer_MW', 'drug/polymer', 'surfactant_concentration', 'ad_status'],
                    labels={
                        'pred_size': 'Predicted Particle Size (nm)',
                        'pred_EE': 'Predicted Entrapment Efficiency (EE%)',
                        'pred_LC': 'Loading Capacity (%)'
                    },
                    title="Formulation Space & Pareto Optimal Trade-off",
                    color_continuous_scale="Viridis"
                )
                fig_pareto.add_vline(x=target_size, line_dash="dash", line_color="red", annotation_text="Target Size")
                fig_pareto.add_hline(y=min_ee, line_dash="dash", line_color="blue", annotation_text="Min EE%")
                
                # Highlight top recommendations on scatter
                fig_pareto.add_trace(go.Scatter(
                    x=recs['pred_size'],
                    y=recs['pred_EE'],
                    mode='markers+text',
                    marker=dict(size=14, color='#E74C3C', symbol='star', line=dict(width=1, color='black')),
                    text=[f"#{i+1}" for i in range(len(recs))],
                    textposition="top center",
                    name="Top Recommendations"
                ))
                st.plotly_chart(fig_pareto, use_container_width=True)

            with t_chart2:
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=[f"Form #{i+1}" for i in range(len(recs))],
                    y=recs['pred_size'],
                    error_y=dict(type='data', array=recs['pred_size_std'], visible=True),
                    name="Particle Size (nm)",
                    marker_color='#2E86AB'
                ))
                fig_bar.add_hline(y=target_size, line_dash="dash", line_color="red", annotation_text=f"Target: {target_size} nm")
                fig_bar.update_layout(title="Predicted Hydrodynamic Diameter (± 95% Confidence Interval)", yaxis_title="Size (nm)")
                st.plotly_chart(fig_bar, use_container_width=True)

            with t_shap:
                st.markdown("#### 🧠 TreeSHAP Feature Attribution Breakdown")
                st.caption("Explains which physicochemical and formulation parameters drove the prediction for the top recommendation.")
                
                explainer = ModelExplainer(bundle['plga_size'], PLGA_FEATURES, "Particle Size (nm)")
                fig_shap = explainer.plot_instance_breakdown(recs.iloc[0].to_dict(), f"SHAP Feature Attribution (Formulation #1 Size)")
                if fig_shap:
                    st.pyplot(fig_shap)

            # Laboratory SOP & Batch Recipe
            st.markdown("---")
            st.markdown("### 🧪 Suggested Laboratory Recipe & SOP")
            batch_vol = st.selectbox("Select Target Batch Volume:", [5.0, 10.0, 25.0, 50.0], index=1)
            
            best_rec = recs.iloc[0].to_dict()
            sop = generate_plga_lab_sop(best_rec, batch_volume_ml=batch_vol, drug_name=curr_drug)
            
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
                pdf_bytes = generate_formulation_pdf_report(
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
            st.info("👈 Select or input your drug properties on the left sidebar and click **'OPTIMIZE FORMULATION'** to begin.")


# ==============================================================================
# TAB 2: CHITOSAN-TPP NANOPARTICLES
# ==============================================================================
with tab_chitosan:
    st.markdown("### 🧪 Chitosan-TPP Nanoparticle Optimizer")
    st.info("**Predicts:** Hydrodynamic Size (nm), Polydispersity Index (PDI), and Zeta Potential (+mV) using Ionic Gelation modeling.")

    c_left, c_right = st.columns([1, 2.2], gap="large")

    with c_left:
        st.markdown("#### ⚙️ Parameters & Constraints")
        cs_mw_mode = st.radio(
            "Chitosan Selection:",
            ["Auto-Explore All MW", "5 kDa", "20 kDa", "Low MW (~50 kDa)", "High MW (~310 kDa)"]
        )
        cs_mw_fixed = None
        if "5 kDa" in cs_mw_mode:
            cs_mw_fixed = 5.0
        elif "20 kDa" in cs_mw_mode:
            cs_mw_fixed = 20.0
        elif "50 kDa" in cs_mw_mode:
            cs_mw_fixed = 50.0
        elif "310 kDa" in cs_mw_mode:
            cs_mw_fixed = 310.0

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

            # Table
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
                recs_cs,
                x='pred_size',
                y='pred_zeta',
                size='chitosan_conc',
                color='chitosan_MW',
                labels={'pred_size': 'Predicted Particle Size (nm)', 'pred_zeta': 'Zeta Potential (+mV)'},
                title="Chitosan Size vs. Zeta Potential Profile",
                text=[f"#{i+1}" for i in range(len(recs_cs))]
            )
            st.plotly_chart(fig_cs_scatter, use_container_width=True)

            # Lab Protocol
            best_cs = recs_cs.iloc[0].to_dict()
            cs_sop = generate_chitosan_lab_sop(best_cs, batch_volume_ml=10.0)
            
            with st.expander("📋 Suggested Ionic Gelation Lab Protocol", expanded=True):
                for step in cs_sop['steps']:
                    st.markdown(f"- {step}")

            # Download PDF
            pdf_cs_bytes = generate_formulation_pdf_report(
                drug_name="Blank Chitosan-TPP Nanoparticles",
                drug_properties={'mol_MW': 'N/A', 'mol_logP': 'N/A', 'mol_TPSA': 'N/A', 'mol_melting_point': 'N/A', 'mol_Hacceptors': 'N/A', 'mol_Hdonors': 'N/A'},
                top_formulations=recs_cs.to_dict(orient='records'),
                lab_sop=cs_sop,
                polymer_system="Chitosan"
            )
            st.download_button(
                label="📄 Download Chitosan Laboratory PDF Certificate",
                data=pdf_cs_bytes,
                file_name="NanoFormula_Chitosan_TPP_Report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.info("👈 Set your target size, PDI, and Zeta constraints and click **'OPTIMIZE CHITOSAN'**.")


# ==============================================================================
# TAB 3: BATCH VIRTUAL SCREENING
# ==============================================================================
with tab_screening:
    st.markdown("### ⚡ High-Throughput Batch Virtual Screening")
    st.markdown("Screen compound libraries to identify formulation feasibility, predicted hydrodynamic diameter, and encapsulation yield across multiple therapeutic candidates.")

    screening_mode = st.radio("Screening Source:", ["🧬 Screen Curated Nanomedicine Library (17 APIs)", "📤 Upload Compound Library (CSV)"], horizontal=True)

    if screening_mode == "🧬 Screen Curated Nanomedicine Library (17 APIs)":
        if st.button("🚀 RUN HIGH-THROUGHPUT SCREENING", type="primary"):
            drug_names = get_drug_names()
            results_list = []
            progress_bar = st.progress(0.0)

            for idx, name in enumerate(drug_names):
                d_info = get_drug_by_name(name)
                d_props = {k: d_info[k] for k in ['mol_MW', 'mol_logP', 'mol_TPSA', 'mol_melting_point', 'mol_Hacceptors', 'mol_Hdonors', 'mol_heteroatoms']}
                
                opt_res = plga_optimizer.optimize(d_props, target_size=160, min_ee=70, n_recommendations=1, n_candidates=5000)
                best_cand = opt_res['recommendations'].iloc[0]
                
                results_list.append({
                    "Drug": name,
                    "Class": d_info.get('therapeutic_class', ''),
                    "MW (g/mol)": d_info['mol_MW'],
                    "LogP": d_info['mol_logP'],
                    "TPSA (Å²)": d_info['mol_TPSA'],
                    "Opt. PLGA MW (kDa)": best_cand['polymer_MW'],
                    "Opt. D/P Ratio": best_cand['drug/polymer'],
                    "Opt. Surf (%)": best_cand['surfactant_concentration'],
                    "Pred. Size (nm)": round(best_cand['pred_size'], 1),
                    "Pred. EE (%)": round(best_cand['pred_EE'], 1),
                    "Pred. LC (%)": round(best_cand['pred_LC'], 1),
                    "AD Status": best_cand['ad_status'].split(' ')[0]
                })
                progress_bar.progress((idx + 1) / len(drug_names))

            df_screen = pd.DataFrame(results_list)
            st.session_state['screen_df'] = df_screen

    if 'screen_df' in st.session_state:
        df_screen = st.session_state['screen_df']
        st.markdown("#### 📊 Screening Results (Ranked by Predicted EE%)")
        st.dataframe(df_screen.sort_values("Pred. EE (%)", ascending=False), use_container_width=True)

        fig_screen = px.scatter(
            df_screen,
            x='LogP',
            y='Pred. EE (%)',
            size='Pred. LC (%)',
            color='Pred. Size (nm)',
            text='Drug',
            title="Drug Lipophilicity (LogP) vs. Predicted Encapsulation Efficiency",
            labels={'LogP': 'Drug LogP', 'Pred. EE (%)': 'Predicted EE (%)'},
            color_continuous_scale='Viridis'
        )
        fig_screen.update_traces(textposition='top center')
        st.plotly_chart(fig_screen, use_container_width=True)

        csv_screen = df_screen.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Full Virtual Screening Results (CSV)", csv_screen, "nanoformula_screening_results.csv", "text/csv")


# ==============================================================================
# TAB 4: BENCHMARKS & SCIENTIFIC TRANSPARENCY
# ==============================================================================
with tab_benchmarks:
    st.markdown("### 🔬 Scientific Rigor & Benchmark Transparency")
    st.markdown("To ensure full reproducibility for academic review, NanoFormula AI models were evaluated using **10-Fold Repeated Cross-Validation** across 8 distinct machine learning algorithms.")

    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("#### 📊 PLGA Model Suite (10-Fold CV Benchmarks)")
        bm_plga_path = os.path.join("benchmarks", "plga_algorithm_benchmarks.csv")
        if os.path.exists(bm_plga_path):
            st.dataframe(pd.read_csv(bm_plga_path), use_container_width=True)
            
    with b_col2:
        st.markdown("#### 📊 Chitosan Model Suite (10-Fold CV Benchmarks)")
        bm_cs_path = os.path.join("benchmarks", "chitosan_algorithm_benchmarks.csv")
        if os.path.exists(bm_cs_path):
            st.dataframe(pd.read_csv(bm_cs_path), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 📈 Publication Parity Plots (300 DPI Actual vs. Predicted)")
    
    p1, p2 = st.columns(2)
    with p1:
        if os.path.exists("benchmarks/plga_parity_plots.png"):
            st.image("benchmarks/plga_parity_plots.png", caption="Figure 1: PLGA Particle Size, EE%, and Loading Capacity Parity Plots")
    with p2:
        if os.path.exists("benchmarks/chitosan_parity_plots.png"):
            st.image("benchmarks/chitosan_parity_plots.png", caption="Figure 2: Chitosan Particle Size, PDI, and Zeta Potential Parity Plots")


# ==============================================================================
# TAB 5: RESEARCH & CITATION
# ==============================================================================
with tab_about:
    st.markdown("### 📄 Research Methodology & Citations")
    st.markdown("""
    #### 💡 Theoretical Framework
    NanoFormula AI addresses the formulation bottleneck in nanomedicine by integrating:
    1. **Chemoinformatics**: Automated RDKit descriptor calculation and PubChem API integration.
    2. **Multi-Model Ensemble Learning**: Blending XGBoost, Random Forest, Gradient Boosting, and Extra Trees.
    3. **Uncertainty Quantification (UQ)**: Predicting mean and 95% confidence intervals to guide lab trials.
    4. **Applicability Domain (AD)**: Leverage calculation (William's Plot) and Mahalanobis distance to prevent extrapolation errors.
    5. **Pareto Multi-Objective Optimization**: Derringer-Suich desirability functions for simultaneous targeting of size, EE%, and loading capacity.
    6. **Automated Wet-Lab SOP Synthesis**: Translating numerical vectors into actionable laboratory recipes and printable PDF certificates.

    ---
    #### 👨‍🔬 Research Team & Lab Affiliation
    - **Developer:** Hardik Sood (B.Tech Pharmaceutical Engineering, IIT (BHU) Varanasi)
    - **Supervisor:** Dr. Ruchi Chawla (Associate Professor, Department of Pharmaceutical Engineering & Technology, IIT (BHU) Varanasi)
    - **Institution:** Indian Institute of Technology (BHU), Varanasi, India

    ---
    #### 📖 Suggested Citation
    ```bibtex
    @article{sood2026nanoformula,
      title={NanoFormula AI: Machine Learning-Driven Multi-Objective Nanoparticle Formulation Optimization and Virtual Screening},
      author={Sood, Hardik and Chawla, Ruchi},
      journal={Journal of Controlled Release},
      year={2026},
      publisher={Elsevier}
    }
    ```
    """)
