# [1] Imports and Model Setup
import streamlit as st
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data
from model import GNNClassifier
from utils import (
    smiles_to_graph,
    drug_name_to_smiles,
    get_available_drug_names,
    get_clinical_insights
)

# Load model
model = GNNClassifier()
model.load_state_dict(torch.load("ddi_model.pt", map_location=torch.device('cpu')))
model.eval()

# Streamlit UI Config
st.set_page_config(page_title="DDI Predictor", layout="wide")
st.title("💊 Drug–Drug Interaction Predictor")

# Sidebar drug names
with st.sidebar:
    st.header("📚 Available Drugs")
    for name in sorted(get_available_drug_names()):
        st.markdown(f"- {name.title()}")

# Drug Inputs
drug_names = sorted(get_available_drug_names())
drug1_name = st.selectbox("Select Drug 1", options=drug_names, index=0)
drug2_name = st.selectbox("Select Drug 2", options=drug_names, index=1)

# Predict
if st.button("🔍 Predict Interaction"):
    smiles1 = drug_name_to_smiles(drug1_name)
    smiles2 = drug_name_to_smiles(drug2_name)

    if not smiles1 or not smiles2:
        st.error("❌ Could not fetch SMILES for one or both drug names.")
    else:
        try:
            st.markdown(f"**Drug 1 SMILES**: `{smiles1}`")
            st.markdown(f"**Drug 2 SMILES**: `{smiles2}`")
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            st.image(Draw.MolsToImage([mol1, mol2], subImgSize=(250, 250)))

            # Convert to graph
            g1 = smiles_to_graph(smiles1)
            g2 = smiles_to_graph(smiles2)
            x = torch.cat([g1.x, g2.x], dim=0)
            edge_index = torch.cat([g1.edge_index, g2.edge_index + g1.x.size(0)], dim=1)
            data = Data(x=x, edge_index=edge_index)
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)

            # Run prediction
            with torch.no_grad():
                confidence = model(data).item()

            evidence = "Strong" if confidence > 0.9 else "Moderate" if confidence > 0.6 else "Weak"

            # ================================
            # ✅ PUT YOUR CLINICAL LOGIC HERE
            # ================================
            insight = get_clinical_insights(drug1_name.lower(), drug2_name.lower())

            if confidence > 0.75:
                default_risk = "HIGH"
                default_type = "Likely Pharmacodynamic Interaction"
                default_mechanism = "May lead to synergistic effects or overlapping toxicity."
                default_significance = "Combination may increase risk of adverse events."
                default_recs = [
                    "Avoid combination if safer alternatives exist",
                    "Carefully monitor side effects or reactions",
                    "Dose adjustment may be required"
                ]
            elif confidence > 0.5:
                default_risk = "MEDIUM"
                default_type = "Possible Pharmacokinetic Interaction"
                default_mechanism = "Potential overlap in metabolism or binding pathways."
                default_significance = "May increase blood levels of one or both drugs."
                default_recs = [
                    "Consider dose monitoring",
                    "Look out for exaggerated or delayed drug effects",
                    "No action needed if short-term"
                ]
            else:
                default_risk = "LOW"
                default_type = "Unlikely Interaction"
                default_mechanism = "No significant overlap in mechanism detected."
                default_significance = "Interaction unlikely to be clinically relevant."
                default_recs = [
                    "Standard dosing is acceptable",
                    "Minimal monitoring required",
                    "No special action needed"
                ]

            if insight:
                interaction_type = insight['type']
                risk = insight['risk']
                mechanism = insight['mechanism']
                significance = insight['significance']
                recommendations = insight['recommendations']
                alert = f"⚠️ {risk.title()} risk interaction! Follow clinical precautions."
            else:
                interaction_type = default_type
                risk = default_risk
                mechanism = default_mechanism
                significance = default_significance
                recommendations = default_recs
                alert = f"⚠️ {risk.title()} risk — based on model prediction confidence."

            # ================================
            # ✅ RENDER DASHBOARD
            # ================================
            color = "red" if risk == "HIGH" else "orange" if risk == "MEDIUM" else "green" if risk == "LOW" else "#6c757d"

            st.markdown("### 📈 Interaction Prediction Results")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Interaction Type**\n\n{interaction_type}")
            with col2:
                st.markdown("**Risk Level**")
                st.markdown(f"<span style='color:white;background-color:{color};padding:4px 8px;border-radius:6px'>{risk}</span>", unsafe_allow_html=True)

            st.markdown("**Prediction Confidence**")
            st.progress(confidence)
            st.caption(f"{int(confidence * 100)}% confident")

            st.markdown("**Evidence Strength**")
            st.caption(evidence + " evidence")

            st.markdown("### 🏥 Clinical Information")
            st.markdown("**Mechanism of Interaction**")
            st.write(mechanism)

            st.markdown("**Clinical Significance**")
            st.write(significance)

            st.markdown("**Recommendations**")
            for rec in recommendations:
                st.markdown(f"- {rec}")

            if alert:
                st.warning(alert)

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
