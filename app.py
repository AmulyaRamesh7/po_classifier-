import json
import streamlit as st
from classifier import classify_po

st.set_page_config(page_title="PO Category Classifier", layout="centered")

st.title("ðŸ“¦ PO L1â€“L2â€“L3 Classifier")

if "last_inputs" not in st.session_state:
    st.session_state["last_inputs"] = None
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False


@st.cache_data(show_spinner=False)
def classify_cached(po_description: str, supplier: str):
    return classify_po(po_description, supplier)


with st.form("po_form"):
    po_description = st.text_area("PO Description", height=120)
    supplier = st.text_input("Supplier (optional)")
    submitted = st.form_submit_button("Classify", disabled=st.session_state["is_processing"])

if submitted:
    if not po_description.strip():
        st.warning("Please enter a PO description.")
    else:
        supplier_normalized = supplier.strip() or "Not provided"
        current_inputs = (po_description, supplier_normalized)

        if st.session_state["last_inputs"] == current_inputs and st.session_state["last_result"]:
            result = st.session_state["last_result"]
            st.info("Reused the most recent result for these inputs.")
        else:
            st.session_state["is_processing"] = True
            with st.spinner("Classifying..."):
                result = classify_cached(po_description, supplier_normalized)
            st.session_state["is_processing"] = False
            st.session_state["last_inputs"] = current_inputs
            st.session_state["last_result"] = result

        try:
            st.json(json.loads(result))
        except Exception:
            st.error("Invalid model response")
            st.text(result)
