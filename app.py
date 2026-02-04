import json
import time

import streamlit as st
from classifier import classify_po

st.set_page_config(page_title="PO Category Classifier", layout="centered")

DEFAULT_MODEL = st.secrets.get("GROQ_MODEL", "openai/gpt-oss-120b")
DEFAULT_TEMPERATURE = float(st.secrets.get("GROQ_TEMPERATURE", 0.0))

st.title("PO L1-L2-L3 Classifier")

if "last_inputs" not in st.session_state:
    st.session_state["last_inputs"] = None
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_latency_s" not in st.session_state:
    st.session_state["last_latency_s"] = None
if "is_processing" not in st.session_state:
    st.session_state["is_processing"] = False
if "po_description" not in st.session_state:
    st.session_state["po_description"] = ""
if "supplier" not in st.session_state:
    st.session_state["supplier"] = ""
if "model" not in st.session_state:
    st.session_state["model"] = DEFAULT_MODEL
if "temperature" not in st.session_state:
    st.session_state["temperature"] = DEFAULT_TEMPERATURE


@st.cache_data(show_spinner=False)
def classify_cached(po_description: str, supplier: str, model: str, temperature: float):
    return classify_po(po_description, supplier, model=model, temperature=temperature)


with st.form("po_form"):
    po_description = st.text_area(
        "PO Description",
        height=140,
        placeholder="Describe the purchase in a sentence or two.",
        help="Required. Be specific about the item or service.",
        key="po_description",
    )
    supplier = st.text_input(
        "Supplier (optional)",
        placeholder="e.g., ACME Supplies",
        help="Optional, but can help disambiguate.",
        key="supplier",
    )

    with st.expander("Advanced settings"):
        model = st.text_input(
            "Model",
            value=st.session_state["model"],
            help="Override the default model from secrets.",
            key="model",
        )
        temperature = st.number_input(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            value=float(st.session_state["temperature"]),
            help="Lower values are more deterministic.",
            key="temperature",
        )

    can_submit = bool(po_description.strip()) and not st.session_state["is_processing"]
    submitted = st.form_submit_button("Classify", disabled=not can_submit)

if submitted:
    supplier_normalized = supplier.strip() or "Not provided"
    current_inputs = (po_description.strip(), supplier_normalized, model.strip(), float(temperature))

    if st.session_state["last_inputs"] == current_inputs and st.session_state["last_result"]:
        result = st.session_state["last_result"]
        st.info("Reused the most recent result for these inputs.")
    else:
        st.session_state["is_processing"] = True
        start = time.perf_counter()
        try:
            with st.spinner("Classifying..."):
                result = classify_cached(
                    po_description.strip(),
                    supplier_normalized,
                    model.strip(),
                    float(temperature),
                )
        except ValueError as exc:
            st.session_state["is_processing"] = False
            st.error(str(exc))
            result = None
        except RuntimeError as exc:
            st.session_state["is_processing"] = False
            st.error(str(exc))
            result = None
        finally:
            st.session_state["is_processing"] = False
        elapsed = time.perf_counter() - start
        st.session_state["last_latency_s"] = elapsed
        st.session_state["last_inputs"] = current_inputs
        st.session_state["last_result"] = result

    if st.session_state["last_latency_s"] is not None:
        st.caption(f"Completed in {st.session_state['last_latency_s']:.2f}s")

    if result:
        try:
            st.json(json.loads(result))
        except Exception:
            st.warning("Model response is not valid JSON. Showing raw output.")
            st.text_area("Model output", value=result, height=220)
