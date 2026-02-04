import streamlit as st
from groq import Groq
from prompts import SYSTEM_PROMPT

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

DEFAULT_MODEL = st.secrets.get("GROQ_MODEL", "openai/gpt-oss-120b")
DEFAULT_TEMPERATURE = float(st.secrets.get("GROQ_TEMPERATURE", 0.0))


def _build_user_prompt(po_description: str, supplier: str) -> str:
    return f"""
PO Description:
{po_description}

Supplier:
{supplier}
"""


def classify_po(
    po_description: str,
    supplier: str = "Not provided",
    model: str | None = None,
    temperature: float | None = None,
):
    if not isinstance(po_description, str) or not po_description.strip():
        raise ValueError("po_description must be a non-empty string.")

    supplier_value = supplier if isinstance(supplier, str) and supplier.strip() else "Not provided"
    user_prompt = _build_user_prompt(po_description.strip(), supplier_value.strip())

    try:
        response = client.chat.completions.create(
            model=model or DEFAULT_MODEL,
            temperature=DEFAULT_TEMPERATURE if temperature is None else float(temperature),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"Classification request failed: {exc}") from exc

    return response.choices[0].message.content
