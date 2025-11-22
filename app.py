import streamlit as st
import pandas as pd

from PIL import Image
import io

# Try to import OCR + embeddings, but keep the app robust if they fail
try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False

# Streamlit page config
st.set_page_config(
    page_title="Refugee Workforce Integration – AI Demo",
    layout="wide"
)

st.title("Refugee Workforce Integration – AI Demo Prototype")
st.write(
    "This demo showcases the core logic of the proposed model: "
    "**Document → Skill Extraction → Job Matching → Integration Support.**"
)

# --------------------------------------------------------------------
# 1. DOCUMENT UPLOAD + OCR (SIMULATED / BEST-EFFORT)
# --------------------------------------------------------------------
st.header("1. Upload Refugee Document (OCR)")

uploaded_file = st.file_uploader(
    "Upload an ID / certificate / experience document image",
    type=["png", "jpg", "jpeg"]
)

extracted_text = ""

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Document", width=300)
    st.write("Attempting OCR text extraction...")

    if HAS_TESSERACT:
        try:
            extracted_text = pytesseract.image_to_string(image)
        except Exception:
            # Fallback to demo text if Tesseract binary is not available
            extracted_text = (
                "Sample OCR output:\n"
                "Refugee has 3 years experience as a construction helper and mason. "
                "Worked on building sites, basic logistics and loading tasks."
            )
    else:
        # If pytesseract not installed, use demo text
        extracted_text = (
            "Sample OCR output:\n"
            "Refugee has 3 years experience as a construction helper and mason. "
            "Worked on building sites, basic logistics and loading tasks."
        )

st.text_area("Extracted Text (OCR Output)", value=extracted_text, height=150)

# --------------------------------------------------------------------
# 2. SIMPLE SKILL EXTRACTION (RULE-BASED DEMO)
# --------------------------------------------------------------------
st.header("2. Skill Extraction (Simplified NLP)")

sample_skills = [
    "mason", "construction", "plumbing", "carpentry",
    "logistics", "packing", "warehouse", "cooking",
    "helper", "electrician", "nursing", "patient care"
]

extracted_skills = []

if extracted_text:
    text_lower = extracted_text.lower()
    for skill in sample_skills:
        if skill in text_lower:
            extracted_skills.append(skill.capitalize())

if extracted_skills:
    st.write("**Detected Skills:** ", ", ".join(extracted_skills))
else:
    st.write("No
