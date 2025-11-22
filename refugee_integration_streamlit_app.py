import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import pytesseract
from PIL import Image
import io

st.set_page_config(page_title="Refugee Workforce Integration AI Demo", layout="wide")

st.title("Refugee Workforce Integration – AI Demo Prototype")
st.write("This is a simplified sharable demo showcasing the core logic: OCR → Skill Extraction → Job Matching.")

# ------------------- OCR Section -------------------
st.header("1. Upload Refugee Document (OCR Simulation)")
uploaded_file = st.file_uploader("Upload ID / Certificate (Image)", type=["jpg", "png", "jpeg"])

extracted_text = ""
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Document", width=300)
    st.write("Extracting text… (demo mode)")
    try:
        extracted_text = pytesseract.image_to_string(image)
    except:
        extracted_text = "Sample OCR output: Mason experience, construction helper, Hindi speaker, 3 years experience."

st.text_area("Extracted Text (OCR Output)", extracted_text, height=150)

# ------------------- Skill Extraction -------------------
st.header("2. Skill Extraction (Simplified NLP)")

sample_skills = ["mason", "plumbing", "carpentry", "logistics", "packing", "cooking", "helper", "electrician"]

extracted_skills = []
if extracted_text:
    text_lower = extracted_text.lower()
    for s in sample_skills:
        if s in text_lower:
            extracted_skills.append(s.capitalize())

st.write("**Detected Skills:**", extracted_skills)

# ------------------- Job Matching Engine -------------------
st.header("3. Job Matching Engine (Vector Similarity Demo)")

job_data = {
    "Job Role": ["Construction Helper", "Logistics Assistant", "Kitchen Staff", "Electrician Trainee"],
    "Required Skills": [
        "mason helper construction",
        "logistics packing warehouse",
        "cooking kitchen helper",
        "electrician wiring"
    ]
}

jobs_df = pd.DataFrame(job_data)
st.write("### Job Listings Database", jobs_df)

model = SentenceTransformer("all-MiniLM-L6-v2")

if extracted_skills:
    refugee_profile = " ".join(extracted_skills)
    refugee_emb = model.encode(refugee_profile)

    job_scores = []
    for req in jobs_df["Required Skills"]:
        job_emb = model.encode(req)
        score = util.cos_sim(refugee_emb, job_emb).item()
        job_scores.append(score)

    jobs_df["Match Score"] = job_scores
    st.write("### Ranked Job Matches", jobs_df.sort_values(by="Match Score", ascending=False))

# ------------------- Chatbot Placeholder -------------------
st.header("4. Integration Assistance Chatbot (Placeholder)")
st.write("In a full system, this area provides cultural/legal guidance using an LLM.")
user_query = st.text_input("Ask a question (demo):")
if user_query:
    st.write("**Sample Response:** In India, ensure your FRRO registration is updated. Employers may require ID and skill proof.")

st.success("Demo Complete – You can host this app online using Streamlit Community Cloud.")
