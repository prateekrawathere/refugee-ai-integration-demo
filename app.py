import streamlit as st
import pandas as pd
from PIL import Image

# Try optional imports
try:
    import pytesseract
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_EMBEDDINGS = True
except Exception:
    HAS_EMBEDDINGS = False


st.set_page_config(page_title="Refugee Workforce Integration – AI Demo", layout="wide")

st.title("Refugee Workforce Integration – AI Demo Prototype")
st.write(
    "This demo showcases the core logic: **Document → Skill Extraction → Job Matching → Integration Support**."
)

# --------------------------------------------------------------------
# 1. DOCUMENT UPLOAD + OCR
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
            extracted_text = (
                "Sample OCR output: Refugee has 3 years experience as a construction helper "
                "and mason. Worked on building sites, basic logistics and loading tasks."
            )
    else:
        extracted_text = (
            "Sample OCR output: Refugee has 3 years experience as a construction helper "
            "and mason. Worked on building sites, basic logistics and loading tasks."
        )

st.text_area("Extracted Text (OCR Output)", extracted_text, height=150)

# --------------------------------------------------------------------
# 2. SKILL EXTRACTION
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
    st.write("**Detected Skills:** " + ", ".join(extracted_skills))
else:
    st.write("No predefined skills detected. Try including words like 'mason', 'logistics', or 'nursing'.")

# --------------------------------------------------------------------
# 3. JOB MATCHING ENGINE
# --------------------------------------------------------------------
st.header("3. Job Matching Engine")

job_data = {
    "Job Role": [
        "Construction Helper",
        "Logistics Assistant",
        "Warehouse Loader",
        "Kitchen Staff",
        "Nursing Assistant",
        "Electrician Trainee",
    ],
    "Required Skills": [
        "mason helper construction site",
        "logistics packing loading",
        "warehouse loading packing",
        "cooking kitchen helper",
        "nursing patient care basic first aid",
        "electrician wiring basic electrical",
    ],
    "Sector": [
        "Construction",
        "Logistics",
        "Logistics",
        "Food Services",
        "Healthcare",
        "Construction",
    ],
}

jobs_df = pd.DataFrame(job_data)
st.write("### Sample Job Database")
st.dataframe(jobs_df, use_container_width=True)

if extracted_skills and HAS_EMBEDDINGS:
    st.write("Calculating match scores using semantic embeddings...")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    refugee_profile = " ".join(extracted_skills)
    refugee_emb = model.encode(refugee_profile)

    scores = []
    for req in jobs_df["Required Skills"]:
        job_emb = model.encode(req)
        score = util.cos_sim(refugee_emb, job_emb).item()
        scores.append(score)

    jobs_df["Match Score"] = scores
    jobs_df_sorted = jobs_df.sort_values(by="Match Score", ascending=False)

    st.write("### Ranked Job Matches")
    st.dataframe(jobs_df_sorted, use_container_width=True)

elif extracted_skills and not HAS_EMBEDDINGS:
    st.warning(
        "Sentence-transformers is not installed, so semantic matching is disabled. "
        "You can still present this as a conceptual demo."
    )
else:
    st.info("Upload a document and extract skills to see job matching results.")

# --------------------------------------------------------------------
# 4. CHATBOT (PLACEHOLDER)
# --------------------------------------------------------------------
st.header("4. Integration Support Assistant (Demo)")

st.write(
    "This section simulates a chatbot. A full version would use a Generative AI model "
    "to provide legal, cultural, and workplace guidance."
)

user_query = st.text_input("Ask a question about work in India:")

if user_query:
    if "document" in user_query.lower():
        response = (
            "In India, employers typically require ID proof and any available skill "
            "documents. Registered refugees may need FRRO registration."
        )
    elif "hours" in user_query.lower():
        response = (
            "Many entry-level roles in India involve 8–9 hour shifts, usually 6 days a week."
        )
    else:
        response = (
            "Demo response: In the full version, an AI assistant would give detailed "
            "legal and cultural information in multiple languages."
        )

    st.write("**Assistant:**", response)

st.success("Demo complete! You can now share this Streamlit app link in your assignment.")
