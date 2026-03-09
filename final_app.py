


import streamlit as st
import joblib
from google import genai 
import PyPDF2
from docx import Document
import re

# --- 1. Settings & API Setup ---
st.set_page_config(page_title="AI Support Portal", layout="wide")

# Dost ki ya nayi API Key yahan dalo
API_KEY = "AIzaSyB2Rz4pCPTfA6yp-fxSVLMKXN1RtqQML4g"
client = genai.Client(api_key=API_KEY)

# --- 2. Load ML Models ---
@st.cache_resource
def load_models():
    try:
        classifier = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return classifier, vectorizer
    except Exception as e:
        st.error(f"Model Load Error: {e}")
        return None, None

classifier, vectorizer = load_models()

# --- 3. Extraction & Email Regex Function ---
def get_text_from_file(uploaded_file):
    text = ""
    try:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
        return text
    except Exception as e:
        st.error(f"File Reading Error: {e}")
        return ""

def extract_email(text):
    # Yeh function text mein se email dhoond lega
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None

# --- 4. UI Layout ---
st.title("🤖 AI Customer Support System")
st.markdown("---")

# --- SECTION A: EMAIL ID PROVIDE KARO ---
st.subheader("Step 1: Customer Details")
col1, col2 = st.columns(2)

with col1:
    email_choice = st.radio("Email ID kaise provide karni hai?", ["Type Email ID", "Upload Email Document"])

customer_email = ""
extracted_text = ""

if email_choice == "Type Email ID":
    customer_email = st.text_input("Enter Customer Email ID:", placeholder="example@mail.com")
else:
    email_file = st.file_uploader("Upload Email Doc (PDF/Docx)", type=["pdf", "docx"], key="email_pdf")
    if email_file:
        extracted_text = get_text_from_file(email_file)
        customer_email = extract_email(extracted_text)
        if customer_email:
            st.success(f"Detected Email: {customer_email}")
        else:
            st.warning("File mein koi Email ID nahi mili. Kripya manually type karein.")

# --- SECTION B: ISSUE PROVIDE KARO ---
st.subheader("Step 2: Describe the Issue")
user_issue = st.text_area("Yahan issue likho (Current Problem):", height=100)

# --- 5. Action Section ---
st.markdown("---")
if st.button("Generate Personalized Response"):
    if user_issue and customer_email:
        if classifier and vectorizer:
            try:
                # 1. Classification (Local ML Model)
                issue_vec = vectorizer.transform([user_issue.lower()])
                category = classifier.predict(issue_vec)[0]
                st.info(f"**Detected Category:** {category}")
                
                # 2. AI Generation (Using 2.0-flash-lite for better quota)
                with st.spinner("AI draft taiyar kar raha hai..."):
                    prompt = f"""
                    You are a professional support agent. 
                    Customer Email: {customer_email}
                    Issue Category: {category}
                    Current Issue: {user_issue}
                    
                    Task: Write a professional email response to the customer. 
                    Address them by their email prefix if name is unknown. 
                    Be empathetic and provide a solution for the {category} issue.
                    """
                    
                    response = client.models.generate_content(
                        model="gemini-2.5-flash", 
                        contents=prompt
                    )
                    
                    st.subheader("AI Drafted Response:")
                    st.code(response.text, language="markdown")
                    st.success(f"Draft ready for {customer_email}!")
                    
            except Exception as e:
                st.error(f"Generation Error: {e}")
        else:
            st.error("Model files load nahi ho payi hain!")
    else:
        st.warning("Pehle Email ID aur Issue dono provide karo!")
