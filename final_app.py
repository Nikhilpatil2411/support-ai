import streamlit as st
import joblib
import google.generativeai as genai

# API Configuration - Using 'gemini-pro' which is the most stable name
genai.configure(api_key="AIzaSyCmEIhEgXi8wraN_K2X6nK41PZGelvyjk4")
# Yahan 'models/gemini-pro' likha hai, ye 404 nahi dega
model = genai.GenerativeModel('gemini-pro')

# Load ML Models
try:
    classifier = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    st.error(f"Model Load Error: {e}")

st.title("AI Customer Support System")

user_email = st.text_input("Your Email")
user_issue = st.text_area("Describe your issue")

if st.button("Generate Response"):
    if user_email and user_issue:
        # 1. Classification (Mera Local Model)
        issue_vec = vectorizer.transform([user_issue.lower()])
        category = classifier.predict(issue_vec)[0]
        
        st.success(f"Analysis Complete! Category: {category}")
        
        # 2. AI Response (Gemini)
        try:
            with st.spinner("AI is drafting..."):
                # Simple and direct call
                response = model.generate_content(f"Write a support email for: {user_issue}")
                st.subheader("Drafted AI Response")
                st.write(response.text)
        except Exception as e:
            # Agar ab bhi error aaye toh user ko category toh dikhegi hi
            st.error(f"API Error: {str(e)}")
            st.warning("Note: The local ML classification worked, but the AI API is currently busy.")