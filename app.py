import streamlit as st
import pickle
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize model
model = genai.GenerativeModel('gemini-1.5-pro')

# Load pre-trained model and vectorizer
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Function to get AI-generated feedback for rejected resumes
def get_ai_feedback(resume_text):
    try:
        prompt = (
            "You are an expert career advisor. Analyze the following resume text and provide specific, constructive "
            "feedback on how to improve it for job applications:\n\n"
            f"{resume_text}"
        )
        # Generate feedback using the model
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating feedback: {e}"

# Streamlit Layout
st.title("Resume Classifier")

# Add a radio button to choose between Employer and Applicant
role = st.radio("Select your role", ('Employer', 'Applicant'))

# If Applicant or Employer, allow them to either upload a resume or type one
if role in ['Applicant', 'Employer']:
    st.subheader("Upload or Type Resume for Evaluation")

    # Option to upload resume
    uploaded_resume = st.file_uploader("Choose a resume (in .txt format)", type='txt')

    # Option to type resume manually
    typed_resume = st.text_area("Alternatively, type your resume here:")

    # Decide whether to use uploaded or typed resume
    if uploaded_resume is not None:
        # Read the uploaded resume
        resume_text = uploaded_resume.read().decode("utf-8")
    elif typed_resume.strip() != "":
        resume_text = typed_resume
    else:
        st.warning("Please upload or type your resume to proceed.")
        resume_text = None

    if resume_text is not None:
        # Predict if accepted or rejected
        resume_tfidf = tfidf_vectorizer.transform([resume_text])
        prediction = rf_model.predict(resume_tfidf)

        # Display results
        if prediction == ['accepted']:
            st.success("Your resume has been accepted!")
        else:
            st.error("Unfortunately, your resume was rejected.")
            with st.spinner("Generating feedback..."):
                feedback = get_ai_feedback(resume_text)
            st.write("Feedback:", feedback)

elif role == 'Employer':
    st.subheader("Welcome Employer!")
    st.write("""
        Our resume classifier helps predict whether a resume is likely to be accepted or rejected based on various factors such as:
        - Relevance of skills and experience
        - Structure and formatting of the resume
        - Keyword match with job descriptions
    """)
    st.write("You can use this tool to analyze resumes and understand why certain resumes may be rejected or accepted.")
