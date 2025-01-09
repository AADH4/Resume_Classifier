import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load pre-trained model and vectorizer
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Function to get feedback for rejected resumes
def get_rejection_feedback(resume_text):
    # Example feedback logic based on certain features
    feedback = "We recommend improving the clarity and structure of your resume. "
    # Here, add logic that analyzes the text further for common rejection reasons
    if len(resume_text.split()) < 200:
        feedback += "Consider expanding on your experience and skills section."
    if resume_text.count('relevant experience') == 0:
        feedback += "Be sure to highlight your relevant work experience."
    return feedback

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
            feedback = get_rejection_feedback(resume_text)
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
