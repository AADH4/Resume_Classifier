import streamlit as st
import pickle
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-pro')

# Load pre-trained model and vectorizer
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Function to get AI-generated feedback for applicants
def get_applicant_feedback(resume_text, prediction):
    try:
        if prediction == 'accepted':
            prompt = (
                "You are an expert career advisor. The following resume has been deemed acceptable. "
                "However, provide suggestions on how to further improve it to meet the standards of top-tier companies such as FAANG:\n\n"
                f"{resume_text}"
            )
        else:
            prompt = (
                "You are an expert career advisor. Analyze the following resume text and provide specific, constructive "
                "feedback on how to improve it for job applications, also explain WHY the candidate was most likely rejected:\n\n"
                f"{resume_text}"
            )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating feedback: {e}"

# Function to get AI-generated feedback for employers
def get_employer_feedback(resume_text, prediction):
    try:
        prompt = (
            "You are an expert hiring consultant. Analyze the following resume text and explain the reasons it was "
            f"'{prediction}' by an AI model. Provide a brief summary that can help employers understand the evaluation:\n\n"
            f"{resume_text}"
        )
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating feedback: {e}"

# Streamlit Layout
st.title("Resume Classifier")

# Add a radio button to choose between Employer and Applicant
role = st.radio("Select your role", ('Employer', 'Applicant'))

# If Applicant or Employer, allow them to either upload a resume or type one
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
    prediction = rf_model.predict(resume_tfidf)[0]

    # Display results
    if prediction == 'accepted':
        st.success("The resume has been accepted!")
    else:
        st.error("Unfortunately, the resume was rejected.")

    # Generate tailored feedback
    with st.spinner("Generating feedback..."):
        if role == 'Applicant':
            feedback = get_applicant_feedback(resume_text, prediction)
        elif role == 'Employer':
            feedback = get_employer_feedback(resume_text, prediction)

    st.write("Feedback:", feedback)
