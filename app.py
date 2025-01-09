import streamlit as st
import pickle
import os
from dotenv import load_dotenv
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-pro')

# Load pre-trained model and vectorizer
with open('model_v2.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('tfidf_vectorizer_v2.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Function to calculate keyword match
def keyword_match(resume_text, job_description):
    try:
        # Combine resume and job description into a single dataset
        documents = [job_description, resume_text]
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        # Cosine Similarity between job description and resume
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        similarity_percentage = round(similarity[0][0] * 100, 2)
        
        # Extract missing keywords
        job_keywords = set(vectorizer.get_feature_names_out())
        resume_keywords = set(resume_text.split())
        missing_keywords = job_keywords - resume_keywords
        
        return similarity_percentage, missing_keywords
    except Exception as e:
        return 0, f"Error: {e}"

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
st.title("Resume Classifier with Keyword Match")

# Add a radio button to choose between Employer and Applicant
role = st.radio("Select your role", ('Employer', 'Applicant'))

# If Applicant or Employer, allow them to either upload a resume or type one
st.subheader("Upload or Type Resume for Evaluation")

# Option to upload resume
uploaded_resume = st.file_uploader("Choose a resume (in .txt format)", type='txt')

# Option to type resume manually
typed_resume = st.text_area("Alternatively, type your resume here:")

# Option to paste job description
job_description = st.text_area("Paste the job description here (optional):")

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

    # Keyword match analysis if job description is provided
    if job_description.strip():
        match_percentage, missing_keywords = keyword_match(resume_text, job_description)
        st.write(f"Keyword Match Percentage: **{match_percentage}%**")
        
        if missing_keywords:
            st.write("**Missing Keywords:**")
            st.write(", ".join(missing_keywords))
        else:
            st.write("Great! Your resume covers all important keywords.")
