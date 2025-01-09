import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load the trained model and TF-IDF vectorizer
with open('rf_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as tfidf_file:
    tfidf_vectorizer = pickle.load(tfidf_file)

# Function to make predictions
def predict_resume_status(resume_text):
    # Transform the input text using the trained TF-IDF vectorizer
    resume_tfidf = tfidf_vectorizer.transform([resume_text])
    
    # Predict the label (Accepted or Rejected) using the trained model
    prediction = rf_model.predict(resume_tfidf)
    
    return prediction[0]

# Streamlit UI
st.title("Resume Classifier: Accepted or Rejected")
st.write("Enter the resume text in the box below and get the prediction!")

# Resume text input
resume_input = st.text_area("Resume Text", height=300)

# Button to make predictions
if st.button("Predict"):
    if resume_input.strip() != "":
        # Make the prediction
        result = predict_resume_status(resume_input)
        
        if result == 'accepted':
            st.success("Predicted Label: **Accepted**")
        else:
            st.error("Predicted Label: **Rejected**")
    else:
        st.warning("Please enter the resume text before predicting.")

