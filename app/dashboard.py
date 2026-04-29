import streamlit as st
import joblib
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load artifacts
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

st.title("📊 Social Media Sentiment Analysis Dashboard")
st.markdown("Enter a review, tweet, or comment below to analyze its sentiment using Machine Learning.")

# User Input
user_input = st.text_area("Enter Text Here:", placeholder="e.g., I absolutely love the new update, it works perfectly!")

if st.button("Analyze Sentiment"):
    if user_input:
        cleaned_input = clean_text(user_input)
        vec_input = vectorizer.transform([cleaned_input])
        
        prediction = model.predict(vec_input)[0]
        probabilities = model.predict_proba(vec_input)[0]
        classes = model.classes_
        
        st.subheader(f"Predicted Sentiment: **{prediction}**")
        
        # Color coding
        if prediction == "Positive":
            st.success("This text has a Positive sentiment! 😃")
        elif prediction == "Negative":
            st.error("This text has a Negative sentiment! 😡")
        else:
            st.info("This text has a Neutral sentiment. 😐")
            
        # Visualization
        st.write("### Prediction Confidence")
        fig, ax = plt.subplots(figsize=(6,3))
        sns.barplot(x=classes, y=probabilities, palette="viridis", ax=ax)
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        
    else:
        st.warning("Please enter some text to analyze.")

st.sidebar.header("About")
st.sidebar.info("This dashboard uses an NLP pipeline (TF-IDF) and a Logistic Regression model to classify text into Positive, Negative, or Neutral categories. Built for demonstration purposes.")