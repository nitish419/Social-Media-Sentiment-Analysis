# 📊 Social Media Sentiment Analysis Dashboard

## 📌 Overview
An end-to-end Machine Learning project that classifies social media text into Positive, Negative, or Neutral sentiments. Built using Natural Language Processing (NLP) techniques and a Logistic Regression model, served through an interactive Streamlit dashboard.

## 🎯 Problem Statement
Brands receive thousands of mentions daily. Manually analyzing this data is impossible. This project automates the process, allowing businesses to gauge customer satisfaction, track product feedback, and manage brand reputation instantly.

## 🛠 Tech Stack
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **NLP & Modeling:** Scikit-Learn (TF-IDF, Logistic Regression), NLTK, Regex
- **Frontend/UI:** Streamlit
- **Visualization:** Matplotlib, Seaborn

## 🏗 Architecture
Text Input -> Text Cleaning (Regex) -> TF-IDF Vectorization -> Logistic Regression Model -> Sentiment Prediction -> Streamlit UI.

## 🚀 How to Run Locally
1. Clone the repository:
   `git clone https://github.com/yourusername/Social-Media-Sentiment-Analyzer.git`
2. Install dependencies:
   `pip install -r requirements.txt`
3. Generate data & Train model:
   `python src/data_generator.py`
   `python src/train_model.py`
4. Launch Dashboard:
   `streamlit run app/dashboard.py`

## 📈 Learning Outcomes
- End-to-end ML pipeline creation.
- Feature extraction using TF-IDF.
- Deploying machine learning models via web interfaces.