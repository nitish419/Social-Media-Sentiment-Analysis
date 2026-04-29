import pandas as pd
import re
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def clean_text(text):
    text = re.sub(r'http\S+', '', text) # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special chars and numbers
    text = text.lower().strip() # Lowercase and remove trailing spaces
    return text

def train():
    print("Loading data...")
    df = pd.read_csv("data/synthetic_reviews.csv")
    
    print("Cleaning text...")
    df['Cleaned_Text'] = df['Text'].apply(clean_text)
    
    X = df['Cleaned_Text']
    y = df['Sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)
    
    # Evaluation
    predictions = model.predict(X_test_vec)
    acc = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    
    # Save Model and Vectorizer
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    print("Model and Vectorizer saved in /models folder.")

if __name__ == "__main__":
    train()