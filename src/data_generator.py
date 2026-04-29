import pandas as pd
import os

def generate_synthetic_data():
    data = [
        ("I absolutely love the new iPhone, the battery life is amazing!", "Positive"),
        ("Worst delivery experience ever. My food was cold.", "Negative"),
        ("The movie was okay, nothing special but not bad.", "Neutral"),
        ("Highly recommend this laptop for students. Very fast.", "Positive"),
        ("Customer support is terrible. I have been on hold for an hour.", "Negative"),
        ("I bought this shirt yesterday. It is blue.", "Neutral"),
        ("What a fantastic product! Will buy again.", "Positive"),
        ("The app crashes every time I open it. Useless.", "Negative"),
        ("Just a standard coffee mug. It holds coffee.", "Neutral"),
        ("I am so happy with the quick refund process. Great job!", "Positive"),
        ("The quality is completely garbage. Do not buy.", "Negative"),
        ("It arrived on Tuesday.", "Neutral"),
        ("Best purchase of the year! So beautiful.", "Positive"),
        ("I hate the new UI update, it's so confusing.", "Negative"),
        ("The weather is nice today.", "Neutral")
    ] * 50  # Multiply by 50 to create a dataset of 750 rows for training

    df = pd.DataFrame(data, columns=["Text", "Sentiment"])
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/synthetic_reviews.csv", index=False)
    print("Dataset generated successfully at data/synthetic_reviews.csv")

if __name__ == "__main__":
    generate_synthetic_data()