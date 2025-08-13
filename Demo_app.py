import streamlit as st
import joblib
import re
import string
from bs4 import BeautifulSoup

# --------------------------
# Load saved model and vectorizer
# --------------------------
model = joblib.load('logistic_regression_imdb_optimized.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# --------------------------
# Text cleaning function
# --------------------------
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = " ".join(text.split())
    return text

# --------------------------
# Predict function
# --------------------------
def predict_sentiment(user_input):
    clean_input = clean_text(user_input)
    vect_input = vectorizer.transform([clean_input])
    prediction = model.predict(vect_input)[0]
    return "Positive 😊" if prediction == 1 else "Negative 😞"

# --------------------------
# Streamlit UI
# --------------------------
st.title("🎬 IMDb Sentiment Analysis")
st.write("Enter a movie review to predict whether it is positive or negative.")

user_input = st.text_area("Enter your review:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a review first!")
    else:
        sentiment = predict_sentiment(user_input)
        st.success(f"Predicted Sentiment: {sentiment}")
