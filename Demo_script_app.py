import streamlit as st
import joblib
import re
import string
from bs4 import BeautifulSoup

# --------------------------
# Load saved model and vectorizer
# -------------------------- 
model = joblib.load('logistic_regression_imdb_optimized.pkl')  # Load the pre-trained Logistic Regression model
vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Load the pre-trained TF-IDF vectorizer for text transformation

# --------------------------
# Text cleaning function
# --------------------------
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags from the input text
    text = text.lower()  # Convert text to lowercase for consistency
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs (http, https, www)
    text = re.sub(r'\d+', '', text)  # Remove numbers from the text
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = " ".join(text.split())  # Remove extra spaces and normalize whitespace
    return text  # Return the cleaned text

# --------------------------
# Predict function
# --------------------------
def predict_sentiment(user_input):
    clean_input = clean_text(user_input)  # Clean the user-provided review text
    vect_input = vectorizer.transform([clean_input])  # Transform cleaned text into TF-IDF features
    prediction = model.predict(vect_input)[0]  # Predict sentiment using the trained model (0 = Negative, 1 = Positive)
    return "Positive 😊" if prediction == 1 else "Negative 😞"  # Return sentiment label with an emoji

# --------------------------
# Streamlit UI
# --------------------------
st.title("🎬 IMDb Sentiment Analysis")  # Set the title of the Streamlit app
st.write("Enter a movie review to predict whether it is positive or negative.")  # Display instructions for the user

user_input = st.text_area("Enter your review:")  # Create a text area for user input

if st.button("Predict Sentiment"):  # Create a button to trigger sentiment prediction
    if user_input.strip() == "":  # Check if the input is empty or only whitespace
        st.warning("Please enter a review first!")  # Display a warning if no input is provided
    else:
        sentiment = predict_sentiment(user_input)  # Call the predict function with user input
        st.success(f"Predicted Sentiment: {sentiment}")  # Display the predicted sentiment