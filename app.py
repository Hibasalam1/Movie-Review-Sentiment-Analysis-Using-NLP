import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Title
st.title("🎬 Movie Review Sentiment Analyzer")

# Input box
review = st.text_area("Enter your movie review:")

# Button
if st.button("Predict"):

    review = clean_text(review)

    review_tfidf = vectorizer.transform([review])

    prediction = model.predict(review_tfidf)

    if prediction[0] == 1:
        st.success("Positive 😊")
    else:
        st.error("Negative 😞")