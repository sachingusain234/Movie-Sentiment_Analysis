import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

import streamlit as st

# Page config
st.set_page_config(page_title="Movie Review Sentiment Classifier", page_icon="🎬", layout="centered")

# Title
st.title("🎬 Movie Review Sentiment Classifier")
st.write("Enter a movie review below and click **Classify** to predict whether it's Positive or Negative.")

# Load model
model = load_model('Simple_RNN_imdb.h5')

# Load IMDB word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


# Preprocessing
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    return sentiment, prediction[0][0]


# -----------------------------
# FIX: modify session_state BEFORE widget creation
# -----------------------------

# Init session state
if "review_text" not in st.session_state:
    st.session_state.review_text = ""

# Buttons first (before text_area)
col1, col2 = st.columns([1, 1])

with col2:
    if st.button("✨ Insert Example Review"):
        st.session_state.review_text = "This movie was absolutely amazing!"

with col1:
    classify = st.button("🔍 Classify Review")

# Now render the text box AFTER state updates
review = st.text_area(
    "✍️ Write your movie review here:",
    height=150,
    key="review_text"
)


# --- Prediction ---
if classify:
    if not st.session_state.review_text.strip():
        st.warning("⚠️ Please enter a review before clicking *Classify*.")
    else:
        sentiment, score = predict_sentiment(st.session_state.review_text)

        st.subheader("📌 Prediction Result")
        st.write(f"**Review:** {st.session_state.review_text}")

        if sentiment == "Positive":
            st.success(f"🟢 Sentiment: **Positive**")
        else:
            st.error(f"🔴 Sentiment: **Negative**")

        st.info(f"📊 Confidence Score: **{score:.4f}**")
