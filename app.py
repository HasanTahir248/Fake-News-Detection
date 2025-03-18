import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("📰 Fake News Detection System")
st.subheader("Enter a news article to check if it's real or fake:")

# User input
news_text = st.text_area("Paste news article here...", height=200)

if st.button("Check"):
    if news_text.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Preprocess input and predict
        news_tfidf = vectorizer.transform([news_text])
        prediction = model.predict(news_tfidf)[0]
        result = "🟢 Real News" if prediction == 0 else "🔴 Fake News"
        st.subheader(f"Prediction: {result}")
