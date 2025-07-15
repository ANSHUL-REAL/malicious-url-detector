import streamlit as st
import joblib
import re
from datetime import datetime

# Load model and vectorizer
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except FileNotFoundError:
    st.error("Models not found. Please make sure 'model.pkl' and 'vectorizer.pkl' are in the current directory.")
    st.stop()

# Page setup
st.set_page_config(
    page_title="Malicious URL Detector",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Page title and instructions
st.title("Malicious URL Detector")
st.write("Enter a URL below to check whether it's safe or potentially harmful.")

# Input field for URL
url_input = st.text_input(
    "URL",
    placeholder="https://example.com",
    help="Provide a valid URL beginning with http:// or https://"
)

# Analyze button logic
if st.button("Analyze"):
    if not url_input.strip():
        st.warning("Please enter a URL to analyze.")
    elif not re.match(r'^https?://', url_input):
        st.error("The URL must start with http:// or https://")
    else:
        try:
            # Vectorize and predict
            X = vectorizer.transform([url_input])
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0]
            confidence = max(probability)

            # Output result
            if prediction == 1:
                st.success("This URL appears to be safe.")
                st.markdown(f"Confidence: {confidence:.2%}")
            else:
                st.error("This URL appears to be malicious.")
                st.markdown(f"Confidence: {confidence:.2%}")

            # Timestamp
            st.markdown(f"Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            st.error("An error occurred while analyzing the URL.")
            st.code(str(e))
