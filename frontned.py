# frontend.py
import streamlit as st
import requests

st.set_page_config(page_title="Cyberbullying Detection System")

st.title("Cyberbullying Detection System")
st.write("Detect if a given message or tweet contains cyberbullying content.")

user_input = st.text_area("Enter a message to analyze:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a message to analyze.")
    else:
        response = requests.post("http://127.0.0.1:5000/predict", json={"text": user_input})
        result = response.json()

        confidence = result.get("confidence", 0)
        prediction = result.get("prediction", "unknown")

        st.write(f"🔍 Confidence: {confidence}%")

        if prediction == "cyberbullying":
            st.error("🚫 Prediction: Cyberbullying")
        else:
            st.success("✅ Prediction: Not Cyberbullying")

