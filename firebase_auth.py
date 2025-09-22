# firebase_auth.py
import streamlit as st
import pyrebase

firebaseConfig = {
    "apiKey": "YAIzaSyAO0O5rV6qBAH2rvKMcrJyEPofa5kRZKzoOUR_API_KEY",
    "authDomain": "procode-modeling-platform.firebaseapp.com",
    "projectId": "procode-modeling-platform",
    "storageBucket": "procode-modeling-platform.firebasestorage.app",
    "messagingSenderId": "557435622129",
    "appId": "1:557435622129:web:a2971130b34f3189ea0f50",
    "measurementId": "G-BEMNMY4JL6",
    "databaseURL": "https://dummy.firebaseio.com"
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

def email_password_login():
    st.subheader("📧 Login with Email & Password")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login with Email"):
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            st.session_state["user"] = email
            st.success(f"✅ Logged in as {email}")
            return True
        except Exception as e:
            st.error("❌ Invalid email or password")
            return False
    return False


def google_login():
    st.subheader("🔑 Login with Google")
    st.button(
        "[Click here to sign in with Google](https://accounts.google.com/o/oauth2/v2/auth)"
    )
    st.info("NOTE👉 Google login is not working as expected, Please use Email sign in")