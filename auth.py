# auth.py
import streamlit as st

# Dummy user store (replace with Firebase later)
USERS = {
    "admin": {"password": "procodeadmin@123", "name": "Admin"},
}

def login_form():
    st.subheader("🔐 Login with Username & Password")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USERS and USERS[username]["password"] == password:
            st.session_state["user"] = USERS[username]["name"]
            st.success(f"Welcome {USERS[username]['name']} 🎉")
            return True
        else:
            st.error("Invalid username or password")
            return False
    return False
