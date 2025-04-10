import pyrebase

# Pyrebase Configuration
firebase_config = {
    "apiKey": "AIzaSyAO0O5rV6qBAH2rvKMcrJyEPofa5kRZKzo",
    "authDomain": "procode-modeling-platform.firebaseapp.com",
    "databaseURL": "",
    "projectId": "procode-modeling-platform",
    "storageBucket": "procode-modeling-platform.firebasestorage.app"
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# Test User Signup
email = "testuser@example.com"
password = "testpassword123"
user = auth.create_user_with_email_and_password(email, password)

print(f"User created successfully! Email: {user['email']}")

# Test Login
login = auth.sign_in_with_email_and_password(email, password)
print(f"Login successful! Token: {login['idToken']}")
