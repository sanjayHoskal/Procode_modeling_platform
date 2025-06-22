import pyrebase
from sales_prediction_project.firebase_config import firebaseConfig

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

def google_sign_in(id_token):
    try:
        # Exchange Google ID token for Firebase custom token
        user = auth.sign_in_with_custom_token(id_token)
        return user
    except Exception as e:
        return None
