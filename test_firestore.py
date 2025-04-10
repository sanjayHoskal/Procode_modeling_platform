import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK with your service account key
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

# Firestore Client
db = firestore.client()

# Add a test document to Firestore
doc_ref = db.collection("test_collection").document("test_doc")
doc_ref.set({
    "message": "Hello, Firebase!",
    "status": "Success",
    "timestamp": firebase_admin.firestore.SERVER_TIMESTAMP
})

print("Test document added to Firestore successfully!")
