import face_recognition
print("face_recognition imported successfully!")
try:
    from face_recognition_models import face_recognition_model_location
    print("models located at:", face_recognition_model_location())
except Exception as e:
    print("Error:", e)
