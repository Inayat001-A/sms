import sys
try:
    import face_recognition_models
    print("Successfully imported face_recognition_models")
    print("Location:", face_recognition_models.__file__)
except Exception as e:
    print("Failed to import face_recognition_models:")
    import traceback
    traceback.print_exc()

try:
    import dlib
    print("Successfully imported dlib v", dlib.__version__)
except Exception as e:
    print("Failed to import dlib:")
    import traceback
    traceback.print_exc()
