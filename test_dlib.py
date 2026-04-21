import cv2
import face_recognition
import numpy as np
bgr = cv2.imread('known_faces/Inayat.jpg')
print("Image shape BGR:", bgr.shape, bgr.dtype)

# Try Grayscale for face_locations
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
print("Image shape Grayscale:", gray.shape, gray.dtype)

try:
    locations = face_recognition.face_locations(gray)
    print("Grayscale Locations found:", len(locations))
except Exception as e:
    print("Grayscale Location Error:", type(e), e)

# Try RGB for face_encodings
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
try:
    if 'locations' in locals() and locations:
        encodings = face_recognition.face_encodings(rgb, locations)
        print("Grayscale/RGB Hybrid Encodings found:", len(encodings))
except Exception as e:
    print("RGB Encoding Error:", type(e), e)

try:
    enc = face_recognition.face_encodings(gray, locations)
    print("Pure Grayscale Encodings found:", len(encodings))
except Exception as e:
    print("Grayscale Encoding Error:", type(e), e)
