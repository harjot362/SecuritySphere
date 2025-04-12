

# import cv2
# import os
# import face_recognition
# import numpy as np
# import subprocess  # macOS sound player

# # Load Haar cascade for face detection
# cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
# face_cascade = cv2.CascadeClassifier(cascade_path)

# # Load known faces
# known_faces_dir = os.path.join(os.path.dirname(__file__), "known_faces")
# known_face_encodings = []
# known_face_names = []

# for filename in os.listdir(known_faces_dir):
#     image_path = os.path.join(known_faces_dir, filename)
#     image = face_recognition.load_image_file(image_path)
#     encoding = face_recognition.face_encodings(image)

#     if encoding:  # Ensure encoding exists
#         known_face_encodings.append(encoding[0])
#         known_face_names.append(os.path.splitext(filename)[0])  # Remove file extension

# # Alarm sound file path
# sound_path = os.path.join(os.path.dirname(__file__), "alarm.mp3")

# # Create a folder for saving detected faces
# save_dir = os.path.join(os.path.dirname(__file__), "detected_faces")
# os.makedirs(save_dir, exist_ok=True)

# # Capture video from webcam
# cap = cv2.VideoCapture(0)

# # Check if the camera opened successfully
# if not cap.isOpened():
#     print("❌ Error: Could not open webcam.")
#     exit()

# image_count = 0  # Counter for saved images

# while True:
#     # Read the frame
#     ret, img = cap.read()
#     if not ret:
#         print("❌ Error: Could not read frame.")
#         break

#     # Convert to grayscale (for Haar cascade)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Detect faces using Haar cascade
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#     # Convert image to RGB (required for face_recognition)
#     rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#     for (x, y, w, h), face_encoding in zip(faces, face_encodings):
#         name = "Unknown"
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        
#         if True in matches:
#             matched_idx = matches.index(True)
#             name = known_face_names[matched_idx]

#         # Draw rectangle around detected face
#         color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
#         cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
#         cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

#         # Save unknown face
#         if name == "Unknown":
#             print("🚨 Unknown person detected! Triggering alarm.")
#             face_img = img[y:y+h, x:x+w]  # Crop face
#             image_count += 1
#             face_filename = os.path.join(save_dir, f"unknown_{image_count}.jpg")
#             cv2.imwrite(face_filename, face_img)
#             print(f"📸 Face saved: {face_filename}")

#             # Play alarm sound on macOS
#             if os.path.exists(sound_path):
#                 subprocess.run(["afplay", sound_path])  # Change to 'playsound' for Windows
#             else:
#                 print(f"⚠️ Alarm sound file not found at: {sound_path}")

#     # Display the frame
#     cv2.imshow('Security Camera', img)

#     # Stop if 'q' key is pressed
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break

# # Release the camera and close windows
# cap.release()
# cv2.destroyAllWindows()

from flask import Flask, render_template, Response
import cv2
import os
import face_recognition
import numpy as np
import subprocess  # macOS sound player

# Initialize Flask app
app = Flask(__name__)

# Load Haar cascade for face detection
cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load known faces
known_faces_dir = os.path.join(os.path.dirname(__file__), "known_faces")
known_face_encodings = []
known_face_names = []

for filename in os.listdir(known_faces_dir):
    image_path = os.path.join(known_faces_dir, filename)
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)

    if encoding:  # Ensure encoding exists
        known_face_encodings.append(encoding[0])
        known_face_names.append(os.path.splitext(filename)[0])  # Remove file extension

# Alarm sound file path
sound_path = os.path.join(os.path.dirname(__file__), "alarm.mp3")

# Create a folder for saving detected faces
save_dir = os.path.join(os.path.dirname(__file__), "detected_faces")
os.makedirs(save_dir, exist_ok=True)

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

image_count = 0  # Counter for saved images

def gen_frames():
    global image_count
    while True:
        # Read the frame
        ret, img = cap.read()
        if not ret:
            print("❌ Error: Could not read frame.")
            break

        # Convert to grayscale (for Haar cascade)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar cascade
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Convert image to RGB (required for face_recognition)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (x, y, w, h), face_encoding in zip(faces, face_encodings):
            name = "Unknown"
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            if True in matches:
                matched_idx = matches.index(True)
                name = known_face_names[matched_idx]

            # Draw rectangle around detected face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            # Save unknown face
            if name == "Unknown":
                print("🚨 Unknown person detected! Triggering alarm.")
                face_img = img[y:y+h, x:x+w]  # Crop face
                image_count += 1
                face_filename = os.path.join(save_dir, f"unknown_{image_count}.jpg")
                cv2.imwrite(face_filename, face_img)
                print(f"📸 Face saved: {face_filename}")

                # Play alarm sound on macOS
                if os.path.exists(sound_path):
                    subprocess.run(["afplay", sound_path])  # Change to 'playsound' for Windows
                else:
                    print(f"⚠️ Alarm sound file not found at: {sound_path}")

        # Convert image to JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            print("❌ Error encoding image.")
            break

        # Yield the frame to Flask to stream to the client
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)
