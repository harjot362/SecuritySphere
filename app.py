# # from flask import Flask, render_template, Response
# # import cv2
# # import face_recognition
# # import os
# # from datetime import datetime
# # from playsound import playsound

# # app = Flask(__name__)

# # # Load known faces
# # known_faces_dir = 'known_faces'
# # known_encodings = []
# # known_names = []

# # for filename in os.listdir(known_faces_dir):
# #     image = face_recognition.load_image_file(f"{known_faces_dir}/{filename}")
# #     encoding = face_recognition.face_encodings(image)[0]
# #     known_encodings.append(encoding)
# #     known_names.append(os.path.splitext(filename)[0])

# # camera = cv2.VideoCapture(0)

# # def generate_frames():
# #     while True:
# #         success, frame = camera.read()
# #         if not success:
# #             break
# #         else:
# #             rgb_frame = frame[:, :, ::-1]
# #             face_locations = face_recognition.face_locations(rgb_frame)
# #             face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

# #             for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
# #                 matches = face_recognition.compare_faces(known_encodings, face_encoding)
# #                 name = "Unknown"

# #                 if True in matches:
# #                     first_match_index = matches.index(True)
# #                     name = known_names[first_match_index]
# #                 else:
# #                     # Save image of unknown face
# #                     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# #                     cv2.imwrite(f"detected_faces/unknown_{timestamp}.jpg", frame)
# #                     playsound('alarm.mp3')

# #                 cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
# #                 cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# #             ret, buffer = cv2.imencode('.jpg', frame)
# #             frame = buffer.tobytes()

# #             yield (b'--frame\r\n'
# #                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # @app.route('/video_feed')
# # def video_feed():
# #     return Response(generate_frames(),
# #                     mimetype='multipart/x-mixed-replace; boundary=frame')

# # if __name__ == "__main__":
# #     app.run(debug=True)

# from flask import Flask, render_template, Response
# import cv2
# import face_recognition
# import numpy as np
# import os
# from playsound import playsound
# import threading

# app = Flask(__name__)

# # Load known faces
# known_face_encodings = []
# known_face_names = []
# known_faces_dir = 'known_faces'

# for filename in os.listdir(known_faces_dir):
#     if filename.endswith('.jpg') or filename.endswith('.png'):
#         image_path = os.path.join(known_faces_dir, filename)
#         image = face_recognition.load_image_file(image_path)
#         encoding = face_recognition.face_encodings(image)[0]
#         known_face_encodings.append(encoding)
#         known_face_names.append(os.path.splitext(filename)[0])
#         print(f"Loaded encoding for: {filename}")

# # Alarm function

# def play_alarm():
#     playsound('alarm.mp3')
# # Frame generator for webcam feed
# def generate_frames():
#     video_capture = cv2.VideoCapture(0)

#     while True:
#         success, frame = video_capture.read()
#         if not success:
#             break

#         # Resize and convert to RGB
#         small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#         rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#             name = "Unknown"

#             if True in matches:
#                 first_match_index = matches.index(True)
#                 name = known_face_names[first_match_index]

#             # Scale back up
#             top *= 4
#             right *= 4
#             bottom *= 4
#             left *= 4

#             # Draw rectangle
#             color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
#             cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
#             cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

#             # Trigger alarm for unknown
#             if name == "Unknown":
#                 threading.Thread(target=play_alarm).start()

#         # Encode and yield frame
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# # Home page
# @app.route('/')
# def index():
#     return render_template('index.html')

# # Video feed route
# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, render_template, Response
import cv2
import face_recognition

app = Flask(__name__)

video_capture = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            # Process the frame with face_recognition
            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left) in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)