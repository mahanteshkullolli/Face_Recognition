import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

# load known faces
video = cv2.VideoCapture(0)
mantu = face_recognition.load_image_file("1.jpg")
mantu_encoding = face_recognition.face_encodings(mantu)[0]

known_face_encoding = [mantu_encoding]
known_face_names = ["mantu"]
students=known_face_names.copy()
face_location = []
face_encoding = []
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding,face_encoding )
        face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match = np.argmin(face_distance)

        if matches[best_match]:
            name = known_face_names[best_match]

        if name in known_face_names:
            font=cv2.FONT_HERSHEY_SIMPLEX
            bottomleftcorneroftext=(10,100)
            fontScale=1.5
            fontColor=(153,255,204)
            thickness=3
            linetype=2
            cv2.putText(frame,name+" present",bottomleftcorneroftext,font,fontScale,fontColor,thickness,linetype)

            if name in students:
                students.remove(name)
                current_time=now.strftime("%H-%M-%S")
                lnwriter.writerow([name,current_time])

    cv2.imshow("mantu", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
