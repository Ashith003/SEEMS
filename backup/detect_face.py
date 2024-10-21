import os
import warnings
import cv2
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

# Loading the Haar cascade file for face detection
alg = "C:/Users/Ashith/PycharmProjects/BACKUP/Project - Hostel Entry and Exit/haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

def live_recognition():
    # Connect to the database
    conn = psycopg2.connect("postgres://avnadmin:AVNS_BLa8MIMKBVmJVz56ncO@pg-847ce69-ashithavgowda-f8b3.g.aivencloud.com:19235/defaultdb?sslmode=require")
    cur = conn.cursor()

    # Open a connection to the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream from webcam.")
        cur.close()
        conn.close()
        exit()

    ibed = imgbeddings()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=7, minSize=(100, 100))

        for (x, y, w, h) in faces:
            # Draw a thinner silver rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (192, 192, 192), 1)
            face_roi = frame[y:y + h, x:x + w]
            face_img = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            embedding = ibed.to_embeddings(face_img)
            embedding_list = embedding[0].tolist()

            # Convert embedding to a string format suitable for PostgreSQL
            embedding_array = np.array(embedding_list)
            embedding_str = "[" + ",".join(str(val) for val in embedding_array) + "]"

            # Execute the query with proper handling of database results
            try:
                cur.execute("SELECT picture FROM pictures ORDER BY embedding <-> %s LIMIT 1;", (embedding_str,))
                rows = cur.fetchall()

                if rows:
                    picture_name = rows[0][0]
                    base_name = os.path.splitext(picture_name)[0]
                    full_name = ''.join(filter(str.isalpha, base_name))  # Extract name by filtering non-alphabet characters

                    if full_name:
                        text_size = cv2.getTextSize(full_name, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                        text_x = x
                        text_y = y - 10 if y - 10 > 10 else y + 10
                        cv2.putText(frame, full_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    else:
                        print(f"Warning: No valid name extracted from {picture_name}")
            except Exception as e:
                print(f"Database query error: {e}")

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cur.close()
    conn.close()

if __name__ == "__main__":
    live_recognition()
