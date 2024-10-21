import cv2
import os
import warnings
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2

warnings.filterwarnings("ignore")

alg = "C:/Users/Ashith/PycharmProjects/BACKUP/Project - Hostel Entry and Exit/haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

input_dir = "C:/Users/Ashith/PycharmProjects/BACKUP/Project - Hostel Entry and Exit/TrainingImage"
output_dir = "C:/Users/Ashith/PycharmProjects/BACKUP/Project - Hostel Entry and Exit/stored-faces/"
desired_width = 150
desired_height = 150
os.makedirs(output_dir, exist_ok=True)


def process_images():
    for filename in os.listdir(input_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(input_dir, filename)
            color_img = cv2.imread(file_path)
            faces = haar_cascade.detectMultiScale(cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY), scaleFactor=1.1,
                                                  minNeighbors=5, minSize=(100, 100))

            if len(faces) == 0:
                print(f"No faces detected in {filename}")
            else:
                print(f"Detected {len(faces)} faces in {filename}")

            for i, (x, y, w, h) in enumerate(faces):
                padding = 10
                x, y, w, h = max(x - padding, 0), max(y - padding, 0), min(x + w + padding, color_img.shape[1]), min(
                    y + h + padding, color_img.shape[0])
                cropped_image = color_img[y:h, x:w]
                resized_image = cv2.resize(cropped_image, (desired_width, desired_height))
                base_name = os.path.splitext(filename)[0]
                target_file_name = os.path.join(output_dir, f'{base_name}_{i}.jpg')
                cv2.imwrite(target_file_name, resized_image)
                print(f"Saved resized face to: {target_file_name}")


def store_embeddings():
    conn = psycopg2.connect(
        "postgres://avnadmin:AVNS_q-VA6mAIg2KrcCKdKfs@pg-4bb659c-seams1318-130a.h.aivencloud.com:20039/defaultdb?sslmode=require")
    cur = conn.cursor()
    ibed = imgbeddings()

    for filename in os.listdir(output_dir):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img = Image.open(os.path.join(output_dir, filename))
            embedding = ibed.to_embeddings(img)
            embedding_list = embedding[0].tolist()
            base_name = os.path.splitext(filename)[0].rsplit('_', 1)[0]
            name = ''.join(filter(str.isalpha, base_name))

            if not name:
                print(f"Skipped {filename} as its name is not alphabetic.")
                continue

            cur.execute(
                "INSERT INTO pictures (picture, embedding) VALUES (%s, %s) ON CONFLICT (picture) DO UPDATE SET embedding = EXCLUDED.embedding;",
                (name, embedding_list))
            print(f"Stored embedding for: {filename}")

    conn.commit()
    cur.close()
    conn.close()
