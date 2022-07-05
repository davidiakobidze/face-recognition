import cv2 as cv
import face_recognition
import numpy as np

MIN_SIZE = 250
detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_distance(first_face, second_face):
    return round(sum([(i - j) ** 2 for i, j in zip(first_face, second_face)]), 3)


def get_features(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv.imdecode(np_arr, 1)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    rects = detector.detectMultiScale(gray)

    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    encodings = face_recognition.face_encodings(rgb, boxes)
    if encodings:
        return [round(i, 4) for i in encodings[0]]
    return []


if __name__ == '__main__':
    with open('result.png', 'rb') as file:
        embeddings_1 = get_features(file.read())
        print(embeddings_1)

    with open('biden2.png', 'rb') as file:
        embeddings_2 = get_features(file.read())
        print(embeddings_2)

    with open('trump.png', 'rb') as file:
        embeddings_3 = get_features(file.read())
        print(embeddings_3)

    distance = face_distance(embeddings_1, embeddings_2)
    print("Distance between 1 and 2 faces", distance)

    distance_2 = face_distance(embeddings_1, embeddings_3)
    print("Distance between 1 and 3 faces", distance_2)
