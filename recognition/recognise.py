import os
import pickle

import cv2
import imutils
import numpy as np

from recognition.shared_parameters import DETECTOR, EMBEDDING_MODEL, RECOGNISER, LE, CONFIDENCE


def recognise(image) -> tuple:
    # load our serialized face detector from disk
    protoPath = os.path.sep.join([DETECTOR, "deploy.prototxt"])
    modelPath = os.path.sep.join([DETECTOR, "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    # load our serialized face embedding model from disk
    embedder = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL)
    # load the actual face recognition model along with the label encoder
    recognizer = pickle.loads(open(RECOGNISER, "rb").read())
    le = pickle.loads(open(LE, "rb").read())

    np_image_arr = np.fromstring(image, np.uint8)
    image = cv2.imdecode(np_image_arr, 1)

    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections
        if confidence > CONFIDENCE:
            # compute the (x, y)-coordinates of the bounding box for the
            # face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
                                             (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the associated
            # probability
            percentage = proba * 100
            text = "{}: {:.2f}%".format(name, percentage)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            print(text)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            return name, percentage
    return "None", 0
    # show the output image
    # cv2.imwrite("result.png", image)
