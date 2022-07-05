# import the necessary packages
import os
import pickle

import cv2
import imutils
import numpy as np
from imutils import paths

from recognition.shared_parameters import DETECTOR, EMBEDDING_MODEL, DATASET, CONFIDENCE, EMBEDDINGS


def extract_embeddings():
    # load our serialized face detector from disk
    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join([DETECTOR, "deploy.prototxt"])
    modelPath = os.path.sep.join([DETECTOR, "res10_300x300_ssd_iter_140000.caffemodel"])
    detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch(EMBEDDING_MODEL)

    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(DATASET))

    knownEmbeddings = []
    knownNames = []
    # initialize the total number of faces processed
    total = 0

    # loop over the image paths
    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print(imagePath)
        print("[INFO] processing image {}/{}".format(i + 1,
                                                     len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        if image is None:
            continue
        image = imutils.resize(image, width=600)
        (h, w) = image.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections = detector.forward()

        # ensure at least one face was found
        if len(detections) > 0:

            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]
                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()

                knownNames.append(name)
                knownEmbeddings.append(vec.flatten())
                total += 1

    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total))
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(EMBEDDINGS, "wb")
    f.write(pickle.dumps(data))
    f.close()
