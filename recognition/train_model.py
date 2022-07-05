import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from recognition.shared_parameters import EMBEDDINGS, RECOGNISER, LE


def train_model():
    # load the face embeddings
    print("Loading face embeddings of the dataset")
    data = pickle.loads(open(EMBEDDINGS, "rb").read())

    # encode the labels
    print("Encoding image labels")
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    # train the model used to accept the 128-d embeddings of the face and
    # then produce the actual face recognition
    print("Training the model using SVM...")
    recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer.fit(data["embeddings"], labels)

    # write the actual face recognition model to disk
    f = open(RECOGNISER, "wb")
    f.write(pickle.dumps(recognizer))
    f.close()

    # write the label encoder to disk
    f = open(LE, "wb")
    f.write(pickle.dumps(le))
    f.close()
