import time
from pathlib import Path

from recognition.extract_embeddings import extract_embeddings
from recognition.train_model import train_model


def add_person(name: str, images: list):
    directory = "recognition/dataset/{}".format(name)
    Path(directory).mkdir(parents=True, exist_ok=True)
    for image in images:
        with open('{}/{}.png'.format(directory, time.time()), 'wb') as file:
            file.write(image)

    extract_embeddings()
    train_model()
