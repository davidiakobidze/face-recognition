from flask import Response
from flask_restful import Resource

from recognition.extract_embeddings import extract_embeddings
from recognition.train_model import train_model


class Train(Resource):
    def get(self):
        try:
            extract_embeddings()
            train_model()
            return Response({}, status=200, mimetype='application/json')

        except Exception as e:
            raise e
