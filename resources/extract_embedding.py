from flask import Response
from flask_restful import Resource

from recognition.extract_embeddings import extract_embeddings


class ExtractEmbedding(Resource):
    def get(self):
        try:
            extract_embeddings()
            return Response({}, status=200, mimetype='application/json')

        except Exception as e:
            raise e
