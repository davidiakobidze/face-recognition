import json

from flask import Response, request
from flask_restful import Resource

from recognition.recognise import recognise


class Recognise(Resource):
    def post(self):
        try:
            files = request.files
            images = [files[key].read() for key in list(files)]
            recognise_result = {"None": 0}

            for image in images:
                name, percentage = recognise(image)
                if recognise_result.get(name):
                    recognise_result[name] += percentage
                else:
                    recognise_result[name] = percentage
            sorted_result = {k: v for k, v in sorted(recognise_result.items(), key=lambda item: item[1])}
            name = list(sorted_result)[-1]
            result = {
                "name": name,
                "score": round(recognise_result[name] / len(images), 2)
            }
            print(result)

            if result['score'] > 81:
                return Response(response=json.dumps(result), status=200, mimetype='application/json')
            return Response({}, status=200, mimetype='application/json')

        except Exception as e:
            return Response(status=500)
