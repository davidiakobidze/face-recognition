from flask import Response, request
from flask_restful import Resource

from recognition.add_person import add_person


class AddPerson(Resource):
    def post(self):
        try:
            files = request.files
            images = [files[key].read() for key in list(files)]
            name = request.form.get('name')
            add_person(name, images)
            return Response({}, status=200, mimetype='application/json')

        except Exception as e:
            raise e
