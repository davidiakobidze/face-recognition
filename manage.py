from flask import Flask
from flask_restful import Api

from resources.add_person import AddPerson
from resources.detect import Recognise
from resources.extract_embedding import ExtractEmbedding
from resources.train import Train

app = Flask(__name__)
api = Api(app)

api.add_resource(Train, '/train')
api.add_resource(ExtractEmbedding, '/extract-embedding')
api.add_resource(Recognise, '/recognise')
api.add_resource(AddPerson, '/add-person')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)  # for cloud servers port should not be added!
