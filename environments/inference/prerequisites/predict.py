import os
import flask

from movie_detector.ml.neural_network import GenreClassifier
from sentence_transformers import SentenceTransformer
from torch import load, tensor, float32

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")


class GenrePredictionService():
    model = None
    @classmethod
    def get_model(cls):
        if cls.model is None:
            cls.model = GenreClassifier()
            map_path = os.path.join(model_path, 'model.pth')
            cls.model.load_state_dict(load(map_path))
            cls.model.eval()
        return cls.model

    @classmethod
    def predict(cls, req: dict, threshold: float = 0.5):
        if 'titles' not in req.keys():
            print("No titles provided")
            return None
        try:
            print('Loading model...')
            model = cls.get_model()
        except Exception as e:
            print(f'Model retrieval failed: {e}')
            return None
        print('Loading sentence transformers...')
        hgf_model = SentenceTransformer('all-MiniLM-L6-v2')
        titles = req['titles']
        genres = req['genres'] if 'genres' in req.keys() else None
        try:
            print('Embedding...')
            embedded_titles = hgf_model.encode(titles)
            tensors = tensor(embedded_titles, dtype=float32)
            print('Predicting...')
            predictions = model(tensors)
            predictions = {t: p.tolist() for t, p in zip(titles, predictions)}
            print('Predictions done.')
            return predictions
        except Exception as e:
            print(f'Prediction failed: {e}')
            return None


app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    health = GenrePredictionService.get_model() is not None
    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    if flask.request.content_type == "application/json":
        req = flask.request.json
        try:
            predictions:dict = GenrePredictionService.predict(req)
            return flask.jsonify(predictions)
        except KeyError as e:
            print("KeyError", e)
            return flask.Response(response="Bad Request", status=400, mimetype="text/plain")
    else:
        print('Format not supported')
        return flask.Response(
            response="This endpoint only support json.", status=415, mimetype="text/plain"
        )
