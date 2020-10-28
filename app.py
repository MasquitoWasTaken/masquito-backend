import base64
from binascii import a2b_base64
from difflib import ndiff
import json
from flask import Flask, request
from imageai.Prediction.Custom import CustomImagePrediction
from io import BytesIO
import logging
import numpy as np
import os
from PIL import Image
import tempfile


# Contains ImageAI Predictor class
class Predictor:
    def __init__(self, model):
        execution_path = os.getcwd()
        self.prediction = CustomImagePrediction()
        self.prediction.setModelTypeAsResNet()
        self.prediction.setModelPath(os.path.join(
            execution_path, "training_data/models/" + model))
        self.prediction.setJsonPath(os.path.join(
            execution_path, "training_data/json/model_class.json"))
        self.prediction.loadModel(num_objects=3)

    def predict(self, image):
        predictions, probabilities = self.prediction.predictImage(
            image, result_count=3)
        result = {}
        for prediction, probability in zip(predictions, probabilities):
            result[prediction] = "{0:.8f}".format(probability.item())
        return result


# Intentionally global variable
predictor = Predictor("model_ex-067_acc-0.953125.h5")

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
app = Flask(__name__)


# API's only route
@app.route("/mask", methods=["POST"])
def post_mask():
    image_uri = request.args.get("image").replace(" ", "+")
    image_encoded = image_uri.split(",")[1]
    binary_data = a2b_base64(image_encoded)
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(binary_data)
        filename = f.name
    prediction = predictor.predict(filename)
    os.remove(filename)
    return app.response_class(
        response=json.dumps(prediction),
        status=200,
        mimetype="application/json"
    )


if __name__ == "__main__":
    app.run("127.0.0.1", 8080)
