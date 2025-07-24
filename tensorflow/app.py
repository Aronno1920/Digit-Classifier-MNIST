######## Import Library
import io
import os
import re
import numpy as np
import base64

from PIL import Image
import tensorflow as tf
from contextlib import redirect_stdout
from flask import Flask, render_template, request, jsonify
#################
from model_builder import train_model


######## Start Application
app = Flask(__name__)

if os.path.exists("model/mnist_model.keras"):
    model =  tf.keras.models.load_model("model/mnist_model.keras")
#################



######## Load Root - Index page
@app.route("/")
def index():
    return render_template("index.html")
#################



######## Call Predict Function
@app.route("/predict", methods=["POST"])
def predict():
    data_url = request.json["image"]
    image_data = re.sub('^data:image/.+;base64,', '', data_url)

    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("L")
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = (255 - image_array) / 255.0  # invert + normalize

    image_array = np.expand_dims(image_array, axis=0)   # shape: (1, 28, 28)
    image_array = image_array.reshape(1, 784)  

    if os.path.exists("model/mnist_model.keras"):
        model =  tf.keras.models.load_model("model/mnist_model.keras")

        prediction = model.predict(image_array)
        probs = prediction[0].tolist()
        digit = int(np.argmax(probs))

    return jsonify({"digit": digit, "probs": probs})
#################



######## Call Model Training Function
@app.route("/train", methods=["GET", "POST"])
def train():
    logs = ""
    if request.method == "POST":
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            history = train_model()

            print("Training completed!")
            print("Final accuracy:", history['accuracy'][-1])

        logs = buffer.getvalue()

    return render_template("train.html", logs=logs)
#################



if __name__ == "__main__":
    app.run(debug=True)
