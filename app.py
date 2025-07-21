######## Import Library
import io
import numpy as np
import base64

import tensorflow as tf
import re
from PIL import Image
from flask import Flask, render_template, request, jsonify
######## Import Library

######## Start Application
app = Flask(__name__)

######## Load Model
model =  tf.keras.models.load_model('model/mnist_model.keras')

######## Load Root - Index page
@app.route("/")
def index():
    return render_template("index.html")

######## Load Root - Index page
@app.route("/predict", methods=["POST"])
def predict():
    data_url = request.json["image"]
    image_data = re.sub('^data:image/.+;base64,', '', data_url)
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("L")
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = (255 - image_array) / 255.0  # Invert + normalize

    # Convert from (28, 28) to (1, 28, 28)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=0)

    # Now shape is (1, 1, 28, 28), matching model input (None, 1, 28, 28)
    prediction = model.predict(image_array)
    probs = prediction[0].tolist()
    digit = int(np.argmax(probs))

    return jsonify({"digit": digit, "probs": probs})



if __name__ == "__main__":
    app.run(debug=True)
