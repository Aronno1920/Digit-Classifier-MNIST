import io
import numpy as np
import base64

from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
model = load_model('model/mnist_model.keras')

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    image = image.resize((28, 28))
    image = np.array(image) / 255.0
    image = image.reshape(1, 28 * 28)
    return image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()
        image_data = data['image'].split(',')[1]  # Remove header
        image_bytes = base64.b64decode(image_data)
        input_data = preprocess_image(image_bytes)
        pred = model.predict(input_data, verbose=0)
        prediction = int(np.argmax(pred))
        return jsonify({'prediction': prediction})

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
