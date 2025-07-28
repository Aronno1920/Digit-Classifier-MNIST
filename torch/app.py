######## Import Libraries
import os
import io
import re
import base64
import numpy as np
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template
from contextlib import redirect_stdout

import torch
from torchvision import transforms
from model_builder import train_model, MNISTModel

######## Custom Model_Config
from model_config import ModelConfig



######## Start Flask Application
app = Flask(__name__)

######## Load the Trained Model
model_config = ModelConfig()
model_path = model_config.get_model_path()

model = MNISTModel()

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print("PyTorch model loaded successfully.")
else:
    print("Model file not found. Please train the model first.")

######## Route: Index Page
@app.route('/')
def index():
    return render_template('index.html')

######## Route: Predict from Base64 Image
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data.get('image')

    if not img_data:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        # Decode base64 image
        img_str = re.search(r'base64,(.*)', img_data).group(1)
        image_bytes = base64.b64decode(img_str)
        image = Image.open(io.BytesIO(image_bytes)).convert('L')

        # Invert: black background, white digit (like MNIST)
        image = ImageOps.invert(image)

        # Crop the digit (remove whitespace)
        np_img = np.array(image)
        coords = np.argwhere(np_img < 255)
        if coords.size == 0:
            return jsonify({'prediction': 'blank'})
        
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        image = image.crop((x0, y0, x1, y1))

        # Resize to 20x20 and center in 28x28
        image = image.resize((20, 20), Image.Resampling.LANCZOS)
        new_image = Image.new('L', (28, 28), 0)
        new_image.paste(image, ((28 - 20) // 2, (28 - 20) // 2))

        # Optional: Save for debugging
        new_image.save("debug_final_input.png")

        # Normalize and convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        input_tensor = transform(new_image).unsqueeze(0)  # Shape: [1, 1, 28, 28]

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.argmax(dim=1).item()

        return jsonify({'prediction': prediction})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

######## Route: Train the Model
@app.route("/train", methods=["GET", "POST"])
def train():
    logs = ""
    if request.method == "POST":
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            history = train_model()
            
            print("Training completed!")
            print(f"Final Accuracy: {history['accuracy'][-1]:.4f}")
        logs = buffer.getvalue()
    return render_template("train.html", logs=logs)

######## Image Preprocessing Helper (if needed elsewhere)
def preprocess_canvas_image(img):    
    if img.mode != 'L':
        img = img.convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28))

    # Normalize to [0, 1] and flatten
    img_arr = np.array(img) / 255.0
    img_arr = img_arr.reshape(1, 784).astype(np.float32)
    tensor = torch.from_numpy(img_arr)
    return tensor

######## Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
