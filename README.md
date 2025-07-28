
# MNIST Digit Classifier

A deep learning web app that classifies handwritten digits (0–9) using the **MNIST dataset**. Built with **TensorFlow** or **PyTorch**, deployed via a lightweight Flask frontend with a live canvas to draw digits.



## 📌 Features

- 🧠 Deep Neural Network with Batch Normalization & Dropout  
- 📊 Accuracy & Loss Visualization  
- ✍️ Web-based Canvas to Draw Digits  
- 🔮 Real-Time Prediction using Trained Model  
- 🧪 98–99% Test Accuracy  




## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/aronno1920/mnist-digit-classifier.git
cd mnist-digit-classifier
```
If you have Python version 3.13 or higher installed, proceed with TensorFlow.
```bash
cd tensorflow
```

If you have Python version 3.12 or lower installed, proceed with torrch.
```bash
cd torch
```


2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Train the model:**
```bash
Go to "Train Model" and Click Start Training button
```

5. **Run the Flask app:**
```bash
cd app
python app.py
```



## 🧠 Model Summary

- Input: 28x28 grayscale images (flattened or CNN input)
- Hidden Layers: 2–3 dense layers with ReLU activation
- Batch Normalization and Dropout for regularization
- Output: 10-class Softmax (digits 0–9)



## 📊 Training & Evaluation

- **Dataset:** MNIST (from `tensorflow.keras.datasets`)
- **Loss Function:** `categorical_crossentropy`
- **Optimizer:** `Adam(learning_rate=0.001)`
- **Epochs:** `Best epochs number using EarlyStopping & ModelCheckpoint`
- **Final Accuracy:** `~98% on test set`


## 🌟 Support

If you find this project helpful, please consider ⭐ starring the repo and sharing it!
