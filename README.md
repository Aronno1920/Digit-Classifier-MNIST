# 🧠 Digit Classifier – MNIST (Flask + TensorFlow)

This project is a **handwritten digit recognizer** powered by a trained **TensorFlow** model on the **MNIST dataset**. It features a modern HTML5 canvas frontend and a **Flask backend** to serve predictions from a `.keras` model.


---


## 📦 Project Structure
mnist-digit-classifier/
├── app/
│ ├── static/
│ │ ├── script.js
│ │ └── style.css
│ ├── templates/
│ │ ├── index.html
│ │ └── train.html
├── model/
│ └── mnist_model.h5
├── notebooks/
│ └── training.ipynb
├── screenshots/
│ └── demo.gif / accuracy_plot.png
├── README.md
├── app.py
└── requirements.txt


---

## 🛠️ Features

- 🎨 Draw a digit (0–9) on canvas
- ⚙️ Preprocess and resize to 28×28
- 🧠 Predict using TensorFlow model
- 📊 Displays prediction + confidence
- 🌐 Flask API backend (`/predict`)

---

## 💡 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/aronno1920/Digit-Classifier-MNIST.git
cd Digit-Classifier-MNIST
