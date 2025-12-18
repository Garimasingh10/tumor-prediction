# Brain Tumor Detection Web App 🧠

This project is a deep‑learning based web application that detects brain tumors from MRI images using a Convolutional Neural Network (CNN) and a Flask backend. It classifies MRI scans into four categories: **Glioma**, **Meningioma**, **Pituitary**, and **No Tumor**, and shows the predicted class with a confidence score in a modern web UI. [web:61][web:65][web:68]

## Features

- Upload MRI brain images through a web interface  
- Preprocessing (resize + normalization) before prediction  
- CNN model loaded from `modelbrain.h5` using TensorFlow/Keras  
- Prediction of tumor type with confidence percentage  
- Flask backend + HTML/CSS frontend for local deployment [web:61][web:69]

## Tech Stack

- **Backend:** Python, Flask  
- **Deep Learning:** TensorFlow / Keras, NumPy, Pandas, scikit‑learn  
- **Frontend:** HTML, CSS, Jinja2 templates [web:61][web:62][web:69]

## How to Run Locally

git clone https://github.com/Garimasingh10/tumor-prediction.git
cd tumor-prediction

python -m venv myenv
myenv\Scripts\activate # on Windows

pip install -r requirements.txt
python app.py


