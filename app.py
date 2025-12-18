from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("modelbrain.h5")

# Class labels (change order if needed)
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Image preprocessing
def preprocess_image(image):
    image = image.resize((128,128))  # same size as training
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            image = Image.open(file)
            processed_image = preprocess_image(image)

            result = model.predict(processed_image)
            prediction = class_names[np.argmax(result)]
            confidence = round(np.max(result) * 100, 2)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
