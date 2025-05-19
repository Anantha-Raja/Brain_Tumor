from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os
import re  # Import the re module for regex

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model = load_model("keras_model.h5", compile=False)

# Load class labels
class_names = open("labels.txt", "r").readlines()

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def preprocess_image(image_path):
    """ Preprocess the image to match the model input """
    size = (224, 224)
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Create a batch of 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    return data

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Preprocess the image
            processed_image = preprocess_image(file_path)

            # Make prediction
            prediction = model.predict(processed_image)
            index = np.argmax(prediction)
            class_name = class_names[index].strip()

            # Remove numeric prefix from the class name
            class_name = re.sub(r"^\d+\s*", "", class_name)

            confidence_score = round(float(prediction[0][index]) * 100, 2)

            return render_template("index.html", image_path=file_path, class_name=class_name, confidence=confidence_score)

    return render_template("index.html", image_path=None, class_name=None, confidence=None)

if __name__ == "__main__":
    app.run(debug=True)
