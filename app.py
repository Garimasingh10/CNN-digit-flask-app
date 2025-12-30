from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load CNN model
model = load_model("cnn_model.h5")  # ensure this file is here

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]
    if file.filename == "":
        return "No file selected"

    # Save image
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(img_path)

    # Preprocess image (28x28 grayscale)
    img = Image.open(img_path).convert("L")
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1) / 255.0

    # Predict
    prediction = model.predict(img)
    result = int(np.argmax(prediction))

    # Build URL for showing image
    rel_path = os.path.join("uploads", file.filename)
    image_url = url_for("static", filename=rel_path)

    return render_template(
        "result.html",
        prediction=result,
        image_url=image_url
    )

if __name__ == "__main__":
    print("APP FILE IS RUNNING...")
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
