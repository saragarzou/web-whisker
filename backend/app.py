from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import base64
import io
from PIL import Image

from .nn import NeuralNetwork

model = NeuralNetwork((784, 128, 64, 10))
model.load_weights("model_weights.npz")

app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app)

@app.route('/')
def index():
    """Serve the front-end HTML page."""
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' key"}), 400

    try:
        X = preprocess_image(data["image"])
        prediction = int(model.predict(X)[0])
        return jsonify({"prediction": prediction})
    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

def preprocess_image(base64_str: str) -> np.ndarray:
    """
    Decode base64 image string and preprocess into 784x1 input vector.
    Assumes a 280x280 canvas image that needs to be downscaled.
    """
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return image_array.reshape(-1, 1)

if __name__ == '__main__':
    app.run(debug=True)
