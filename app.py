from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import importlib
import os
import sys

app = Flask(__name__)
CORS(app)

image = "detec.png"
whitelist = {"png", "jpg", "jpeg", "webp"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in whitelist:
        return jsonify({"error": "Invalid file type"}), 400

    try:
        file.save(image)

        if "predict" in sys.modules:
            import predict
            importlib.reload(predict)
        else:
            import predict

        return jsonify({
            "result": predict.pred,
            "confidence": round(predict.conf * 100, 2),
            "reason": predict.reason
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8080, debug = False)