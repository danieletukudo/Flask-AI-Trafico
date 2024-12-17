from flask import Flask, request, jsonify, url_for, send_from_directory
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Paths
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Replace with your custom model path if needed


@app.route("/", methods=["GET"])
def home():
    return "Welcome to the Flask YOLOv8 Image Detection API"


@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Save input file securely
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        # Define output path
        output_filename = f"output_{filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # Perform object detection
        results = model.predict(source=input_path, save=True, project=OUTPUT_FOLDER, name="output", exist_ok=True)

        # YOLO saves output images in the output folder
        detected_image_path = os.path.join(OUTPUT_FOLDER, "output", os.path.basename(input_path))

        # Rename the processed image for simplicity
        os.rename(detected_image_path, output_path)

        # Generate a public link to the processed image
        file_url = url_for("static_files", filename=output_filename, _external=True)

        return jsonify({"message": "Image detection complete", "output_url": file_url})


@app.route("/static/<path:filename>", methods=["GET"])
def static_files(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
