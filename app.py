import os
import random
import cv2
import time
from flask import Flask, render_template, jsonify, request, send_from_directory

app = Flask(__name__)

# Dataset Paths
BASE_PATH = r"/Misc"
LR_PATH = os.path.join(BASE_PATH, "RESISC45_LR")
HR_PATH = os.path.join(BASE_PATH, "RESISC45_HR")
UPSCALED_PATH = os.path.join(BASE_PATH, "RESISC45_Upscaled")
UPLOAD_FOLDER = os.path.join(BASE_PATH, "Uploads")

# Models configuration
MODELS = {
    "edsr_x2": {"path": "EDSR_x2.pb", "name": "edsr", "scale": 2},
    "espcn_x4": {"path": "ESPCN_x4.pb", "name": "espcn", "scale": 4},
    "fsrcnn_x3": {"path": "FSRCNN_x3.pb", "name": "fsrcnn", "scale": 3},
    "lapsrn_x8": {"path": "LapSRN_x8.pb", "name": "lapsrn", "scale": 8}
}

# Ensure output folders exist
os.makedirs(UPSCALED_PATH, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Function to get Super Resolution model
def get_sr_model(model_key="edsr_x2"):
    model_info = MODELS.get(model_key, MODELS["edsr_x2"])  # Default to EDSR if invalid

    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    model_path = os.path.join(BASE_PATH, "Models", model_info["path"])

    # Check if model exists, if not, use default EDSR
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Using default EDSR model.")
        model_path = os.path.join(BASE_PATH, "Models", MODELS["edsr_x2"]["path"])
        model_info = MODELS["edsr_x2"]

    sr.readModel(model_path)
    sr.setModel(model_info["name"], model_info["scale"])

    return sr, model_info["scale"]


def get_random_image():
    """Selects a random image from the LR dataset."""
    images = os.listdir(LR_PATH)
    if not images:
        return None, None, None
    img_name = random.choice(images)
    return os.path.join(LR_PATH, img_name), os.path.join(HR_PATH, img_name), img_name


def upscale_image(image_path, output_folder, model_key="edsr_x2"):
    """Upscales an image and saves it in the given output folder."""
    os.makedirs(output_folder, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    h, w, c = img.shape  # Get original dimensions

    # Ensure 3-channel BGR image
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Get model for upscaling
    sr, scale_factor = get_sr_model(model_key)
    upscaled_img = sr.upsample(img)

    # Generate output filename with model info
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    upscaled_path = os.path.join(output_folder, f"{name}_{model_key}{ext}")

    cv2.imwrite(upscaled_path, upscaled_img)
    return upscaled_path, (h, w), scale_factor


@app.route("/")
def index():
    """Renders the main page."""
    return render_template("index.html")


@app.route("/get_images")
def get_images():
    """Fetches a random image and upscales it."""
    model_key = request.args.get("model", "edsr_x2")

    lr_image_path, hr_image_path, img_name = get_random_image()
    if not lr_image_path or not hr_image_path:
        return jsonify({"error": "No images found in dataset!"})

    upscaled_image_path, dims, scale = upscale_image(lr_image_path, UPSCALED_PATH, model_key)
    if not upscaled_image_path:
        return jsonify({"error": "Error upscaling image!"})

    return jsonify({
        "lr_img": f"/images/lr/{img_name}",
        "upscaled_img": f"/images/upscaled/{os.path.basename(upscaled_image_path)}",
        "hr_img": f"/images/hr/{img_name}",
        "scale_factor": scale
    })


@app.route("/upload", methods=["POST"])
def upload():
    """Handles image uploads, upscales, and returns paths."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    model_key = request.form.get("model", "edsr_x2")

    if file.filename == "":
        return jsonify({"error": "No selected file"})

    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)  # Save the uploaded file

    start_time = time.time()
    upscaled_path, (h, w), scale_factor = upscale_image(filepath, UPLOAD_FOLDER, model_key)
    processing_time = time.time() - start_time

    if not upscaled_path:
        return jsonify({"error": "Error processing image!"})

    return jsonify({
        "original_img": f"/images/upload/{filename}",
        "upscaled_img": f"/images/upload/{os.path.basename(upscaled_path)}",
        "original_size": f"{w} x {h}",
        "upscaled_size": f"{w * scale_factor} x {h * scale_factor}",
        "processing_time": f"{processing_time:.2f} seconds",
        "scale_factor": scale_factor,
        "model_used": model_key
    })


@app.route("/images/<dataset>/<filename>")
def serve_image(dataset, filename):
    """Serves images from different folders."""
    folder_map = {
        "lr": LR_PATH,
        "hr": HR_PATH,
        "upscaled": UPSCALED_PATH,
        "upload": UPLOAD_FOLDER
    }
    folder = folder_map.get(dataset)
    if not folder:
        return "Invalid dataset", 400

    return send_from_directory(folder, filename)


if __name__ == "__main__":
    app.run(debug=True)