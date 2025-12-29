from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
import cv2
import numpy as np
import io
import os
import json
import base64

app = Flask(__name__)
CORS(app, expose_headers=["X-Reference-Image", "X-Histogram-Data"])

# Folder containing reference images
REFERENCE_FOLDER = "ref"


def gray2rgb(gray_img, color_img, alpha=0.4):
    """
    alpha: controls strength of reference color
           0.0 = pure grayscale
           0.2 = very subtle color hint
           0.5 = moderate color
           1.0 = full reference color
    """
    if len(gray_img.shape) == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    color_img = cv2.resize(color_img, (gray_img.shape[1], gray_img.shape[0]), interpolation=cv2.INTER_AREA)

    gray_ycc = cv2.cvtColor(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2YCrCb)
    color_ycc = cv2.cvtColor(color_img, cv2.COLOR_BGR2YCrCb)

    Yg = gray_ycc[:, :, 0].astype(np.float32)
    Cr_ref = color_ycc[:, :, 1].astype(np.float32)
    Cb_ref = color_ycc[:, :, 2].astype(np.float32)

    # Smooth reference chroma to avoid noise
    Cr_smooth = cv2.GaussianBlur(Cr_ref, (3,3), 0)
    Cb_smooth = cv2.GaussianBlur(Cb_ref, (3,3), 0)

    # Blend chroma: smaller alpha → subtle color
    Cr_out = ((1 - alpha) * 128 + alpha * Cr_smooth).astype(np.uint8)
    Cb_out = ((1 - alpha) * 128 + alpha * Cb_smooth).astype(np.uint8)

    out_ycc = np.zeros_like(color_ycc)
    out_ycc[:, :, 0] = Yg
    out_ycc[:, :, 1] = Cr_out
    out_ycc[:, :, 2] = Cb_out

    return cv2.cvtColor(out_ycc, cv2.COLOR_YCrCb2BGR)


def calculate_histogram(image, is_normalized=True):
    """Calculate histogram for an image (grayscale or color)"""
    if len(image.shape) == 2:  # Grayscale image
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.flatten()
        if is_normalized and hist.sum() > 0:
            hist = (hist / hist.sum() * 100).tolist()  # Convert to percentages
        else:
            hist = hist.tolist()
        return {"gray": hist}
    
    elif len(image.shape) == 3:  # Color image (BGR format in OpenCV)
        # Split into channels
        channels = cv2.split(image)
        color_names = ['blue', 'green', 'red']  # OpenCV uses BGR
        
        histograms = {}
        for i, channel in enumerate(channels):
            hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
            hist = hist.flatten()
            if is_normalized and hist.sum() > 0:
                hist = (hist / hist.sum() * 100).tolist()  # Convert to percentages
            else:
                hist = hist.tolist()
            histograms[color_names[i]] = hist
        
        return histograms
    
    return None


def find_reference_image(filename):
    # Remove extension from uploaded filename
    name, _ = os.path.splitext(filename)
    
    # First, try to find exact match with same name (case-insensitive)
    for ext in [".jpg", ".png", ".tif", ".tiff", ".jpeg", ".JPG", ".PNG", ".TIF", ".TIFF", ".JPEG"]:
        ref_path = os.path.join(REFERENCE_FOLDER, name + ext)
        if os.path.exists(ref_path):
            return ref_path
    
    # If no exact match found, return None to trigger popup
    return None


@app.route("/colorize", methods=["POST"])
def colorize():
    if "gray" not in request.files:
        return jsonify({"error": "Grayscale image required"}), 400

    gray_file = request.files["gray"]
    gray_filename = gray_file.filename

    # Find matching reference image
    ref_path = find_reference_image(gray_filename)

    if ref_path is None:
        return jsonify({
            "error": "NO_REFERENCE_FOUND",
            "message": f"No reference image found for '{gray_filename}'. Please upload a reference image."
        }), 404

    try:
        # Read images
        gray_file.seek(0)  # Reset file pointer
        gray_bytes = np.frombuffer(gray_file.read(), np.uint8)
        gray_img = cv2.imdecode(gray_bytes, cv2.IMREAD_UNCHANGED)

        ref_img = cv2.imread(ref_path)

        if gray_img is None:
            return jsonify({"error": "Invalid grayscale image data"}), 400
        if ref_img is None:
            return jsonify({"error": "Invalid reference image data"}), 400

        # Calculate grayscale histogram (raw values, not normalized)
        gray_histogram = calculate_histogram(gray_img, is_normalized=False)

        # Colorize the image
        result = gray2rgb(gray_img, ref_img)

        # Calculate color histogram (raw values, not normalized)
        color_histogram = calculate_histogram(result, is_normalized=False)

        # Encode result image
        _, buffer = cv2.imencode(".png", result)
        
        # Prepare histogram data
        histogram_data = {
            "input": gray_histogram,
            "output": color_histogram
        }

        # Create response
        response = send_file(
            io.BytesIO(buffer),
            mimetype="image/png"
        )
        
        # Send reference image name and histogram data in headers
        response.headers["X-Reference-Image"] = os.path.basename(ref_path)
        response.headers["X-Histogram-Data"] = json.dumps(histogram_data)
        
        return response

    except Exception as e:
        print(f"Error during colorization: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/upload_reference", methods=["POST"])
def upload_reference():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    if "filename" not in request.form:
        return jsonify({"error": "No filename provided"}), 400
    
    file = request.files["image"]
    filename = request.form["filename"]
    
    # Extract base name without extension
    name, _ = os.path.splitext(filename)
    
    # Save with same name as input image but with .png extension
    ref_filename = name + ".png"
    file_path = os.path.join(REFERENCE_FOLDER, ref_filename)
    
    try:
        file.save(file_path)
        return jsonify({
            "success": True,
            "message": f"Reference image '{ref_filename}' uploaded successfully",
            "filename": ref_filename
        })
    except Exception as e:
        return jsonify({"error": f"Failed to save image: {str(e)}"}), 500


@app.route("/reference/<filename>")
def get_reference_image(filename):
    path = os.path.join(REFERENCE_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Reference image not found"}), 404
    return send_file(path)


@app.route("/check_reference/<filename>")
def check_reference(filename):
    ref_path = find_reference_image(filename)
    exists = ref_path is not None
    return jsonify({
        "exists": exists,
        "reference_name": os.path.basename(ref_path) if exists else None
    })


@app.route("/health")
def health_check():
    return jsonify({"status": "healthy", "service": "image-colorization-api"})


if __name__ == "__main__":
    # Create reference folder if it doesn't exist
    if not os.path.exists(REFERENCE_FOLDER):
        os.makedirs(REFERENCE_FOLDER)
        print(f"Created reference folder: {REFERENCE_FOLDER}")
    
    print("Server starting...")
    print(f"Reference images should be placed in: {os.path.abspath(REFERENCE_FOLDER)}")
    print("Note: System now requires exact filename match for reference images")
    app.run(debug=True, port=5000)