from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import io
import os

app = Flask(__name__)
CORS(app,expose_headers=["X-Reference-Image"])

# Folder containing reference images
REFERENCE_FOLDER = "ref"


def gray2rgb(gray_img, color_img):
    if len(gray_img.shape) == 3:
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)

    gray_rgb = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    ycbcr_color = cv2.cvtColor(color_img, cv2.COLOR_BGR2YCrCb)
    ycbcr_gray = cv2.cvtColor(gray_rgb, cv2.COLOR_BGR2YCrCb)

    ms = ycbcr_color[:, :, 0].astype(np.float32)
    mt = ycbcr_gray[:, :, 0].astype(np.float32)

    d1 = ms.max() - ms.min()
    d2 = mt.max() - mt.min()

    dx1 = (ms * 255) / (255 - d1 + 1e-6)
    dx2 = (mt * 255) / (255 - d2 + 1e-6)

    h, w = dx2.shape
    nimage = np.zeros_like(ycbcr_gray)

    for i in range(h):
        for j in range(w):
            iy = dx2[i, j]
            diff = np.abs(dx1 - iy)
            r, c = np.unravel_index(np.argmin(diff), diff.shape)

            nimage[i, j, 0] = ycbcr_gray[i, j, 0]
            nimage[i, j, 1] = ycbcr_color[r, c, 1]
            nimage[i, j, 2] = ycbcr_color[r, c, 2]

    return cv2.cvtColor(nimage.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def find_reference_image(filename):
    name, _ = os.path.splitext(filename)

    for ext in [".jpg", ".png", ".tif", ".tiff"]:
        ref_path = os.path.join(REFERENCE_FOLDER, name + ext)
        if os.path.exists(ref_path):
            return ref_path

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
            "error": f"No reference image found for {gray_filename}"
        }), 404

    # Read images
    gray_bytes = np.frombuffer(gray_file.read(), np.uint8)
    gray_img = cv2.imdecode(gray_bytes, cv2.IMREAD_UNCHANGED)

    ref_img = cv2.imread(ref_path)

    if gray_img is None or ref_img is None:
        return jsonify({"error": "Invalid image data"}), 400

    result = gray2rgb(gray_img, ref_img)
    _, buffer = cv2.imencode(".png", result)

    response = send_file(
        io.BytesIO(buffer),
        mimetype="image/png"
    )
    # Send reference image name in header
    response.headers["X-Reference-Image"] = os.path.basename(ref_path)
    return response


@app.route("/reference/<filename>")
def get_reference_image(filename):
    path = os.path.join(REFERENCE_FOLDER, filename)
    if not os.path.exists(path):
        return jsonify({"error": "Reference image not found"}), 404
    return send_file(path)


if __name__ == "__main__":
    app.run(debug=True)
