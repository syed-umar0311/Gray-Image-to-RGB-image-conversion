import { useState } from "react";
import axios from "axios";

function App() {
  const [grayscaleImage, setGrayscaleImage] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [referenceName, setReferenceName] = useState(null);
  const [showReference, setShowReference] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleGrayscaleUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setGrayscaleImage(file);
      setError("");
    }
  };

  const handleConvert = async () => {
    if (!grayscaleImage) {
      setError("Please upload a grayscale image");
      return;
    }

    setLoading(true);
    setError("");
    setResultImage(null);
    setReferenceName(null);
    setShowReference(false);

    const formData = new FormData();
    formData.append("gray", grayscaleImage);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/colorize",
        formData,
        { responseType: "blob" }
      );

      const imageUrl = URL.createObjectURL(response.data);
      setResultImage(imageUrl);

      // Get reference image name from header
      // After getting response
      const refName =
        response.headers["x-reference-image"] ||
        response.headers["X-Reference-Image"];
      setReferenceName(refName);
    } catch (err) {
      setError("Conversion failed");
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (!resultImage) return;
    const link = document.createElement("a");
    link.href = resultImage;
    link.download = "colorized.png";
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent">
            Gray to Color Converter
          </h1>
          <p className="text-gray-400 text-lg">
            Colorize grayscale images using a reference image
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          {/* Grayscale Image */}
          <div className="bg-gray-800 rounded-lg p-6 mb-6 border border-gray-700">
            <h2 className="text-xl font-semibold mb-2 text-blue-400">
              Upload Grayscale Image
            </h2>
            <input
              type="file"
              accept="image/*"
              onChange={handleGrayscaleUpload}
            />
            {grayscaleImage && (
              <img
                src={URL.createObjectURL(grayscaleImage)}
                className="mt-4 rounded-lg max-h-80 mx-auto"
                alt="gray"
              />
            )}
          </div>

          {/* Convert Button */}
          <div className="text-center mb-6">
            <button
              onClick={handleConvert}
              disabled={loading}
              className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg font-bold hover:scale-105 transition"
            >
              {loading ? "Processing..." : "Convert"}
            </button>
          </div>

          {/* Error */}
          {error && (
            <div className="bg-red-900 text-red-200 p-3 rounded mb-6">
              {error}
            </div>
          )}

          {/* Result */}
          {resultImage && (
            <div className="bg-gray-800 rounded-lg p-6 border border-gray-700">
              <div className="flex justify-between mb-4">
                <h2 className="text-xl font-semibold text-green-400">
                  Colorized Result
                </h2>
                <button
                  onClick={handleDownload}
                  className="bg-green-600 px-4 py-2 rounded hover:bg-green-700"
                >
                  Download
                </button>
              </div>
              <img
                src={resultImage}
                className="rounded-lg mx-auto"
                alt="result"
              />

              {/* Show Reference Image Button */}
              {referenceName && (
                <div className="mt-6 text-center">
                  <button
                    onClick={() => setShowReference(!showReference)}
                    className="px-6 py-3 bg-indigo-600 text-white font-semibold rounded-lg hover:bg-indigo-700 transition"
                  >
                    {showReference
                      ? "Hide Reference Image"
                      : "Show Reference Image"}
                  </button>
                </div>
              )}

              {/* Reference Image */}
              {showReference && referenceName && (
                <div className="mt-6 bg-gray-800 rounded-lg p-6 border border-gray-700">
                  <h3 className="text-xl font-semibold text-yellow-400 mb-4 text-center">
                    Reference Image Used
                  </h3>
                  <img
                    src={`http://127.0.0.1:5000/reference/${referenceName}`}
                    alt="Reference"
                    className="mx-auto rounded-lg max-h-96"
                  />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
