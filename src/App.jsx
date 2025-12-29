import { useState, useEffect } from "react";
import axios from "axios";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Line } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function App() {
  const [grayscaleImage, setGrayscaleImage] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [referenceName, setReferenceName] = useState(null);
  const [showReference, setShowReference] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [histogramData, setHistogramData] = useState(null);
  const [showHistograms, setShowHistograms] = useState(false);
  const [yAxisMax, setYAxisMax] = useState(25000);
  const [showReferencePopup, setShowReferencePopup] = useState(false);
  const [newReferenceImage, setNewReferenceImage] = useState(null);
  const [uploadingReference, setUploadingReference] = useState(false);
  const [uploadError, setUploadError] = useState("");

  // Calculate appropriate Y-axis max based on data
  const calculateYAxisMax = (data, type) => {
    if (!data) return 25000;
    
    let maxValue = 0;
    
    if (type === 'input' && data.gray) {
      maxValue = Math.max(...data.gray);
    } else if (type === 'output') {
      const redMax = data.red ? Math.max(...data.red) : 0;
      const greenMax = data.green ? Math.max(...data.green) : 0;
      const blueMax = data.blue ? Math.max(...data.blue) : 0;
      maxValue = Math.max(redMax, greenMax, blueMax);
    }
    
    // Round up to nearest 5000
    return Math.ceil(maxValue / 5000) * 5000;
  };

  // Prepare histogram data for Chart.js
  const prepareHistogramData = (data, type) => {
    const labels = Array.from({ length: 256 }, (_, i) => i);
    
    if (type === 'input') {
      return {
        labels,
        datasets: [
          {
            label: 'Grayscale Intensity',
            data: data.gray || [],
            borderColor: 'rgb(156, 163, 175)',
            backgroundColor: 'rgba(156, 163, 175, 0.3)',
            tension: 0.1,
            fill: true,
            borderWidth: 2
          }
        ]
      };
    } else if (type === 'output') {
      return {
        labels,
        datasets: [
          {
            label: 'Red Channel',
            data: data.red || [],
            borderColor: 'rgb(239, 68, 68)',
            backgroundColor: 'rgba(239, 68, 68, 0.3)',
            tension: 0.1,
            borderWidth: 2
          },
          {
            label: 'Green Channel',
            data: data.green || [],
            borderColor: 'rgb(34, 197, 94)',
            backgroundColor: 'rgba(34, 197, 94, 0.3)',
            tension: 0.1,
            borderWidth: 2
          },
          {
            label: 'Blue Channel',
            data: data.blue || [],
            borderColor: 'rgb(59, 130, 246)',
            backgroundColor: 'rgba(59, 130, 246, 0.3)',
            tension: 0.1,
            borderWidth: 2
          }
        ]
      };
    }
    return null;
  };

  // Chart options with custom Y-axis scale
  const chartOptions = (title, type) => {
    const yMax = type === 'input' 
      ? calculateYAxisMax(histogramData?.input, 'input')
      : calculateYAxisMax(histogramData?.output, 'output');
    
    const yStep = Math.ceil(yMax / 5); // Divide into 5 steps
    const yTicks = [0, yStep, yStep*2, yStep*3, yStep*4, yMax];
    
    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: '#ccc',
            font: {
              size: 12
            }
          }
        },
        title: {
          display: true,
          text: title,
          color: '#fff',
          font: {
            size: 16,
            weight: 'bold'
          }
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
          titleColor: '#fff',
          bodyColor: '#fff',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Pixel Intensity (0-255)',
            color: '#ccc',
            font: {
              size: 12,
              weight: 'bold'
            }
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.1)',
            drawBorder: false
          },
          ticks: {
            color: '#ccc',
            maxTicksLimit: 10,
            font: {
              size: 11
            }
          }
        },
        y: {
          title: {
            display: true,
            text: 'Frequency',
            color: '#ccc',
            font: {
              size: 12,
              weight: 'bold'
            }
          },
          beginAtZero: true,
          max: yMax,
          grid: {
            color: 'rgba(255, 255, 255, 0.1)',
            drawBorder: false
          },
          ticks: {
            color: '#ccc',
            stepSize: yStep,
            callback: function(value) {
              // Format large numbers with commas
              return value.toLocaleString();
            },
            font: {
              size: 11
            }
          }
        }
      },
      interaction: {
        intersect: false,
        mode: 'nearest'
      },
      elements: {
        point: {
          radius: 0, // Hide points for cleaner look
          hoverRadius: 5
        }
      }
    };
  };

  // Get statistics for display
  const getHistogramStats = (data, type) => {
    if (!data) return null;
    
    if (type === 'input' && data.gray) {
      const max = Math.max(...data.gray);
      const avg = data.gray.reduce((a, b) => a + b, 0) / data.gray.length;
      const nonZero = data.gray.filter(v => v > 0).length;
      
      return {
        maxValue: max.toLocaleString(),
        average: Math.round(avg).toLocaleString(),
        activePixels: nonZero
      };
    } else if (type === 'output') {
      const redMax = data.red ? Math.max(...data.red) : 0;
      const greenMax = data.green ? Math.max(...data.green) : 0;
      const blueMax = data.blue ? Math.max(...data.blue) : 0;
      
      return {
        redMax: redMax.toLocaleString(),
        greenMax: greenMax.toLocaleString(),
        blueMax: blueMax.toLocaleString(),
        overallMax: Math.max(redMax, greenMax, blueMax).toLocaleString()
      };
    }
    
    return null;
  };

  const handleGrayscaleUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setGrayscaleImage(file);
      setHistogramData(null);
      setShowHistograms(false);
      setError("");
      setShowReferencePopup(false);
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
    setHistogramData(null);
    setShowHistograms(false);
    setShowReferencePopup(false);
    setUploadError("");

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
      const refName =
        response.headers["x-reference-image"] ||
        response.headers["X-Reference-Image"];
      setReferenceName(refName);

      // Get histogram data from header
      const histData = response.headers["x-histogram-data"] ||
                      response.headers["X-Histogram-Data"];
      
      if (histData) {
        try {
          const parsedData = JSON.parse(histData);
          setHistogramData(parsedData);
        } catch (parseError) {
          console.error("Error parsing histogram data:", parseError);
        }
      }
    } catch (err) {
      if (err.response && err.response.data) {
        // Handle blob response for errors
        const errorBlob = err.response.data;
        const errorText = await errorBlob.text();
        try {
          const errorData = JSON.parse(errorText);
          if (errorData.error === "NO_REFERENCE_FOUND") {
            // Show popup for missing reference image
            setShowReferencePopup(true);
            setError("");
          } else {
            setError(errorData.message || "Conversion failed");
          }
        } catch (parseError) {
          setError("Conversion failed");
        }
      } else {
        setError("Conversion failed");
        console.error(err);
      }
    } finally {
      setLoading(false);
    }
  };

  const handleUploadReference = async () => {
    if (!newReferenceImage) {
      setUploadError("Please select a reference image");
      return;
    }

    setUploadingReference(true);
    setUploadError("");

    const formData = new FormData();
    formData.append("image", newReferenceImage);
    formData.append("filename", grayscaleImage.name);

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/upload_reference",
        formData
      );

      // Close popup and retry conversion
      setShowReferencePopup(false);
      setNewReferenceImage(null);
      // Retry the conversion
      setTimeout(() => {
        handleConvert();
      }, 500);
    } catch (err) {
      setUploadError(err.response?.data?.message || "Failed to upload reference image");
    } finally {
      setUploadingReference(false);
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

  const handleSkipReference = () => {
    setShowReferencePopup(false);
    setError("Cannot proceed without a reference image. Please upload one.");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      {/* Reference Image Popup */}
      {showReferencePopup && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-800 rounded-xl p-6 max-w-md w-full border border-gray-700 shadow-2xl">
            <h2 className="text-2xl font-bold mb-4 text-yellow-400 flex items-center gap-2">
              <span className="inline-block p-2 bg-yellow-900 rounded-lg">
                ⚠️
              </span>
              Reference Image Required
            </h2>
            <p className="text-gray-300 mb-6">
              No reference image found for <span className="font-bold">{grayscaleImage?.name}</span>. 
              Please upload a reference image to continue with colorization.
            </p>
            
            <div className="mb-6">
              <label className="block text-gray-400 mb-2 font-semibold">
                Upload Reference Image
              </label>
              <input
                type="file"
                accept="image/*"
                onChange={(e) => setNewReferenceImage(e.target.files[0])}
                className="block w-full text-sm text-gray-400
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-lg file:border-0
                  file:text-sm file:font-bold
                  file:bg-gradient-to-r file:from-blue-600 file:to-blue-800 
                  file:text-white hover:file:from-blue-700 hover:file:to-blue-900
                  cursor-pointer"
              />
              {newReferenceImage && (
                <p className="text-green-400 mt-2 text-sm">
                  ✓ Selected: {newReferenceImage.name}
                </p>
              )}
            </div>

            {uploadError && (
              <div className="bg-gradient-to-r from-red-900 to-red-800 text-red-100 p-3 rounded-lg mb-4 text-sm">
                ⚠️ {uploadError}
              </div>
            )}

            <div className="flex gap-4">
              <button
                onClick={handleUploadReference}
                disabled={uploadingReference || !newReferenceImage}
                className="flex-1 px-4 py-3 bg-gradient-to-r from-green-600 to-emerald-700 rounded-lg font-bold hover:scale-105 transition-all hover:shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {uploadingReference ? (
                  <>
                    <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Uploading...
                  </>
                ) : (
                  "Upload & Continue"
                )}
              </button>
              <button
                onClick={handleSkipReference}
                className="px-4 py-3 bg-gradient-to-r from-gray-700 to-gray-800 rounded-lg font-bold hover:scale-105 transition-all hover:shadow-lg border border-gray-600"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="container mx-auto px-4 py-8">
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-600 bg-clip-text text-transparent">
            Gray to Color Converter
          </h1>
          <p className="text-gray-400 text-lg">
            Colorize grayscale images using a reference image
          </p>
        </div>

        <div className="max-w-7xl mx-auto">
          {/* Grayscale Image */}
          <div className="bg-gray-800 rounded-lg p-6 mb-6 border border-gray-700 shadow-xl">
            <h2 className="text-2xl font-bold mb-4 text-blue-400 flex items-center gap-2">
              <span className="inline-block p-2 bg-blue-900 rounded-lg">
                📤
              </span>
              Upload Grayscale Image
            </h2>
            <input
              type="file"
              accept="image/*"
              onChange={handleGrayscaleUpload}
              className="block w-full text-sm text-gray-400
                file:mr-4 file:py-3 file:px-6
                file:rounded-lg file:border-0
                file:text-sm file:font-bold
                file:bg-gradient-to-r file:from-blue-600 file:to-blue-800 
                file:text-white hover:file:from-blue-700 hover:file:to-blue-900
                cursor-pointer"
            />
            {grayscaleImage && (
              <div className="mt-6 p-4 bg-gray-900 rounded-lg">
                <p className="text-green-400 mb-2 font-semibold">
                  ✓ Image loaded: {grayscaleImage.name}
                </p>
                <img
                  src={URL.createObjectURL(grayscaleImage)}
                  className="rounded-lg max-h-80 mx-auto border-2 border-gray-600"
                  alt="Uploaded grayscale"
                />
              </div>
            )}
          </div>

          {/* Convert Button */}
          <div className="text-center mb-8">
            <button
              onClick={handleConvert}
              disabled={loading}
              className="px-10 py-5 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 rounded-xl font-bold text-lg hover:scale-105 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing Image...
                </span>
              ) : (
                "✨ Convert to Color"
              )}
            </button>
          </div>

          {/* Error */}
          {error && (
            <div className="bg-gradient-to-r from-red-900 to-red-800 text-red-100 p-4 rounded-lg mb-6 text-center font-semibold border border-red-700">
              ⚠️ {error}
            </div>
          )}

          {/* Result Section */}
          {resultImage && (
            <div className="space-y-8">
              {/* Colorized Result */}
              <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-xl">
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
                  <h2 className="text-2xl font-bold text-green-400 flex items-center gap-2">
                    <span className="inline-block p-2 bg-green-900 rounded-lg">
                      🎨
                    </span>
                    Colorized Result
                  </h2>
                  <button
                    onClick={handleDownload}
                    className="px-6 py-3 bg-gradient-to-r from-green-600 to-emerald-700 rounded-lg font-bold hover:scale-105 transition-all hover:shadow-lg flex items-center gap-2"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                    </svg>
                    Download Image
                  </button>
                </div>
                <div className="p-4 bg-gray-900 rounded-lg">
                  <img
                    src={resultImage}
                    className="rounded-lg mx-auto max-h-96 border-2 border-gray-600 shadow-lg"
                    alt="Colorized result"
                  />
                </div>
              </div>

              {/* Histogram Controls */}
              {histogramData && (
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-xl">
                  <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
                    <h2 className="text-2xl font-bold text-yellow-400 flex items-center gap-2">
                      <span className="inline-block p-2 bg-yellow-900 rounded-lg">
                        📊
                      </span>
                      Image Histograms
                    </h2>
                    <button
                      onClick={() => setShowHistograms(!showHistograms)}
                      className="px-6 py-3 bg-gradient-to-r from-indigo-600 to-purple-700 rounded-lg font-bold hover:scale-105 transition-all hover:shadow-lg flex items-center gap-2"
                    >
                      {showHistograms ? (
                        <>
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 15l7-7 7 7"></path>
                          </svg>
                          Hide Histograms
                        </>
                      ) : (
                        <>
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                          </svg>
                          Show Histograms
                        </>
                      )}
                    </button>
                  </div>

                  {showHistograms && (
                    <div className="space-y-8">
                      {/* Input Grayscale Histogram */}
                      <div className="bg-gray-900 p-6 rounded-xl border border-gray-700">
                        <div className="flex flex-col lg:flex-row gap-6">
                          <div className="lg:w-3/4">
                            <h3 className="text-xl font-bold mb-4 text-blue-400 flex items-center gap-2">
                              <span className="p-2 bg-blue-900 rounded-lg">
                                ⚫
                              </span>
                              Input Grayscale Histogram
                            </h3>
                            <div className="h-80">
                              <Line 
                                data={prepareHistogramData(histogramData.input, 'input')}
                                options={chartOptions('Grayscale Intensity Distribution', 'input')}
                              />
                            </div>
                          </div>
                          <div className="lg:w-1/4">
                            <div className="bg-gray-800 p-4 rounded-lg h-full">
                              <h4 className="font-bold text-lg mb-3 text-gray-300">Statistics</h4>
                              {getHistogramStats(histogramData.input, 'input') && (
                                <div className="space-y-2">
                                  <div className="flex justify-between">
                                    <span className="text-gray-400">Max Frequency:</span>
                                    <span className="font-bold text-white">
                                      {getHistogramStats(histogramData.input, 'input')?.maxValue}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-gray-400">Average Frequency:</span>
                                    <span className="font-bold text-white">
                                      {getHistogramStats(histogramData.input, 'input')?.average}
                                    </span>
                                  </div>
                                  <div className="flex justify-between">
                                    <span className="text-gray-400">Active Intensity Levels:</span>
                                    <span className="font-bold text-white">
                                      {getHistogramStats(histogramData.input, 'input')?.activePixels}
                                    </span>
                                  </div>
                                  <div className="pt-3 mt-3 border-t border-gray-700">
                                    <p className="text-sm text-gray-400">
                                      Shows distribution of pixel intensities (0=black, 255=white)
                                    </p>
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Output Color Histogram */}
                      <div className="bg-gray-900 p-6 rounded-xl border border-gray-700">
                        <div className="flex flex-col lg:flex-row gap-6">
                          <div className="lg:w-3/4">
                            <h3 className="text-xl font-bold mb-4 text-green-400 flex items-center gap-2">
                              <span className="p-2 bg-green-900 rounded-lg">
                                🌈
                              </span>
                              Output Color Histogram
                            </h3>
                            <div className="h-80">
                              <Line 
                                data={prepareHistogramData(histogramData.output, 'output')}
                                options={chartOptions('RGB Channel Distribution', 'output')}
                              />
                            </div>
                          </div>
                          <div className="lg:w-1/4">
                            <div className="bg-gray-800 p-4 rounded-lg h-full">
                              <h4 className="font-bold text-lg mb-3 text-gray-300">Channel Max Values</h4>
                              {getHistogramStats(histogramData.output, 'output') && (
                                <div className="space-y-3">
                                  <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                      <div className="w-3 h-3 rounded-full bg-red-500"></div>
                                      <span className="text-gray-300">Red:</span>
                                    </div>
                                    <span className="font-bold text-white">
                                      {getHistogramStats(histogramData.output, 'output')?.redMax}
                                    </span>
                                  </div>
                                  <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                      <div className="w-3 h-3 rounded-full bg-green-500"></div>
                                      <span className="text-gray-300">Green:</span>
                                    </div>
                                    <span className="font-bold text-white">
                                      {getHistogramStats(histogramData.output, 'output')?.greenMax}
                                    </span>
                                  </div>
                                  <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                      <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                                      <span className="text-gray-300">Blue:</span>
                                    </div>
                                    <span className="font-bold text-white">
                                      {getHistogramStats(histogramData.output, 'output')?.blueMax}
                                    </span>
                                  </div>
                                  <div className="pt-3 mt-3 border-t border-gray-700">
                                    <div className="flex justify-between">
                                      <span className="text-gray-400">Overall Max:</span>
                                      <span className="font-bold text-yellow-400">
                                        {getHistogramStats(histogramData.output, 'output')?.overallMax}
                                      </span>
                                    </div>
                                    <p className="text-sm text-gray-400 mt-2">
                                      Shows distribution of RGB color channels in the output image
                                    </p>
                                  </div>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Reference Image */}
              {referenceName && (
                <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-xl">
                  <div className="text-center mb-6">
                    <button
                      onClick={() => setShowReference(!showReference)}
                      className="px-8 py-4 bg-gradient-to-r from-indigo-600 to-purple-700 rounded-xl font-bold hover:scale-105 transition-all hover:shadow-lg flex items-center gap-2 mx-auto"
                    >
                      {showReference ? (
                        <>
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 15l7-7 7 7"></path>
                          </svg>
                          Hide Reference Image
                        </>
                      ) : (
                        <>
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                          </svg>
                          Show Reference Image
                        </>
                      )}
                    </button>
                  </div>

                  {showReference && (
                    <div className="mt-6 p-6 bg-gray-900 rounded-xl">
                      <h3 className="text-2xl font-bold text-yellow-400 mb-6 text-center flex items-center justify-center gap-2">
                        <span className="p-2 bg-yellow-900 rounded-lg">
                          📷
                        </span>
                        Reference Image Used
                      </h3>
                      <div className="p-4 bg-gray-800 rounded-lg">
                        <img
                          src={`http://127.0.0.1:5000/reference/${referenceName}`}
                          alt="Reference"
                          className="mx-auto rounded-lg max-h-96 border-2 border-gray-600 shadow-lg"
                        />
                      </div>
                      <p className="text-center text-gray-400 mt-4 font-semibold">
                        Reference: {referenceName}
                      </p>
                    </div>
                  )}
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