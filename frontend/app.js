// app.js
import vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";
const { FaceDetector, FaceLandmarker, ImageClassifier, FilesetResolver, DrawingUtils } = vision;

let faceDetector, faceLandmarker, imageClassifier;
let runningMode = "IMAGE";
let timer = 10;
let extraTime = 0;
let blinkCounter = 0;
let timeInterval;
let isFaceDetected = false;
let isFacePlain = false;
const blinkThreshold = 0.6;
const labels = ["covering", "plain"];
const overlayCanvas = document.getElementById("overlayCanvas");
const overlayCtx = overlayCanvas.getContext("2d");

const video = document.getElementById("webcam");
const canvas = document.getElementById("hiddenCanvas");
const ctx = canvas.getContext("2d");
const webcamPredictions = document.getElementById("webcamPredictions");
const liveView = document.getElementById("liveView");
const warning = document.getElementById("warning");
const timerBar = document.getElementById("timerBar");

let children = [];

const initializeAllModels = async () => {
  const resolver = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm");

  faceDetector = await FaceDetector.createFromOptions(resolver, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
      delegate: "GPU",
    },
    runningMode,
  });

  imageClassifier = await ImageClassifier.createFromOptions(resolver, {
    baseOptions: {
      modelAssetPath: `http://localhost:8000/model`,
    },
    maxResults: 1,
    runningMode,
  });

  faceLandmarker = await FaceLandmarker.createFromOptions(resolver, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      delegate: "GPU",
    },
    outputFaceBlendshapes: true,
    runningMode,
    numFaces: 1,
  });
};

const enableCam = async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  video.addEventListener("loadeddata", startProcess);
};

document.getElementById("webcamButton").addEventListener("click", enableCam);

const startProcess = () => {
  timer = 10;
  extraTime = 0;
  blinkCounter = 0;
  warning.innerText = "";
  startTimer();
  predictWebcam();
};

const startTimer = () => {
  if (timeInterval) return;
  document.getElementById("finalResult").innerHTML = "";
  timerBar.max = 10 + extraTime;
  timerBar.value = timer;

  timeInterval = setInterval(() => {
    if (timer > 0) {
      timer--;
      timerBar.value = timer;
    } else {
      clearInterval(timeInterval);
      timeInterval = null;

      if (blinkCounter === 0) {
        warning.innerText = "Try again. No valid blink detected.";
      } else {
        warning.innerText = "";
      }

      const resultMsg = `
        Face Detected: ${isFaceDetected ? "YES" : "NO"}<br/>
        Face Plain: ${isFacePlain ? "YES" : "NO"}<br/>
        Blink Detected: ${blinkCounter > 0 ? `YES (${blinkCounter})` : "NO"}
      `;
      document.getElementById("finalResult").innerHTML = resultMsg;
    }
  }, 1000);
};

let lastVideoTime = -1;

function isImageBlurred(imageData, threshold = 100) {
  // Approximate blur using edge sharpness
  const gray = [];
  const { data, width, height } = imageData;
  for (let i = 0; i < data.length; i += 4) {
    const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
    gray.push(avg);
  }

  // Simple Laplacian filter approximation
  let variance = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const center = gray[y * width + x];
      const sumNeighbors =
        gray[(y - 1) * width + x] +
        gray[(y + 1) * width + x] +
        gray[y * width + (x - 1)] +
        gray[y * width + (x + 1)];
      const laplacian = 4 * center - sumNeighbors;
      variance += laplacian * laplacian;
    }
  }

  variance /= gray.length;
  return variance < threshold;
}

function isFrameDark(imageData, brightnessThreshold = 50) {
  const { data } = imageData;
  let total = 0;
  for (let i = 0; i < data.length; i += 4) {
    total += 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]; // luminance
  }
  const avg = total / (data.length / 4);
  return avg < brightnessThreshold;
}


async function predictWebcam() {
  if (runningMode === "IMAGE") {
    runningMode = "VIDEO";
    await faceDetector.setOptions({ runningMode: "VIDEO" });
    await imageClassifier.setOptions({ runningMode: "IMAGE" });
    await faceLandmarker.setOptions({ runningMode: "VIDEO" });
  }

  const startTimeMs = performance.now();
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    // Capture current frame
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const frameData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  const blurDetected = isImageBlurred(frameData);
  const darkDetected = isFrameDark(frameData);

  if (blurDetected) warning.innerText = "Frame is blurry — try holding steady.";
  if (darkDetected) warning.innerText = "Low light detected — improve lighting.";



    const result = await faceDetector.detectForVideo(video, startTimeMs);
    const detections = result.detections;
    displayVideoDetections(detections);

    if (detections.length === 1) {
      const box = detections[0].boundingBox;
      await classifyFace(box);
      isFaceDetected = true;
    } else {
      isFaceDetected = false;
    }
  }

  if (timer > 0) {
    requestAnimationFrame(predictWebcam);
  }
}

async function classifyFace(boundingBox) {
  canvas.width = boundingBox.width;
  canvas.height = boundingBox.height;
  ctx.drawImage(
    video,
    boundingBox.originX,
    boundingBox.originY,
    boundingBox.width,
    boundingBox.height,
    0,
    0,
    canvas.width,
    canvas.height
  );
  const croppedImage = new Image();
  croppedImage.src = canvas.toDataURL();

  croppedImage.onload = async () => {
    const result = await imageClassifier.classify(croppedImage);
    const category = result.classifications[0].categories[0];

    webcamPredictions.className = "webcamPredictions";
    webcamPredictions.innerText = `Classification: ${labels[category.index]}\nConfidence: ${Math.round(category.score * 100)}%`;

    if (labels[category.index] === "plain") {
      isFacePlain = true;
      await detectBlink();
    } else {
      isFacePlain = false;
    }
  };
}

async function detectBlink() {
  const result = await faceLandmarker.detectForVideo(video, performance.now());
  if (result.faceBlendshapes.length > 0) {
    const blendShapes = result.faceBlendshapes[0].categories;
    const leftEye = blendShapes.find(b => b.categoryName === "eyeBlinkLeft");
    const rightEye = blendShapes.find(b => b.categoryName === "eyeBlinkRight");

    if (leftEye?.score > blinkThreshold || rightEye?.score > blinkThreshold) {
      blinkCounter++;
      webcamPredictions.innerText += `\nBlink Count: ${blinkCounter}`;
    } else if (timer <= 0 && blinkCounter === 0) {
      extraTime += 3;
      timer += 3;
      timerBar.max += 3;
    }
  }
}

function displayVideoDetections(dets) {
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  const w = video.offsetWidth, h = video.offsetHeight;
  overlayCanvas.width = w; overlayCanvas.height = h;

  const rx = w * 0.25, ry = h * 0.4, cx = w / 2, cy = h / 2;
  overlayCtx.setLineDash([5, 5]);
  overlayCtx.strokeStyle = "#0077cc";
  overlayCtx.beginPath();
  overlayCtx.ellipse(cx, cy, rx, ry, 0, 0, 2 * Math.PI);
  overlayCtx.stroke();

  let warnings = [];
  let faceInside = false;

  if (dets.length === 0) warnings.push("No face detected");
  if (dets.length > 1) warnings.push("Multiple faces detected");

  dets.forEach(det => {
    const { originX, originY, width, height } = det.boundingBox;
    const left = originX * (w / video.videoWidth);
    const top = originY * (h / video.videoHeight);
    const fw = width * (w / video.videoWidth);
    const fh = height * (h / video.videoHeight);
    overlayCtx.setLineDash([]);
    overlayCtx.strokeStyle = "red";
    overlayCtx.strokeRect(left, top, fw, fh);

    const dx = ((left + fw / 2) - cx) / rx;
    const dy = ((top + fh / 2) - cy) / ry;
    if (dx * dx + dy * dy < 1) faceInside = true;
    else {
      let mm = "Face not centered:";
      if (dx > 0.05) mm += " move left";
      if (dx < -0.05) mm += " move right";
      if (dy > 0.05) mm += " move up";
      if (dy < -0.05) mm += " move down";
      warnings.push(mm);
    }
  });

  // Pause timer if face is not centered or not plain
  const shouldPause = !faceInside || !isFacePlain;

  if (shouldPause && timer > 0) {
    if (timeInterval) {
      clearInterval(timeInterval);
      timeInterval = null;
    }
    if (!faceInside) warnings.push("Timer paused — face not centered");
    if (!isFacePlain) warnings.push("Timer paused — face is covered");
  }

  // Resume if everything is fine
  if (!shouldPause && timer > 0 && !timeInterval) {
    startTimer();
  }

  warning.innerHTML = warnings.join("<br>");
}


await initializeAllModels();