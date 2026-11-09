const video = document.getElementById('inputVideo');
const canvas = document.getElementById('outputCanvas');
const ctx = canvas.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const cameraStatus = document.getElementById('cameraStatus');
const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const messageEl = document.getElementById('message');
const letterDisplay = document.getElementById('letterDisplay');
const predictionText = document.getElementById('predictionText');
const confidenceText = document.getElementById('confidenceText');
const messageOutput = document.getElementById('messageOutput');
const letterCount = document.getElementById('letterCount');
const speakBtn = document.getElementById('speakBtn');

let modelWeights = null;
let hands = null;
let camera = null;
let cameraRunning = false;
let history = [];
let lastAdded = '';
let lastSeenAt = 0;
let message = '';
let lastFrameTime = 0;
const STABLE_WINDOW = 7;
const STABLE_COUNT = 5;
const REARM_MS = 850;
const MIN_CONFIDENCE = 0.55;

async function loadModel() {
  try {
    const response = await fetch('./model_weights.json');
    if (!response.ok) throw new Error('Could not load model_weights.json');
    modelWeights = await response.json();
    messageEl.textContent = 'Model loaded. Start the camera when ready.';
  } catch (error) {
    messageEl.textContent = 'Model loading failed. Run this website using a local server and keep model_weights.json beside the files.';
    console.error(error);
  }
}

// The uploaded Keras network is Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(24, Softmax).
// Dropout layers are inactive during inference.
function predict(landmarks) {
  const input = landmarks.flatMap(point => [point.x, point.y, point.z]);
  const dense = (values, weights, biases, relu) => {
    const out = new Array(biases.length);
    for (let j = 0; j < biases.length; j++) {
      let sum = biases[j];
      for (let i = 0; i < values.length; i++) sum += values[i] * weights[i][j];
      out[j] = relu ? Math.max(0, sum) : sum;
    }
    return out;
  };
  const h1 = dense(input, modelWeights.w1, modelWeights.b1, true);
  const h2 = dense(h1, modelWeights.w2, modelWeights.b2, true);
  const logits = dense(h2, modelWeights.w3, modelWeights.b3, false);
  const maxLogit = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - maxLogit));
  const total = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map(v => v / total);
  let best = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
  return { letter: modelWeights.labels[best], confidence: probs[best] };
}

function renderMessage() {
  messageOutput.textContent = message;
  if (!message) {
    messageOutput.innerHTML = '<span class="empty-message">Your recognised letters will appear here…</span>';
  }
  const count = message.replace(/\s/g, '').length;
  letterCount.textContent = `${count} ${count === 1 ? 'letter' : 'letters'}`;
  speakBtn.disabled = !message.trim();
}

function addLetter(letter) {
  message += letter;
  renderMessage();
  lastAdded = letter;
  lastSeenAt = Date.now();
  messageEl.textContent = `Added "${letter}" to your message.`;
}

function onResults(results) {
  const width = video.videoWidth || 640;
  const height = video.videoHeight || 480;
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height);
  ctx.restore();

  const handLandmarks = results.multiHandLandmarks && results.multiHandLandmarks[0];
  if (!handLandmarks || !modelWeights) {
    history = [];
    lastAdded = '';
    letterDisplay.textContent = '—';
    predictionText.textContent = 'Waiting for a sign';
    confidenceText.textContent = 'Show one hand clearly';
    return;
  }

  const prediction = predict(handLandmarks);
  letterDisplay.textContent = prediction.letter;
  predictionText.textContent = `Letter ${prediction.letter}`;
  confidenceText.textContent = `Model score: ${(prediction.confidence * 100).toFixed(1)}%`;

  if (prediction.confidence >= MIN_CONFIDENCE) {
    history.push(prediction.letter);
    if (history.length > STABLE_WINDOW) history.shift();
    const sameCount = history.filter(x => x === prediction.letter).length;
    if (sameCount >= STABLE_COUNT && (lastAdded !== prediction.letter || Date.now() - lastSeenAt > REARM_MS)) {
      addLetter(prediction.letter);
    }
  } else {
    history = [];
    lastAdded = '';
  }
}

async function startCamera() {
  if (!modelWeights) {
    messageEl.textContent = 'The model is not ready. Check that model_weights.json is in the project folder.';
    return;
  }
  if (!window.Hands || !window.Camera) {
    messageEl.textContent = 'MediaPipe could not load. Check your internet connection and refresh the page.';
    return;
  }
  try {
    messageEl.textContent = 'Requesting camera permission…';
    hands = new Hands({ locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}` });
    hands.setOptions({ maxNumHands: 1, modelComplexity: 1, minDetectionConfidence: 0.65, minTrackingConfidence: 0.6 });
    hands.onResults(onResults);
    camera = new Camera(video, {
      onFrame: async () => { if (hands) await hands.send({ image: video }); },
      width: 640,
      height: 480
    });
    await camera.start();
    cameraRunning = true;
    cameraPlaceholder.classList.add('hidden');
    cameraStatus.classList.add('active');
    cameraStatus.innerHTML = '<i></i> Camera on';
    startBtn.disabled = true;
    stopBtn.disabled = false;
    messageEl.textContent = 'Camera is running. Hold an ASL letter steady in view.';
  } catch (error) {
    console.error(error);
    messageEl.textContent = `Could not start camera: ${error.message || error}. Allow camera access and use localhost or HTTPS.`;
    stopCamera();
  }
}

function stopCamera() {
  if (camera) { camera.stop(); camera = null; }
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
    video.srcObject = null;
  }
  cameraRunning = false;
  if (hands) { hands.close(); hands = null; }
  cameraStatus.classList.remove('active');
  cameraStatus.innerHTML = '<i></i> Camera off';
  cameraPlaceholder.classList.remove('hidden');
  startBtn.disabled = false;
  stopBtn.disabled = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  history = [];
}

document.getElementById('spaceBtn').addEventListener('click', () => {
  if (message && !message.endsWith(' ')) message += ' ';
  renderMessage();
});
document.getElementById('deleteBtn').addEventListener('click', () => {
  message = message.slice(0, -1); renderMessage(); lastAdded = '';
});
document.getElementById('clearBtn').addEventListener('click', () => {
  message = ''; lastAdded = ''; renderMessage();
});
speakBtn.addEventListener('click', () => {
  if (!('speechSynthesis' in window)) {
    messageEl.textContent = 'Speech output is not supported in this browser.';
    return;
  }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(message.trim());
  utterance.lang = 'en-US';
  utterance.rate = 0.9;
  window.speechSynthesis.speak(utterance);
});
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
window.addEventListener('beforeunload', stopCamera);

loadModel();
