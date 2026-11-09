SIGNSPEAK - ASL TO VOICE WEBSITE

FILES
- index.html
- style.css
- script.js
- model_weights.json (generated from your uploaded model.keras)

RUN THE WEBSITE
1. Keep all four files in the same folder.
2. Open the folder in Visual Studio Code.
3. Install/use the Live Server extension.
4. Right-click index.html and choose "Open with Live Server".
5. Allow camera permission in the browser.
6. Click Start camera and show one ASL hand sign.

IMPORTANT
- The provided model accepts 21 MediaPipe hand landmarks (x, y, z), 63 input values.
- The uploaded labels contain 24 static letters: A-I and K-Y. J and Z are not included.
- This website runs the uploaded Dense neural network in JavaScript using its extracted weights. MediaPipe Hands is loaded from a CDN, so internet access is needed.
- Camera access normally requires localhost (Live Server) or HTTPS. Opening index.html directly as a file may block the camera or model loading.
- Browser speech uses the built-in Web Speech API. Voice availability depends on the browser and installed system voices.
- This model recognises static hand shapes. It does not recognise moving letters J and Z.
- Model confidence shown on the page is an inference score, not a guaranteed accuracy measurement.

TROUBLESHOOTING
- If the model does not load, check the browser console and confirm model_weights.json is beside index.html.
- If the camera does not start, allow camera access, close other apps using the webcam, and use Live Server.
- Use a well-lit background and keep your hand fully visible.
