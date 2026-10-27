# AI Hand Painter

A browser-based version of my Python/OpenCV hand painting project. It tracks your hand through the webcam and lets you draw in the air with your index finger.

## Features

- Real-time hand tracking (MediaPipe Hand Landmarker)
- Draw with the index fingertip
- Colour buttons plus a custom colour picker
- Adjustable brush size and line smoothing
- Eraser, undo, clear
- Save the drawing as a PNG
- Pinch thumb and index finger together to pause drawing

## How to use

1. Open the page and click **Start camera**.
2. Allow camera access when the browser asks.
3. Hold up one hand and point with your index finger to draw.
4. Keyboard shortcuts: `Z` undo, `C` clear, `S` save.

## Tech

- HTML, CSS, JavaScript (no build step)
- MediaPipe Tasks Vision (loaded from a CDN)
- Hosted on Vercel

The camera video is processed locally in the browser. Nothing is uploaded.

## Running locally

Browsers only allow camera access on `https://` or `localhost`, so opening the file by double-clicking may not work. Run a local server instead:

```
python -m http.server 8000
```

Then open http://localhost:8000

## Notes

- Needs an internet connection to load the MediaPipe library and hand model.
- Works best in Chrome or Edge with good lighting.

## Original version

The first version was a Python script using OpenCV, MediaPipe and NumPy that ran on the desktop.
