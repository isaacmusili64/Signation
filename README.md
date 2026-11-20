# SignCaption (Stage 1)

Webcam + MediaPipe HandLandmarker + in-browser KNN + live caption bar + text-to-speech.

## Run
    npm install
    npm run dev
Open http://localhost:3000 in Chrome (camera works on localhost or HTTPS only).

## Use
1. Type a word, hold the sign in view, press Record. Repeat for 5 signs.
2. Make a sign: after it holds steady for a moment, the caption appears.

## Notes
- Stage 1 recognises STATIC hand shapes only. Moving signs need the Stage 2 sequence model
  (trained on Google's ASL Signs dataset); `lib/classifier.js` `predict` is the swap point.
- Tunables in app/page.js: CONFIDENCE, HOLD_FRAMES.
