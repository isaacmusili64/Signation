"use client";
import { useEffect, useRef, useState, useCallback } from "react";
import { toFeatures, predict, loadSamples, saveSamples } from "@/lib/classifier";

const WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm";
const MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const CONFIDENCE = 0.7; // below this we show nothing: a wrong caption is worse than none
const HOLD_FRAMES = 8; // same sign must hold this many frames before it is captioned
const SAMPLES_PER_RECORD = 40;
const LINES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

export default function Page() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarkerRef = useRef(null);
  const samplesRef = useRef([]);
  const recordRef = useRef({ label: null, left: 0 });
  const holdRef = useRef({ label: null, count: 0, committed: null });
  const speakRef = useRef(false);

  const [status, setStatus] = useState("Loading hand tracker");
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);
  const [caption, setCaption] = useState("");
  const [log, setLog] = useState([]);
  const [labelInput, setLabelInput] = useState("");
  const [counts, setCounts] = useState({});
  const [recording, setRecording] = useState(false);
  const [speak, setSpeak] = useState(false);

  const refreshCounts = () => {
    const c = {};
    samplesRef.current.forEach((s) => (c[s.label] = (c[s.label] || 0) + 1));
    setCounts(c);
  };

  const commit = useCallback((label) => {
    setCaption(label);
    setLog((l) => [label, ...l].slice(0, 8));
    if (speakRef.current && "speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      window.speechSynthesis.speak(new SpeechSynthesisUtterance(label));
    }
  }, []);

  useEffect(() => { speakRef.current = speak; }, [speak]);

  useEffect(() => {
    let raf, stream, stopped = false;
    samplesRef.current = loadSamples();
    refreshCounts();

    async function start() {
      try {
        const { FilesetResolver, HandLandmarker } = await import("@mediapipe/tasks-vision");
        const fileset = await FilesetResolver.forVisionTasks(WASM);
        landmarkerRef.current = await HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: MODEL, delegate: "GPU" },
          runningMode: "VIDEO",
          numHands: 2,
        });
        setStatus("Starting camera");
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
          audio: false,
        });
        const video = videoRef.current;
        video.srcObject = stream;
        await video.play();
        setReady(true);
        setStatus("Ready");
        loop();
      } catch (e) {
        setError(
          e?.name === "NotAllowedError"
            ? "Camera access was blocked. Allow the camera in your browser's address bar, then reload."
            : "Could not start: " + (e?.message || e)
        );
      }
    }

    let lastTs = -1;
    function loop() {
      if (stopped) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (video && video.readyState >= 2 && video.currentTime !== lastTs) {
        lastTs = video.currentTime;
        const result = landmarkerRef.current.detectForVideo(video, performance.now());
        draw(canvas, video, result);
        handle(result);
      }
      raf = requestAnimationFrame(loop);
    }

    function draw(canvas, video, result) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 3;
      ctx.strokeStyle = "#5fd1b2";
      ctx.fillStyle = "#ffffff";
      for (const lm of result.landmarks || []) {
        for (const [a, b] of LINES) {
          ctx.beginPath();
          ctx.moveTo(lm[a].x * canvas.width, lm[a].y * canvas.height);
          ctx.lineTo(lm[b].x * canvas.width, lm[b].y * canvas.height);
          ctx.stroke();
        }
        for (const p of lm) {
          ctx.beginPath();
          ctx.arc(p.x * canvas.width, p.y * canvas.height, 4, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }

    function handle(result) {
      const hasHand = (result.landmarks || []).length > 0;
      const hold = holdRef.current;
      if (!hasHand) {
        hold.label = null; hold.count = 0; hold.committed = null;
        return;
      }
      const f = toFeatures(result);

      // Recording mode: collect samples for the typed label
      const rec = recordRef.current;
      if (rec.left > 0) {
        samplesRef.current.push({ label: rec.label, f });
        rec.left -= 1;
        if (rec.left === 0) {
          saveSamples(samplesRef.current);
          setRecording(false);
          refreshCounts();
        }
        return;
      }

      const p = predict(samplesRef.current, f);
      if (!p || p.confidence < CONFIDENCE) {
        hold.label = null; hold.count = 0;
        return;
      }
      if (p.label === hold.label) hold.count += 1;
      else { hold.label = p.label; hold.count = 1; }
      if (hold.count >= HOLD_FRAMES && hold.committed !== p.label) {
        hold.committed = p.label;
        commit(p.label);
      }
    }

    start();
    return () => {
      stopped = true;
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
      landmarkerRef.current?.close();
    };
  }, [commit]);

  function startRecording() {
    const label = labelInput.trim().toLowerCase();
    if (!label) return;
    recordRef.current = { label, left: SAMPLES_PER_RECORD };
    setRecording(true);
  }

  function clearAll() {
    samplesRef.current = [];
    saveSamples([]);
    refreshCounts();
    setCaption("");
  }

  const labels = Object.keys(counts);

  return (
    <main className="shell">
      <h1>SignCaption</h1>
      <p className="sub">
        Sign in front of your camera and the words appear as live captions. Hand tracking runs in
        your browser; video never leaves your device.
      </p>

      <div className="stage">
        <video ref={videoRef} playsInline muted />
        <canvas ref={canvasRef} aria-hidden="true" />
        <div className="status" role="status">{recording ? "Recording samples" : status}</div>
        <div className={"caption" + (caption ? "" : " idle")} aria-live="polite">
          {caption || (labels.length ? "Waiting for a sign" : "Teach a sign below to get started")}
        </div>
      </div>

      {error && <p className="error" role="alert">{error}</p>}

      <section className="panel" aria-labelledby="teach">
        <h2 id="teach">Teach a sign</h2>
        <div className="row" style={{ margin: 0 }}>
          <input
            type="text"
            placeholder="Word for this sign, e.g. stop"
            value={labelInput}
            onChange={(e) => setLabelInput(e.target.value)}
            aria-label="Word for this sign"
          />
          <button className="btn primary" disabled={!ready || recording || !labelInput.trim()} onClick={startRecording}>
            Record {SAMPLES_PER_RECORD} samples
          </button>
        </div>
        <p className="hint">
          Hold the sign steady in view, press record, and move it slightly while it captures.
          Record each sign a few times from different distances. Samples are stored only in this browser.
        </p>
        {labels.length > 0 && (
          <div className="row" style={{ marginBottom: 0 }}>
            <div className="chips">
              {labels.map((l) => (
                <span className="chip" key={l}>{l} ({counts[l]})</span>
              ))}
            </div>
            <button className="btn" onClick={clearAll}>Clear all samples</button>
          </div>
        )}
      </section>

      <section className="panel" aria-labelledby="out">
        <h2 id="out">Captions</h2>
        <label className="row" style={{ margin: "0 0 8px" }}>
          <input type="checkbox" checked={speak} onChange={(e) => setSpeak(e.target.checked)} />
          Read captions aloud
        </label>
        {log.length ? (
          <ol className="log">{log.map((w, i) => <li key={i}>{w}</li>)}</ol>
        ) : (
          <p className="hint">Recognised signs will be listed here, newest first.</p>
        )}
      </section>
    </main>
  );
}
