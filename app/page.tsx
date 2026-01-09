"use client";

import React, { useRef, useState, useCallback, useEffect, useMemo } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import {
  Moon, Sun, Info, ScanFace, Activity, Lock, RefreshCcw,
  Settings2, Trash2, Volume2, VolumeX, Brain, List, Search,
  Mic, MicOff, ArrowRight
} from "lucide-react";
import FeedbackSection from "@/components/feedBackSection";

// --- Types ---

interface Landmark { x: number; y: number; z: number; }

interface Sample { label: string; landmarks: Landmark[]; }

interface TrainingSample { label: string; vector: number[]; }

interface CameraSettings { brightness: number; contrast: number; saturation: number; }

interface GestureLibraryItem { emoji: string; title: string; description: string; image?: string; }

interface VideoStageProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  detectedLabel: string;
  confidence: number;
  isLocked: boolean;
  settings: CameraSettings;
}

// --- Inline UI Components ---

const Card = ({ className, children }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={`rounded-xl border bg-card text-card-foreground shadow ${className || ""}`}>{children}</div>
);

const Button = React.forwardRef<HTMLButtonElement, React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "default" | "destructive" | "outline" | "secondary" | "ghost", size?: "default" | "sm" | "icon" | "lg" }>(
  ({ className, variant = "default", size = "default", ...props }, ref) => {
    const variants = {
      default: "bg-primary text-primary-foreground hover:bg-primary/90",
      destructive: "bg-destructive text-destructive-foreground hover:bg-destructive/90",
      outline: "border border-input bg-background hover:bg-accent hover:text-accent-foreground",
      secondary: "bg-secondary text-secondary-foreground hover:bg-secondary/80",
      ghost: "hover:bg-accent hover:text-accent-foreground"
    };
    const sizes = {
      default: "h-10 px-4 py-2",
      sm: "h-9 rounded-md px-3",
      icon: "h-10 w-10",
      lg: "h-14 px-8 text-lg"
    };
    return (
      <button
        ref={ref}
        className={`inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-colors focus-visible:outline-none disabled:pointer-events-none disabled:opacity-50 ${variants[variant]} ${sizes[size]} ${className || ""}`}
        {...props}
      />
    );
  }
);
Button.displayName = "Button";

const Input = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, type, ...props }, ref) => {
    return (
      <input
        type={type}
        className={`flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background file:border-0 file:bg-transparent file:text-sm file:font-medium placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${className || ""}`}
        ref={ref}
        {...props}
      />
    )
  }
);
Input.displayName = "Input";

const Separator = ({ className, orientation = "horizontal" }: { className?: string, orientation?: "horizontal" | "vertical" }) => (
  <div className={`shrink-0 bg-border ${orientation === 'horizontal' ? 'h-[1px] w-full' : 'h-full w-[1px]'} ${className || ""}`} />
);

const Slider = React.forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(
  ({ className, ...props }, ref) => (
    <input
      type="range"
      className={`w-full h-2 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary ${className || ""}`}
      ref={ref}
      {...props}
    />
  )
);
Slider.displayName = "Slider";

// --- Internal Components & Hooks ---

function ModeToggle() {
  const [isDark, setIsDark] = useState(false);
  useEffect(() => {
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    setIsDark(prefersDark);
    if (prefersDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, []);

  const toggleTheme = () => {
    if (isDark) {
      document.documentElement.classList.remove("dark");
      setIsDark(false);
    } else {
      document.documentElement.classList.add("dark");
      setIsDark(true);
    }
  };
  return (
    <Button variant="outline" size="icon" onClick={toggleTheme} title="Toggle Dark/Light Mode">
      {isDark ? <Moon className="h-[1.2rem] w-[1.2rem] text-yellow-300" /> : <Sun className="h-[1.2rem] w-[1.2rem] text-orange-500" />}
    </Button>
  );
}

// --- Gesture Library & Logic ---

const GESTURE_LIBRARY: Record<string, GestureLibraryItem> = {
  "A": { emoji: "✊", title: "Letter A", description: "Fist with thumb resting against the side of the index finger." },
  "B": { emoji: "🖐️", title: "Letter B", description: "Open palm, all fingers straight and touching, thumb tucked across palm." },
  "C": { emoji: "🤟", title: "Letter C", description: "Curve all fingers and thumb to form a 'C' shape." },
  "D": { emoji: "👆", title: "Letter D", description: "Index finger points up; thumb and other fingers form a circle." },
  "E": { emoji: "👊", title: "Letter E", description: "Fingers curled down, thumb tucked underneath." },
  "F": { emoji: "👌", title: "Letter F", description: "Touch thumb and index tips together; keep other three fingers up and spread." },
  "G": { emoji: "👈", title: "Letter G", description: "Index finger and thumb extended parallel, pointing sideways like a pinch." },
  "H": { emoji: "🤜", title: "Letter H", description: "Index and middle fingers extended together pointing sideways." },
  "I": { emoji: "🤙", title: "Letter I", description: "Pinky finger extended straight up, all other fingers closed." },
  "J": { emoji: "🤙", title: "Letter J", description: "Pinky finger makes a J swooping motion." },
  "K": { emoji: "✌️", title: "Letter K", description: "Index up, middle finger points forward, thumb between them." },
  "L": { emoji: "🇱", title: "Letter L", description: "Thumb and index finger extended to form an 'L' shape." },
  "M": { emoji: "👊", title: "Letter M", description: "Fist with thumb tucked under first three fingers." },
  "N": { emoji: "👊", title: "Letter N", description: "Fist with thumb tucked under first two fingers." },
  "O": { emoji: "👌", title: "Letter O", description: "Fingers and thumb touch to make an O shape." },
  "P": { emoji: "👇", title: "Letter P", description: "Index pointing down, middle finger down, thumb between." },
  "Q": { emoji: "👇", title: "Letter Q", description: "Index and thumb pointing down." },
  "R": { emoji: "🤞", title: "Letter R", description: "Index and middle fingers crossed." },
  "S": { emoji: "✊", title: "Letter S", description: "Fist with thumb crossed over fingers." },
  "T": { emoji: "✊", title: "Letter T", description: "Fist with thumb tucked between index and middle finger." },
  "U": { emoji: "✌️", title: "Letter U", description: "Index and middle fingers extended straight up and touching." },
  "V": { emoji: "✌️", title: "Letter V", description: "Index and middle fingers extended straight up in a 'V' shape." },
  "W": { emoji: "🤟", title: "Letter W", description: "Index, middle, and ring fingers extended and spread." },
  "X": { emoji: "☝️", title: "Letter X", description: "Index finger hooked like a pirate hook." },
  "Y": { emoji: "🤙", title: "Letter Y", description: "Thumb and pinky finger extended, others tucked in." },
  "Z": { emoji: "☝️", title: "Letter Z", description: "Index finger draws a Z in the air." },

  // --- NUMBERS (STATIC) ---
  "1": { emoji: "☝️", title: "Number 1", description: "Only index finger extended straight up." },
  "2": { emoji: "✌️", title: "Number 2", description: "Index and middle fingers extended and spread." },
  "3": { emoji: "🤟", title: "Number 3", description: "Thumb, index, and middle fingers extended." },
  "4": { emoji: "🖖", title: "Number 4", description: "Four fingers extended and spread, thumb tucked in." },
  "5": { emoji: "✋", title: "Number 5", description: "All fingers and thumb extended and spread wide." },

  // --- PHRASES (STATIC) ---
  "I Love You": { emoji: "🤟", title: "I Love You", description: "Thumb, index, and pinky extended; middle and ring fingers tucked." },
  "OK": { emoji: "👌", title: "OK", description: "Thumb and index finger forming a circle, other fingers extended." }
};

const Finger = {
  Thumb: 0,
  Index: 1,
  Middle: 2,
  Ring: 3,
  Pinky: 4,
};

// --- Math Helpers ---

const calculateDistance = (p1: { x: number, y: number }, p2: { x: number, y: number }) => {
  return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
};

const isExtended = (landmarks: Landmark[], fingerIdx: number) => {
  const tip = landmarks[fingerIdx * 4 + 4];
  const pip = landmarks[fingerIdx * 4 + 2];
  const mcp = landmarks[fingerIdx * 4 + 1];

  if (fingerIdx === Finger.Thumb) {
    const distanceTipToPinky = calculateDistance(tip, landmarks[17]);
    const distanceMcpToPinky = calculateDistance(mcp, landmarks[17]);
    return distanceTipToPinky > distanceMcpToPinky;
  }

  // Fingers usually point UP in standard webcam view
  // We compare tip Y to PIP Y (remember Y increases downwards in canvas)
  return calculateDistance(tip, landmarks[0]) > calculateDistance(pip, landmarks[0]);
};

// We define these outside the component to ensure they are available
function dist(p1: Landmark, p2: Landmark): number {
  return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
}

function isFingerExtended(landmarks: Landmark[], fingerName: string): boolean {
  const fingerMap: Record<string, number> = {
    "Thumb": Finger.Thumb, "Index": Finger.Index, "Middle": Finger.Middle, "Ring": Finger.Ring, "Pinky": Finger.Pinky,
  };
  const fingerIdx = fingerMap[fingerName];
  if (fingerIdx === undefined) return false;
  return isExtended(landmarks, fingerIdx);
}

function isFingerCurled(landmarks: Landmark[], fingerName: string): boolean {
  return !isFingerExtended(landmarks, fingerName);
}

// --- The Brains (Geometric Fallback - UPGRADED) ---
const recognizeGeometricGesture = (landmarks: Landmark[]) => {
  if (!landmarks || landmarks.length === 0) return null;

  // Finger states (rotation-safe)
  const thumb = isFingerExtended(landmarks, "Thumb");
  const index = isFingerExtended(landmarks, "Index");
  const middle = isFingerExtended(landmarks, "Middle");
  const ring = isFingerExtended(landmarks, "Ring");
  const pinky = isFingerExtended(landmarks, "Pinky");

  // Distances
  const dThumbIndex = dist(landmarks[4], landmarks[8]);
  const dIndexMiddle = dist(landmarks[8], landmarks[12]);
  const dIndexPinky = dist(landmarks[8], landmarks[20]);

  // ---- HIGH CONFIDENCE FIRST ----

  // I LOVE YOU
  if (thumb && index && pinky && !middle && !ring) {
    return "I Love You";
  }

  // F / OK
  if (dThumbIndex < 0.05 && middle && ring && pinky) {
    return "F";
  }

  // A
  if (!index && !middle && !ring && !pinky && thumb) {
    return "A";
  }

  // B
  if (index && middle && ring && pinky && !thumb && dIndexPinky < 0.15) {
    return "B";
  }

  // C
  if (thumb && dIndexPinky > 0.12 && dIndexPinky < 0.25) {
    return "C";
  }

  // D
  if (index && !middle && !ring && !pinky && dThumbIndex < 0.05) {
    return "D";
  }

  // G (sideways)
  if (index && thumb && !middle && !ring && !pinky && dThumbIndex > 0.07) {
    return "G";
  }

  // H
  if (index && middle && !ring && !pinky && dIndexMiddle < 0.05) {
    return "H";
  }

  // L
  if (thumb && index && !middle && !ring && !pinky) {
    return "L";
  }

  // I
  if (pinky && !index && !middle && !ring) {
    return "I";
  }

  // ---- NUMBERS / SIMPLE FALLBACKS ----

  if (index && !middle && !ring && !pinky) return "1";
  if (index && middle && !ring && !pinky) {
    return dIndexMiddle < 0.05 ? "U" : "V";
  }
  if (thumb && index && middle && !ring && !pinky) return "3";
  if (!thumb && index && middle && ring && pinky) return "4";
  if (thumb && index && middle && ring && pinky) return "5";
  if (thumb && pinky && !index && !middle && !ring) return "Y";

  return null;
};


// --- NEW ML Logic: k-Nearest Neighbors (k-NN) ---

const calculateEuclideanDistance = (v1: number[], v2: number[]): number => {
  if (v1.length !== v2.length) return Infinity;
  let sum = 0;
  for (let i = 0; i < v1.length; i++) {
    sum += Math.pow(v1[i] - v2[i], 2);
  }
  return Math.sqrt(sum);
};

const K = 5;

const predictKNN = (inputVector: number[], trainingData: TrainingSample[]): { label: string, confidence: number } | null => {
  if (trainingData.length === 0 || inputVector.length === 0) return null;

  const distances = trainingData.map(sample => ({
    label: sample.label,
    distance: calculateEuclideanDistance(inputVector, sample.vector)
  }));

  distances.sort((a, b) => a.distance - b.distance);
  const neighbors = distances.slice(0, Math.min(K, trainingData.length));

  const votes: Record<string, { weight: number, count: number }> = {};
  let totalWeight = 0;

  neighbors.forEach(n => {
    const weight = 1 / (n.distance + 0.0001);

    if (!votes[n.label]) {
      votes[n.label] = { weight: 0, count: 0 };
    }
    votes[n.label].weight += weight;
    votes[n.label].count += 1;
    totalWeight += weight;
  });

  let winner: string | null = null;
  let maxWeight = -Infinity;

  Object.keys(votes).forEach(label => {
    if (votes[label].weight > maxWeight) {
      maxWeight = votes[label].weight;
      winner = label;
    }
  });

  if (!winner) return null;

  const confidence = totalWeight > 0 ? (maxWeight / totalWeight) : 0;

  return { label: winner, confidence: Math.min(confidence * 100, 100) };
};

// --- Drawing Helper ---
const drawLandmarks = (ctx: CanvasRenderingContext2D | null, landmarks: Landmark[]) => {
  if (!ctx) return;
  ctx.lineWidth = 2;
  const connections = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

  ctx.strokeStyle = "#00FF00";
  connections.forEach(([i, j]) => {
    ctx.beginPath();
    ctx.moveTo(landmarks[i].x * ctx.canvas.width, landmarks[i].y * ctx.canvas.height);
    ctx.lineTo(landmarks[j].x * ctx.canvas.width, landmarks[j].y * ctx.canvas.height);
    ctx.stroke();
  });

  landmarks.forEach((p, i) => {
    const x = p.x * ctx.canvas.width;
    const y = p.y * ctx.canvas.height;
    ctx.fillStyle = (i % 4 === 0 && i !== 0) ? "#FFFF00" : "#FF0000";
    ctx.strokeStyle = "#FFFFFF";
    ctx.beginPath();
    ctx.arc(x, y, (i % 4 === 0 && i !== 0) ? 5 : 3, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
  });
};

// --- useLandmarker Hook ---

function useLandmarker({
  videoRef,
  canvasRef,
  onFrameProcessed
}: {
  videoRef: React.RefObject<HTMLVideoElement | null>,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  onFrameProcessed: (vector: number[], rawData: Landmark[]) => void
}) {
  const [status, setStatus] = useState("Loading Model...");
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const requestRef = useRef<number | null>(null);

  const onFrameProcessedRef = useRef(onFrameProcessed);

  useEffect(() => {
    onFrameProcessedRef.current = onFrameProcessed;
  }, [onFrameProcessed]);

  const extractVector = (landmarks: Landmark[]) => {
    return landmarks.flatMap(p => [p.x, p.y, p.z]);
  };

  const predictWebcam = useCallback(() => {
    if (!landmarkerRef.current || !videoRef.current || !canvasRef.current) {
      requestRef.current = requestAnimationFrame(predictWebcam);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      requestRef.current = requestAnimationFrame(predictWebcam);
      return;
    }

    const ctx = canvas.getContext("2d");
    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const startTimeMs = performance.now();

    try {
      if (landmarkerRef.current) {
        const results = landmarkerRef.current.detectForVideo(video, startTimeMs);

        ctx?.clearRect(0, 0, canvas.width, canvas.height);

        if (results.landmarks && results.landmarks.length > 0) {
          const landmarks = results.landmarks[0] as Landmark[];
          drawLandmarks(ctx, landmarks);
          const vector = extractVector(landmarks);

          onFrameProcessedRef.current(vector, landmarks);
        } else {
          onFrameProcessedRef.current([], []);
        }
      }
    } catch (e) {
      console.warn(e);
    }

    requestRef.current = requestAnimationFrame(predictWebcam);
  }, [videoRef, canvasRef]);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let isCancelled = false;

    const initLandmarker = async () => {
      if (typeof window === 'undefined' || !navigator.mediaDevices) {
        if (!isCancelled) setStatus("Not in a browser environment.");
        return;
      }
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        landmarkerRef.current = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        });

        if (!isCancelled) setStatus("Starting Camera...");
        startCamera();
      } catch (error) {
        console.error("Landmarker Init Error:", error);
        if (!isCancelled) setStatus("Failed to load AI.");
      }
    };

    const startCamera = async () => {
      if (videoRef.current && navigator.mediaDevices) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480 }
          });

          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            videoRef.current.onloadeddata = () => {
              predictWebcam();
              if (!isCancelled) setStatus("Ready: Show Hand");
            };
          }
        } catch (err) {
          console.error("Camera Init Error:", err);
          if (!isCancelled) setStatus("Camera Error (Check Permissions)");
        }
      }
    };

    initLandmarker();

    return () => {
      isCancelled = true;
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
      if (stream) stream.getTracks().forEach(track => track.stop());
    };
  }, [predictWebcam, videoRef]);

  return { status };
}

// --- usePredictionModel Hook ---

function usePredictionModel() {
  const [modelType, setModelType] = useState<"none" | "geometric" | "ml">("geometric");
  const [detectedLabel, setDetectedLabel] = useState("");
  const [debugStatus, setDebugStatus] = useState("Waiting...");
  const [confidence, setConfidence] = useState(0);
  const [isLocked, setIsLocked] = useState(false);

  const trainingDataRef = useRef<TrainingSample[]>([]);
  const [mlDataSize, setMlDataSize] = useState(0);

  const bufferRef = useRef<string[]>([]);
  const wristXBufferRef = useRef<number[]>([]);
  const lastSuccessfulDetectionTime = useRef<number>(0);

  const PREDICTION_BUFFER_SIZE = 8;
  const WRIST_BUFFER_SIZE = 20;
  const SCAN_DELAY_MS = 1000;

  const loadMLModel = (data: TrainingSample[]) => {
    trainingDataRef.current = data;
    setMlDataSize(data.length);
  };

  const predict = (inputVector: number[], rawLandmarks: Landmark[]): string | null => {
    if (modelType === "none" || !rawLandmarks || rawLandmarks.length === 0) {
      setDebugStatus("No Hand Detected");
      setConfidence(0);
      setIsLocked(false);
      bufferRef.current = [];
      wristXBufferRef.current = [];
      setDetectedLabel("");
      return null;
    }

    if (Date.now() - lastSuccessfulDetectionTime.current < SCAN_DELAY_MS) {
      setIsLocked(true);
      return detectedLabel;
    }
    setIsLocked(false);

    let result: string | null = null;

    if (modelType === "geometric") {
      result = recognizeGeometricGesture(rawLandmarks);

      const currentWristX = rawLandmarks[0].x;
      wristXBufferRef.current.push(currentWristX);
      if (wristXBufferRef.current.length > WRIST_BUFFER_SIZE) {
        wristXBufferRef.current.shift();
      }

      // Dynamic Hello Wave Check
      if ((result === "B" || result === "5") && wristXBufferRef.current.length === WRIST_BUFFER_SIZE) {
        const xValues = wristXBufferRef.current;
        const travel = xValues.reduce((acc, val, i) => i === 0 ? 0 : acc + Math.abs(val - xValues[i - 1]), 0);
        const displacement = Math.abs(xValues[xValues.length - 1] - xValues[0]);

        if (travel > 0.4 && displacement < 0.3) {
          result = "Hello";
        }
      }

    } else if (modelType === "ml") {
      if (trainingDataRef.current.length > K) {
        const wrist = rawLandmarks[0];
        const normalizedInput = rawLandmarks.flatMap(p => [
          p.x - wrist.x,
          p.y - wrist.y,
          p.z - wrist.z
        ]);

        const mlPrediction = predictKNN(normalizedInput, trainingDataRef.current);
        if (mlPrediction) {
          result = mlPrediction.label;
        }
      } else {
        setDebugStatus(`ML Mode: Need at least ${K} samples to predict.`);
        setConfidence(0);
        return null;
      }
    }

    setDebugStatus(result ? `Match: ${result} (${modelType.toUpperCase()} Mode)` : "Hand Detected - Unknown Pose");

    if (!result) {
      setConfidence(0);
      return null;
    }

    bufferRef.current.push(result);
    if (bufferRef.current.length > PREDICTION_BUFFER_SIZE) {
      bufferRef.current.shift();
    }

    const counts: Record<string, number> = {};
    bufferRef.current.forEach((g) => { counts[g] = (counts[g] || 0) + 1; });

    const winner = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b, "");
    const winCount = counts[winner];
    const calcConfidence = Math.round((winCount / bufferRef.current.length) * 100);

    setConfidence(calcConfidence);

    if (winCount > PREDICTION_BUFFER_SIZE * 0.75) {
      if (winner !== detectedLabel) {
        lastSuccessfulDetectionTime.current = Date.now();
      }
      return winner;
    }

    return null;
  };

  return {
    modelType,
    setModelType,
    detectedLabel,
    setDetectedLabel,
    predict,
    debugStatus,
    confidence,
    isLocked,
    trainingDataRef,
    mlDataSize,
    loadMLModel
  };
}

// --- useTextToSpeech Hook ---

function useTextToSpeech() {
  const [enabled, setEnabled] = useState(true);
  const lastSpokenRef = useRef<string>("");
  const lastTimeRef = useRef<number>(0);

  const speak = (text: string) => {
    if (!enabled || !text) return;
    const now = Date.now();
    if (text === lastSpokenRef.current && now - lastTimeRef.current < 2000) return;

    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    window.speechSynthesis.speak(utter);

    lastSpokenRef.current = text;
    lastTimeRef.current = now;
  };

  const toggle = () => setEnabled((prev) => !prev);
  return { enabled, toggle, speak };
}

// --- NEW HOOK: Voice to Sign Recognition ---
function useSpeechToSign() {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [isSupported, setIsSupported] = useState(true);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const recognitionRef = useRef<any>(null);

  useEffect(() => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if (typeof window !== "undefined" && (window as any).webkitSpeechRecognition) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const SpeechRecognition = (window as any).webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      recognition.onresult = (event: any) => {
        let finalTranscript = "";
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          }
        }
        if (finalTranscript) {
          setTranscript(finalTranscript);
        }
      };

      recognition.onend = () => {
        if (isListening) recognition.start(); // Keep listening
      };

      recognitionRef.current = recognition;
    } else {
      setIsSupported(false);
    }
  }, [isListening]);

  const toggleListen = () => {
    if (!recognitionRef.current) return;
    if (isListening) {
      recognitionRef.current.stop();
      setIsListening(false);
    } else {
      recognitionRef.current.start();
      setIsListening(true);
      setTranscript("");
    }
  };

  return { isListening, transcript, toggleListen, isSupported, setTranscript };
}

// --- useDataCollection Hook ---

function useDataCollection(trainingDataRef: React.MutableRefObject<TrainingSample[]>) {
  const [collecting, setCollecting] = useState(false);
  const [sampleCount, setSampleCount] = useState(0);
  const activeLabelRef = useRef("");
  const dataBufferRef = useRef<Sample[]>([]);

  // Cooldown tracking for individual samples
  const lastSampleTimeRef = useRef<number>(0);
  const SAMPLE_COOLDOWN_MS = 200; // 5 samples per second max

  const startCollecting = (lbl: string) => {
    if (!lbl) return alert("Enter label");
    activeLabelRef.current = lbl;
    setCollecting(true);
    dataBufferRef.current = [];
    setSampleCount(0);
    lastSampleTimeRef.current = 0;
  }
  const stopCollecting = () => setCollecting(false);

  const addSample = (vec: number[], raw: Landmark[]) => {
    const now = Date.now();
    // Check cooldown before adding a new sample
    if (collecting && raw.length > 0 && (now - lastSampleTimeRef.current > SAMPLE_COOLDOWN_MS)) {
      dataBufferRef.current.push({ label: activeLabelRef.current, landmarks: raw });
      setSampleCount(dataBufferRef.current.length);
      lastSampleTimeRef.current = now; // Update time after a successful collection
    }
  }

  const trainModel = () => {
    if (dataBufferRef.current.length === 0) return alert("No samples collected to train the model.");

    const newTrainingData: TrainingSample[] = dataBufferRef.current.map(sample => {
      const wrist = sample.landmarks[0];
      const normalizedVector = sample.landmarks.flatMap(p => [
        p.x - wrist.x,
        p.y - wrist.y,
        p.z - wrist.z
      ]);

      return { label: sample.label, vector: normalizedVector };
    });

    trainingDataRef.current = [...trainingDataRef.current, ...newTrainingData];

    setCollecting(false);
    dataBufferRef.current = [];
    setSampleCount(0);

    return trainingDataRef.current;
  };

  const saveDataset = () => {
    const fullDataset = trainingDataRef.current;
    if (fullDataset.length === 0) return alert("No data available to download.");

    const blob = new Blob([JSON.stringify(fullDataset)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "custom_ml_dataset.json";
    a.click();
  }

  const loadDataset = (onLoad: (data: TrainingSample[]) => void) => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const data = JSON.parse(e.target?.result as string) as TrainingSample[];
            onLoad(data);
          } catch (error) {
            alert("Failed to parse JSON file.");
          }
        };
        reader.readAsText(file);
      }
    };
    input.click();
  };

  return {
    collecting,
    sampleCount,
    startCollecting,
    stopCollecting,
    addSample,
    saveDataset,
    trainModel,
    loadDataset
  };
}

// --- NEW COMPONENT: Voice to Sign Player (THEME AWARE & BIGGER UI) ---
function VoiceSignPlayer({ transcript, isListening, toggleListen }: { transcript: string, isListening: boolean, toggleListen: () => void }) {
  const [currentSign, setCurrentSign] = useState<GestureLibraryItem | null>(null);
  const [queue, setQueue] = useState<string[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentIndex, setCurrentIndex] = useState(0);

  // Parse transcript into gesture queue
  useEffect(() => {
    if (!transcript) return;

    const words = transcript.trim().toUpperCase().split(/\s+/);
    const newQueue: string[] = [];

    words.forEach(word => {
      // 1. Check if whole word exists in Library (e.g., "HELLO")
      const libKey = Object.keys(GESTURE_LIBRARY).find(k => k.toUpperCase() === word);

      if (libKey) {
        newQueue.push(libKey);
      } else {
        // 2. Fallback: Fingerspell the word
        const letters = word.split("");
        letters.forEach(char => {
          if (GESTURE_LIBRARY[char]) {
            newQueue.push(char);
          }
        });
      }
    });

    if (newQueue.length > 0) {
      setQueue(newQueue);
      setIsPlaying(true);
      setCurrentIndex(0);
    }
  }, [transcript]);

  // Playback Loop
  useEffect(() => {
    if (!isPlaying || queue.length === 0) return;

    const interval = setInterval(() => {
      if (currentIndex < queue.length) {
        const key = queue[currentIndex];
        setCurrentSign(GESTURE_LIBRARY[key]);
        setCurrentIndex(prev => prev + 1);
      } else {
        setIsPlaying(false);
        setCurrentSign(null);
        setCurrentIndex(0);
        clearInterval(interval);
      }
    }, 1200); // Speed of playback

    return () => clearInterval(interval);
  }, [isPlaying, currentIndex, queue]);

  // Preview of next sign
  const nextKey = queue[currentIndex] || null;
  const nextSign = nextKey ? GESTURE_LIBRARY[nextKey] : null;

  return (
    <Card className="flex flex-col h-full overflow-hidden border-2 border-primary/20 shadow-xl bg-card">
      {/* Header */}
      <div className="p-4 border-b bg-muted/20 flex justify-between items-center">
        <h3 className="font-bold text-lg uppercase flex items-center gap-2 text-primary">
          <Mic className="w-5 h-5" /> Voice to Sign
        </h3>
        <Button
          size="sm"
          variant={isListening ? "destructive" : "outline"}
          onClick={toggleListen}
          className={`font-semibold ${isListening ? "animate-pulse" : ""}`}
        >
          {isListening ? <><MicOff className="w-4 h-4 mr-2" /> Stop Listening</> : <><Mic className="w-4 h-4 mr-2" /> Start Listening</>}
        </Button>
      </div>

      {/* Main Stage (Adapts to Light/Dark Mode) */}
      <div className="flex-grow bg-background flex flex-col items-center justify-center relative p-8 min-h-[300px]">
        {currentSign ? (
          <div className="text-center animate-in zoom-in duration-200 flex flex-col items-center">
            {/* The BIG Emoji */}
            <div className="text-[150px] leading-none mb-6 drop-shadow-2xl filter hover:brightness-110 transition-all select-none">
              {currentSign.emoji}
            </div>

            {/* The Text Label */}
            <div className="bg-card text-primary px-8 py-3 rounded-full text-3xl font-black shadow-lg border border-primary/20">
              {currentSign.title}
            </div>

            {/* Description */}
            <p className="text-muted-foreground mt-4 text-sm max-w-xs">{currentSign.description}</p>
          </div>
        ) : (
          <div className="text-center text-muted-foreground flex flex-col items-center gap-6">
            {isListening ? (
              <>
                <div className="relative">
                  <div className="w-32 h-32 rounded-full bg-primary/20 animate-ping absolute inset-0" />
                  <div className="w-32 h-32 rounded-full bg-primary/10 flex items-center justify-center border-4 border-primary/30 relative z-10">
                    <Mic className="w-12 h-12 text-primary" />
                  </div>
                </div>
                <p className="text-xl font-medium animate-pulse text-primary">Listening for speech...</p>
              </>
            ) : (
              <>
                <MicOff className="w-24 h-24 opacity-20" />
                <p className="text-lg">Click &quot;Start Listening&quot; to translate voice</p>
              </>
            )}
          </div>
        )}

        {/* Next Up Preview (Small indicator at bottom right) */}
        {isPlaying && nextSign && (
          <div className="absolute bottom-4 right-4 bg-muted/80 backdrop-blur-md p-2 rounded-lg border border-border flex items-center gap-2 text-muted-foreground text-xs shadow-sm">
            <span>Next:</span>
            <span className="text-xl">{nextSign.emoji}</span>
            <ArrowRight className="w-3 h-3" />
          </div>
        )}
      </div>

      {/* Transcript Footer */}
      <div className="p-4 border-t bg-muted/10 min-h-[80px]">
        <p className="text-xs font-bold uppercase text-muted-foreground mb-2 flex items-center gap-2">
          <Activity className="w-3 h-3" /> Live Transcript
        </p>
        <p className="text-xl font-medium text-foreground leading-relaxed">
          {transcript || <span className="text-muted-foreground/40 italic">Waiting for speech...</span>}
        </p>
      </div>
    </Card>
  );
}

// --- Main Component ---

export default function ASLRecorder() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const predictor = usePredictionModel();
  const collector = useDataCollection(predictor.trainingDataRef);
  const tts = useTextToSpeech();
  const speech = useSpeechToSign(); // NEW HOOK

  const [inputLabel, setInputLabel] = useState("");
  const [sentence, setSentence] = useState("");
  const [isFeedbackOpen, setIsFeedbackOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  const [camSettings, setCamSettings] = useState<CameraSettings>({
    brightness: 100,
    contrast: 100,
    saturation: 100
  });

  // --- Optimization: Memoize the Filtered Library ---
  const filteredLibrary = useMemo(() => {
    const entries = Object.entries(GESTURE_LIBRARY);
    if (!searchQuery.trim()) return entries;
    const query = searchQuery.toLowerCase();
    return entries.filter(([key, item]) =>
      key.toLowerCase().includes(query) ||
      item.title.toLowerCase().includes(query)
    );
  }, [searchQuery]);

  const resetCamSettings = () => setCamSettings({ brightness: 100, contrast: 100, saturation: 100 });
  const lastAddedTimeRef = useRef<number>(0);
  const SENTENCE_ADD_COOLDOWN = 2000;

  const handleFrame = useCallback((vector: number[], rawData: Landmark[]) => {
    if (collector.collecting) collector.addSample(vector, rawData);

    if (predictor.modelType !== "none") {
      const result = predictor.predict(vector, rawData);

      if (result && result !== predictor.detectedLabel) {
        predictor.setDetectedLabel(result);
        tts.speak(result);
      }

      if (result && predictor.confidence > 80 && !predictor.isLocked) {
        const now = Date.now();
        if (now - lastAddedTimeRef.current > SENTENCE_ADD_COOLDOWN) {
          const sentenceParts = sentence.trim().split(/\s+/);
          const lastWord = sentenceParts[sentenceParts.length - 1];

          setSentence(prev => lastWord === result ? prev : `${prev.trim()} ${result}`);
          lastAddedTimeRef.current = now;
        }
      }
    }
  }, [collector, predictor, tts, sentence]);

  const { status: cameraStatus } = useLandmarker({
    videoRef,
    canvasRef,
    onFrameProcessed: handleFrame
  });

  const handleTrainModel = () => {
    const newTrainingData = collector.trainModel();
    if (newTrainingData) {
      predictor.setModelType("ml");
      predictor.setDetectedLabel("Model Trained!");
      predictor.loadMLModel(newTrainingData);
      alert(`Trained model with ${newTrainingData.length} samples. Switched to ML Mode.`);
    }
  };

  return (
    <div className="min-h-screen w-full flex flex-col items-center py-6 px-4 bg-background text-foreground transition-colors font-sans overflow-x-hidden">

      {/* Header */}
      <div className="flex items-center justify-between w-full max-w-[1600px] mb-6 px-2">
        <div className="flex flex-col">
          <h1 className="text-3xl font-bold tracking-tight">GAMAY — ASL Trainer</h1>
          <p className="text-muted-foreground">Learn ASL with real-time feedback</p>
        </div>

        <div className="flex gap-2">
          <Button variant="outline" onClick={() => setIsFeedbackOpen(true)}>
            <Info className="w-4 h-4 mr-2" /> Feedback
          </Button>
          <ModeToggle />
        </div>
      </div>

      {/* Main Grid Layout - 4 Columns */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6 w-full max-w-[1600px]">

        {/* Column 1: Hand Info & ML Controls */}
        <div className="xl:col-span-1 flex flex-col gap-4">
          <GestureInfoCard label={predictor.detectedLabel} />

          {/* ML Model Controls */}
          <Card className="p-4 flex flex-col gap-4">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold flex items-center gap-2">
                <Brain className="w-4 h-4 text-purple-600" /> ML Model
              </h3>
              <Button
                variant={predictor.modelType === 'geometric' ? 'secondary' : 'default'}
                size="sm"
                onClick={() => predictor.setModelType(predictor.modelType === 'geometric' ? 'ml' : 'geometric')}
                disabled={predictor.mlDataSize < K}
              >
                {predictor.modelType === 'geometric' ? 'Geometric' : 'k-NN Active'}
              </Button>
            </div>
            <Separator />
            <div className="space-y-2 text-sm">
              <p className="text-muted-foreground">Training Samples Loaded: <span className="font-bold text-primary">{predictor.mlDataSize}</span></p>
              <p className="text-muted-foreground">Active Mode: <span className="font-bold">{predictor.modelType.toUpperCase()}</span></p>
              <p className={`text-xs ${predictor.mlDataSize < K ? 'text-yellow-600' : 'text-muted-foreground'}`}>
                k-NN Mode requires a minimum of {K} samples to operate.
              </p>
            </div>
          </Card>

          {/* Camera Settings */}
          <Card className="p-4 flex flex-col gap-4 flex-grow">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold flex items-center gap-2">
                <Settings2 className="w-4 h-4" /> Camera
              </h3>
              <Button variant="ghost" size="icon" onClick={resetCamSettings} title="Reset">
                <RefreshCcw className="w-3 h-3" />
              </Button>
            </div>
            <div className="space-y-4">
              {['brightness', 'contrast', 'saturation'].map((s) => (
                <div key={s} className="space-y-1">
                  <div className="flex justify-between text-xs text-muted-foreground uppercase">
                    <span>{s}</span>
                    <span>{camSettings[s as keyof CameraSettings]}%</span>
                  </div>
                  <Slider
                    min={s === 'saturation' ? "0" : "50"} max="200"
                    value={camSettings[s as keyof CameraSettings]}
                    onChange={(e) => setCamSettings(prev => ({ ...prev, [s]: parseInt(e.target.value) }))}
                  />
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Columns 2 & 3: Video Stage & Controls */}
        <div className="xl:col-span-2 flex flex-col items-center gap-4">
          <VideoStage
            videoRef={videoRef}
            canvasRef={canvasRef}
            detectedLabel={predictor.detectedLabel}
            confidence={predictor.confidence}
            isLocked={predictor.isLocked}
            settings={camSettings}
          />

          {/* Status Bar */}
          <div className="flex gap-6 text-sm bg-muted/50 px-6 py-3 rounded-full border shadow-sm items-center w-full justify-center flex-wrap">
            <span className="flex items-center gap-2">
              <ScanFace className="w-4 h-4 text-primary" />
              {cameraStatus}
            </span>
            <span className="text-muted-foreground hidden sm:inline">|</span>
            <span className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-500" />
              Confidence: <span className="font-bold">{predictor.confidence}%</span>
            </span>
            <span className="text-muted-foreground hidden sm:inline">|</span>
            <span className="font-mono text-purple-600 font-bold flex items-center gap-2">
              {predictor.isLocked ? <Lock className="w-3 h-3" /> : null}
              {predictor.debugStatus}
            </span>
          </div>

          {/* Main Controls */}
          <div className="w-full flex flex-col gap-4 p-4 border rounded-lg bg-card/50">
            <div className="flex justify-between items-center">
              <h3 className="font-semibold text-lg flex items-center gap-2">
                Audio Settings
              </h3>
              <div className="flex gap-2 items-center">
                <Button
                  variant="outline"
                  onClick={tts.toggle}
                  className={tts.enabled ? "text-green-600 border-green-200" : "text-muted-foreground"}
                >
                  {tts.enabled ? <><Volume2 className="w-4 h-4 mr-2" /> Speech On</> : <><VolumeX className="w-4 h-4 mr-2" /> Speech Off</>}
                </Button>
              </div>
            </div>

            <Separator />

            {/* Advanced / Data Collection Toggle */}
            <details className="text-sm text-muted-foreground cursor-pointer">
              <summary className="hover:text-primary transition-colors font-medium text-base p-1 flex items-center justify-between">
                Advanced: Data Collection & ML Training
                <span className={`text-xs px-2 py-1 rounded-full font-semibold ${collector.collecting ? 'bg-red-500/20 text-red-500' : 'bg-green-500/20 text-green-500'}`}>
                  {collector.collecting ? 'RECORDING' : 'IDLE'}
                </span>
              </summary>

              <div className="mt-4 space-y-3 p-4 border rounded bg-background">
                <h4 className="text-sm font-semibold text-foreground">1. Record New Data Samples</h4>
                <div className="flex gap-2 items-center">
                  <Input
                    placeholder="Sign Label (e.g., 'Mom')"
                    className="w-full"
                    value={inputLabel}
                    onChange={(e) => setInputLabel(e.target.value)}
                    disabled={collector.collecting}
                  />
                  <Button
                    variant={collector.collecting ? "destructive" : "secondary"}
                    onClick={() => collector.collecting ? collector.stopCollecting() : collector.startCollecting(inputLabel)}
                    disabled={!inputLabel}
                  >
                    {collector.collecting ? `Stop Recording` : "Start Recording"}
                  </Button>
                </div>
                <p className="text-xs text-muted-foreground italic">
                  Collecting at max 5 samples per second. Samples collected: <span className="font-bold text-primary">{collector.sampleCount}</span>
                </p>

                <Separator />

                <h4 className="text-sm font-semibold text-foreground">2. Train & Manage Model</h4>
                <div className="flex justify-between items-center text-xs">
                  <Button
                    variant="default"
                    size="sm"
                    onClick={handleTrainModel}
                    disabled={collector.sampleCount === 0 || collector.collecting}
                    className="w-full mr-2"
                  >
                    <Brain className="w-3 h-3 mr-2" /> Train Model with New Data
                  </Button>
                </div>

                <Separator />

                <h4 className="text-sm font-semibold text-foreground">3. Save/Load Full Dataset</h4>
                <div className="flex justify-between items-center text-xs">
                  <Button variant="outline" size="sm" onClick={collector.saveDataset} title="Download Full ML Data">
                    <RefreshCcw className="w-4 h-4 mr-2 rotate-180" /> Save Full Dataset
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => collector.loadDataset(predictor.loadMLModel)} title="Load ML Data">
                    <RefreshCcw className="w-4 h-4 mr-2" /> Load Dataset
                  </Button>
                </div>
              </div>
            </details>
          </div>
        </div>

        {/* Column 4: Translation ON TOP of Library (Stacked) */}
        <div className="xl:col-span-1 flex flex-col gap-4 h-[700px] xl:h-[calc(100vh-140px)] sticky top-6">

          {/* 1. Sentence Display (On Top) */}
          <div className="flex-shrink-0">
            <SentenceDisplay
              sentence={sentence}
              onClear={() => setSentence("")}
            />
          </div>

          {/* 2. NEW VOICE TO SIGN PLAYER - NOW BIGGER AND MATCHING DESIGN */}
          <div className="flex-shrink-0 h-auto min-h-[400px] flex-grow">
            <VoiceSignPlayer
              transcript={speech.transcript}
              isListening={speech.isListening}
              toggleListen={speech.toggleListen}
            />
          </div>

          {/* 3. Searchable Library (Below Translation) */}
          <Card className="flex flex-col flex-grow overflow-hidden border-primary/20 shadow-lg min-h-[250px]">
            {/* Search Header */}
            <div className="p-4 border-b bg-muted/30 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="font-bold text-sm uppercase flex items-center gap-2 text-primary">
                  <List className="w-4 h-4" /> Sign Library
                </h3>
                <span className="text-[10px] font-bold bg-primary/10 text-primary px-2 py-0.5 rounded-full">
                  {filteredLibrary.length} Items
                </span>
              </div>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search signs..."
                  className="pl-9 bg-background/50"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                />
              </div>
            </div>

            {/* Scrollable List */}
            <div className="flex-grow overflow-y-auto p-2 space-y-2 custom-scrollbar bg-muted/5">
              {filteredLibrary.length > 0 ? (
                filteredLibrary.map(([key, item]) => (
                  <div
                    key={key}
                    className={`p-3 rounded-xl border transition-all flex items-center gap-3 cursor-default
                      ${predictor.detectedLabel === key
                        ? 'bg-primary border-primary text-white shadow-lg translate-x-1'
                        : 'bg-card hover:border-primary/40 hover:bg-muted/30'}`}
                  >
                    <span className="text-2xl">{item.emoji}</span>
                    <div className="flex flex-col min-w-0">
                      <span className="font-bold text-sm truncate">{item.title}</span>
                      <span className="text-[10px] line-clamp-1 opacity-70">
                        {item.description}
                      </span>
                    </div>
                  </div>
                ))
              ) : (
                <div className="flex flex-col items-center justify-center h-full opacity-30 text-center p-6">
                  <Search className="w-12 h-12 mb-2" />
                  <p className="text-sm font-bold uppercase">No signs found</p>
                </div>
              )}
            </div>
          </Card>
        </div>
      </div>

      {/* FEEDBACK MODAL IMPLEMENTATION */}
      {isFeedbackOpen && (
        <FeedbackModal onClose={() => setIsFeedbackOpen(false)}>
          <FeedbackSection />
        </FeedbackModal>
      )}

    </div>
  );
}

// --- Sub-Components (Unchanged) ---

function FeedbackModal({ children, onClose }: { children: React.ReactNode, onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-background rounded-xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto transform scale-95 animate-in fade-in zoom-in duration-300"
        onClick={(e) => e.stopPropagation()}
      >
        {children}
        <div className="p-4 flex justify-end">
          <Button variant="secondary" onClick={onClose}>Close</Button>
        </div>
      </div>
    </div>
  );
}

function VideoStage({ videoRef, canvasRef, detectedLabel, confidence, isLocked, settings }: VideoStageProps) {
  const videoStyle = settings ? {
    filter: `brightness(${settings.brightness}%) contrast(${settings.contrast}%) saturate(${settings.saturation}%)`,
    opacity: detectedLabel ? '0.7' : '1.0'
  } : {};

  const bgColor = "bg-gray-900";

  return (
    <Card className={`w-full aspect-[4/3] max-w-[640px] shadow-lg border rounded-lg overflow-hidden relative ${bgColor} shrink-0 group`}>
      <video
        ref={videoRef}
        className="absolute inset-0 w-full h-full object-cover transition-all duration-200"
        autoPlay
        playsInline
        muted
        style={videoStyle}
      />
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
      {detectedLabel && (
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 flex flex-col items-center gap-1 animate-in fade-in zoom-in duration-200 pointer-events-none">
          <div className={`px-6 py-2 text-white rounded-full shadow-lg text-xl font-bold whitespace-nowrap flex items-center gap-2 ${isLocked ? "bg-green-600" : "bg-purple-600/90"}`}>
            {isLocked && <Lock className="w-4 h-4 animate-pulse" />}
            {detectedLabel}
          </div>
          <div className="w-32 h-1.5 bg-gray-700/50 rounded-full overflow-hidden backdrop-blur-sm">
            <div
              className={`h-full transition-all duration-300 ${confidence > 80 ? 'bg-green-500' : 'bg-yellow-500'}`}
              style={{ width: `${confidence}%` }}
            />
          </div>
        </div>
      )}
    </Card>
  );
}

function GestureInfoCard({ label }: { label: string }) {
  const info = GESTURE_LIBRARY[label] || {
    emoji: "👋",
    title: "Start Signing",
    description: "Hold your hand up to the camera. Use the settings below if needed."
  };

  if (label && !GESTURE_LIBRARY[label]) {
    info.title = label;
    info.description = `The gesture "${label}" is not in the built-in library. Consider adding it via data collection!`;
    info.emoji = "✨";
  }

  return (
    <Card className="h-full min-h-[300px] p-6 flex flex-col items-center text-center justify-center bg-card transition-all duration-300">
      <div className="text-[80px] md:text-[100px] mb-4 animate-in zoom-in duration-300 select-none drop-shadow-lg">
        {info.emoji}
      </div>
      <h2 className="text-2xl font-bold mb-2 text-primary">{info.title}</h2>
      <div className="flex items-start gap-2 text-left bg-muted/50 p-4 rounded-lg w-full">
        <Info className="w-5 h-5 shrink-0 mt-0.5 text-muted-foreground" />
        <p className="text-sm text-muted-foreground leading-relaxed">
          {info.description}
        </p>
      </div>
    </Card>
  );
}

function SentenceDisplay({ sentence, onClear }: { sentence: string, onClear: () => void }) {
  return (
    <Card className="h-full min-h-[150px] flex flex-col relative overflow-hidden">
      <div className="p-4 border-b flex justify-between items-center bg-muted/20">
        <label className="font-semibold flex items-center gap-2">
          <Activity className="w-4 h-4" /> Live Translation
        </label>
        <Button variant="ghost" size="sm" onClick={onClear} disabled={!sentence} className="h-8">
          <Trash2 className="w-4 h-4 mr-2" /> Clear
        </Button>
      </div>

      <div className="flex-grow p-4 bg-muted/10 overflow-y-auto">
        {sentence ? (
          <p className="text-xl font-mono leading-relaxed break-words whitespace-pre-wrap">
            {sentence}
          </p>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-muted-foreground/50 italic text-center p-4">
            <ScanFace className="w-8 h-8 mb-2 opacity-20" />
            <p className="text-xs">Waiting for input...</p>
          </div>
        )}
      </div>

      {sentence && (
        <div className="p-2 bg-green-500/10 border-t flex items-center justify-center gap-2 text-xs text-green-600 font-medium">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
          Listening Active
        </div>
      )}
    </Card>
  );
}