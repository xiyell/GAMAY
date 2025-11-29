"use client";

import React, { useRef, useState, useCallback, useEffect } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { Moon, Sun, Info, ScanFace, Activity, Lock, RefreshCcw, Settings2, Trash2 } from "lucide-react";

// --- Types ---

interface Landmark {
  x: number;
  y: number;
  z: number;
}

interface GestureLibraryItem {
  emoji: string;
  title: string;
  description: string;
}

interface Sample {
  label: string;
  landmarks: Landmark[];
}

interface CameraSettings {
  brightness: number;
  contrast: number;
  saturation: number;
}

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

const Button = React.forwardRef<HTMLButtonElement, React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "default" | "destructive" | "outline" | "secondary" | "ghost", size?: "default" | "sm" | "icon" }>(
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
        icon: "h-10 w-10"
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
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      setIsDark(true);
      document.documentElement.classList.add("dark");
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
    <Button variant="outline" size="icon" onClick={toggleTheme}>
      {isDark ? <Moon className="h-[1.2rem] w-[1.2rem]" /> : <Sun className="h-[1.2rem] w-[1.2rem]" />}
    </Button>
  );
}

// --- Gesture Library & Logic ---

const GESTURE_LIBRARY: Record<string, GestureLibraryItem> = {
  "Hello": {
    emoji: "👋",
    title: "Hello",
    description: "Detected a waving motion! Keep your hand open and wave side-to-side."
  },
  "A": {
    emoji: "✊",
    title: "Letter A",
    description: "Make a fist. Ensure your thumb is resting against the side of your index finger, not tucked inside."
  },
  "B": {
    emoji: "✋",
    title: "Letter B",
    description: "Hold your hand open with fingers straight and touching. Thumb tucked slightly."
  },
  "5": {
    emoji: "🖐️",
    title: "Number 5",
    description: "Hold your hand open with fingers spread apart."
  },
  "D": {
    emoji: "☝️",
    title: "Letter D",
    description: "Point your index finger up. Touch your thumb to your middle finger, curling the other fingers down."
  },
  "E / S": {
    emoji: "👊",
    title: "Letter E or S",
    description: "Clench your fingers into a fist. For 'E', curl fingers tight and tuck thumb beneath. For 'S', cross thumb over fingers."
  },
  "F": {
    emoji: "👌",
    title: "Letter F",
    description: "Touch the tip of your index finger to your thumb. Keep the other three fingers spread apart and upright."
  },
  "I": {
    emoji: "ℹ️", 
    title: "Letter I",
    description: "Stick your pinky finger up. Curl all other fingers down and cross your thumb over them."
  },
  "L": {
    emoji: "👆", 
    title: "Letter L",
    description: "Extend your thumb and index finger to make an 'L' shape. Curl the other fingers into your palm."
  },
  "V": {
    emoji: "✌️",
    title: "Letter V",
    description: "Extend your index and middle fingers in a 'V' shape. Curl other fingers and thumb into palm."
  },
  "W": {
    emoji: "👐", 
    title: "Letter W",
    description: "Extend index, middle, and ring fingers. Touch thumb to pinky."
  },
  "Y": {
    emoji: "🤙",
    title: "Letter Y",
    description: "Extend your thumb and pinky. Curl the three middle fingers into your palm."
  },
  "I Love You": {
    emoji: "🤟",
    title: "I Love You",
    description: "Extend thumb, index, and pinky. This combines the letters I, L, and Y."
  }
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
  
  if (fingerIdx === Finger.Thumb) {
    const ip = landmarks[3];
    const wrist = landmarks[0];
    return Math.abs(tip.x - wrist.x) > Math.abs(ip.x - wrist.x) * 1.2;
  }
  
  return tip.y < pip.y; 
};

// --- The Brains ---

const recognizeGeometricGesture = (landmarks: Landmark[]) => {
  if (!landmarks || landmarks.length === 0) return null;

  const wrist = landmarks[0];
  const middleMCP = landmarks[9];
  const handSize = calculateDistance(wrist, middleMCP);
  
  const TOUCH_THRESHOLD = handSize * 0.35; 
  const SPLAY_THRESHOLD = handSize * 0.25;

  const thumbOpen = isExtended(landmarks, Finger.Thumb);
  const indexOpen = isExtended(landmarks, Finger.Index);
  const middleOpen = isExtended(landmarks, Finger.Middle);
  const ringOpen = isExtended(landmarks, Finger.Ring);
  const pinkyOpen = isExtended(landmarks, Finger.Pinky);

  const extendedCount = [indexOpen, middleOpen, ringOpen, pinkyOpen].filter(Boolean).length;

  // Logic Tree
  if (extendedCount === 0) {
     if (!thumbOpen) return "A";
     return "E / S";
  }

  if (indexOpen && !middleOpen && !ringOpen && !pinkyOpen) {
    const thumbTip = landmarks[4];
    const middleTip = landmarks[12];
    if (calculateDistance(thumbTip, middleTip) < TOUCH_THRESHOLD) return "D";

    if (thumbOpen) return "L";
    return "1 / Index";
  }

  if (!indexOpen && !middleOpen && !ringOpen && pinkyOpen) {
    if (thumbOpen) return "Y";
    return "I";
  }

  if (indexOpen && middleOpen && !ringOpen && !pinkyOpen) {
    return "V";
  }

  if (indexOpen && middleOpen && ringOpen && !pinkyOpen) {
     return "W";
  }

  if (!indexOpen && middleOpen && ringOpen && pinkyOpen) {
     const indexTip = landmarks[8];
     const thumbTip = landmarks[4];
     if (calculateDistance(indexTip, thumbTip) < TOUCH_THRESHOLD) return "F";
  }

  if (thumbOpen && indexOpen && !middleOpen && !ringOpen && pinkyOpen) {
    return "I Love You";
  }

  if (indexOpen && middleOpen && ringOpen && pinkyOpen) {
     const d1 = calculateDistance(landmarks[8], landmarks[12]);
     const d2 = calculateDistance(landmarks[12], landmarks[16]);
     
     if (d1 > SPLAY_THRESHOLD || d2 > SPLAY_THRESHOLD) {
        return "5";
     }
     return "B";
  }

  return null;
};

// --- Drawing Helper ---
const drawLandmarks = (ctx: CanvasRenderingContext2D | null, landmarks: Landmark[]) => {
  if(!ctx) return;
  ctx.lineWidth = 2;
  const connections = [[0,1],[1,2],[2,3],[3,4], [0,5],[5,6],[6,7],[7,8], [5,9],[9,10],[10,11],[11,12], [9,13],[13,14],[14,15],[15,16], [13,17],[17,18],[18,19],[19,20], [0,17]];
  
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

// --- Hooks ---

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

  // Memoize predictWebcam to be stable for useEffect
  const predictWebcam = useCallback(() => {
    if (!landmarkerRef.current || !videoRef.current || !canvasRef.current) return;
    
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

    const initLandmarker = async () => {
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
        setStatus("Starting Camera...");
        startCamera();
      } catch (error) {
        console.error("Landmarker Init Error:", error);
        setStatus("Failed to load AI.");
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
                setStatus("Ready: Show Hand");
            };
          }
        } catch (err) {
            console.error("Camera Init Error:", err);
            setStatus("Camera Error (Check Permissions)");
        }
      }
    };

    initLandmarker();

    return () => { 
        if(requestRef.current) cancelAnimationFrame(requestRef.current);
        if (stream) stream.getTracks().forEach(track => track.stop());
    };
  }, [predictWebcam, videoRef]);

  return { status };
}

function usePredictionModel() {
  const [modelType, setModelType] = useState<"none" | "geometric" | "custom">("geometric");
  const [detectedLabel, setDetectedLabel] = useState("");
  const [debugStatus, setDebugStatus] = useState("Waiting...");
  const [confidence, setConfidence] = useState(0);
  const [isLocked, setIsLocked] = useState(false);
  
  const bufferRef = useRef<string[]>([]);
  const wristXBufferRef = useRef<number[]>([]);
  const lastSuccessfulDetectionTime = useRef<number>(0);
  
  const PREDICTION_BUFFER_SIZE = 8;
  const WRIST_BUFFER_SIZE = 20;
  const SCAN_DELAY_MS = 1000;

  const useBuiltInLibrary = () => {
    setModelType("geometric");
  };

  const predict = (inputVector: number[], rawLandmarks: Landmark[]): string | null => {
    if (modelType === "none" || !rawLandmarks || rawLandmarks.length === 0) {
        setDebugStatus("No Hand Detected");
        setConfidence(0);
        setIsLocked(false);
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
    }

    const currentWristX = rawLandmarks[0].x;
    wristXBufferRef.current.push(currentWristX);
    if (wristXBufferRef.current.length > WRIST_BUFFER_SIZE) {
        wristXBufferRef.current.shift();
    }

    if ((result === "B" || result === "5") && wristXBufferRef.current.length === WRIST_BUFFER_SIZE) {
        const xValues = wristXBufferRef.current;
        const travel = xValues.reduce((acc, val, i) => i === 0 ? 0 : acc + Math.abs(val - xValues[i-1]), 0);
        const displacement = Math.abs(xValues[xValues.length-1] - xValues[0]);
        
        if (travel > 0.4 && displacement < 0.3) {
            result = "Hello";
        }
    }

    setDebugStatus(result ? `Match: ${result}` : "Hand Detected - Unknown Pose");
    
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
    detectedLabel,
    setDetectedLabel,
    useBuiltInLibrary,
    predict,
    debugStatus,
    confidence,
    isLocked
  };
}

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

function useDataCollection() {
    const datasetRef = useRef<Sample[]>([]);
    const [collecting, setCollecting] = useState(false);
    const [sampleCount, setSampleCount] = useState(0);
    const activeLabelRef = useRef("");
    
    const startCollecting = (lbl: string) => { 
        if(!lbl) return alert("Enter label"); 
        activeLabelRef.current = lbl; 
        setCollecting(true); 
    }
    const stopCollecting = () => setCollecting(false);
    const addSample = (vec: number[], raw: Landmark[]) => {
        if(collecting) {
            datasetRef.current.push({ label: activeLabelRef.current, landmarks: raw });
            setSampleCount(datasetRef.current.length);
        }
    }
    const saveDataset = () => {
         const blob = new Blob([JSON.stringify(datasetRef.current)], { type: "application/json" });
         const a = document.createElement("a");
         a.href = URL.createObjectURL(blob);
         a.download = "dataset.json";
         a.click();
    }
    return { collecting, sampleCount, startCollecting, stopCollecting, addSample, saveDataset };
}

// --- Main Component ---

export default function ASLRecorder() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  
  const collector = useDataCollection();
  const predictor = usePredictionModel();
  const tts = useTextToSpeech();

  const [inputLabel, setInputLabel] = useState("");
  const [sentence, setSentence] = useState("");
  
  const [camSettings, setCamSettings] = useState<CameraSettings>({
      brightness: 100,
      contrast: 100,
      saturation: 100
  });

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
              setSentence(prev => prev.endsWith(result) ? prev : `${prev} ${result}`);
              lastAddedTimeRef.current = now;
          }
      }
    }
  }, [collector, predictor, tts]); 

  const { status: cameraStatus } = useLandmarker({
    videoRef,
    canvasRef,
    onFrameProcessed: handleFrame
  });

  return (
    <div className="min-h-screen w-full flex flex-col items-center py-6 px-4 bg-background text-foreground transition-colors font-sans overflow-x-hidden">
      
      {/* Header */}
      <div className="flex items-center justify-between w-full max-w-7xl mb-6 px-2">
        <div className="flex flex-col">
            <h1 className="text-3xl font-bold tracking-tight">GAMAY — ASL Trainer</h1>
            <p className="text-muted-foreground">Learn ASL with real-time feedback</p>
        </div>
        <ModeToggle />
      </div>

      {/* Main Grid Layout - 3 Columns */}
      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6 w-full max-w-[1400px]">
          
          {/* Left Column: Hand Info & Camera Settings */}
          <div className="xl:col-span-1 flex flex-col gap-4">
             {/* Identified Hand Card */}
             <GestureInfoCard label={predictor.detectedLabel} />
             
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
                    <div className="space-y-1">
                        <div className="flex justify-between text-xs text-muted-foreground">
                            <span>Brightness</span>
                            <span>{camSettings.brightness}%</span>
                        </div>
                        <Slider 
                            min="50" max="200" 
                            value={camSettings.brightness}
                            onChange={(e) => setCamSettings(s => ({...s, brightness: parseInt(e.target.value)}))}
                        />
                    </div>
                    <div className="space-y-1">
                        <div className="flex justify-between text-xs text-muted-foreground">
                            <span>Contrast</span>
                            <span>{camSettings.contrast}%</span>
                        </div>
                        <Slider 
                            min="50" max="200" 
                            value={camSettings.contrast}
                            onChange={(e) => setCamSettings(s => ({...s, contrast: parseInt(e.target.value)}))}
                        />
                    </div>
                    <div className="space-y-1">
                        <div className="flex justify-between text-xs text-muted-foreground">
                            <span>Saturation</span>
                            <span>{camSettings.saturation}%</span>
                        </div>
                        <Slider 
                            min="0" max="200" 
                            value={camSettings.saturation}
                            onChange={(e) => setCamSettings(s => ({...s, saturation: parseInt(e.target.value)}))}
                        />
                    </div>
                </div>
             </Card>
          </div>

          {/* Center Column: Video Stage & Controls */}
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
                <div className="flex gap-6 text-sm bg-muted/50 px-6 py-3 rounded-full border shadow-sm items-center w-full justify-center">
                    <span className="flex items-center gap-2">
                        <ScanFace className="w-4 h-4 text-primary" />
                        {cameraStatus}
                    </span>
                    <span className="text-muted-foreground">|</span>
                    <span className="flex items-center gap-2">
                         <Activity className="w-4 h-4 text-blue-500" />
                         Confidence: <span className="font-bold">{predictor.confidence}%</span>
                    </span>
                    <span className="text-muted-foreground">|</span>
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
                                {tts.enabled ? "🔊 On" : "🔇 Off"}
                            </Button>
                        </div>
                    </div>
                    
                    <Separator />

                    {/* Advanced / Data Collection Toggle */}
                    <details className="text-sm text-muted-foreground cursor-pointer">
                        <summary className="hover:text-primary transition-colors">Advanced: Data Collection</summary>
                        <div className="mt-4 flex gap-2 items-center p-4 border rounded bg-background">
                            <Input 
                                placeholder="Custom Label" 
                                className="w-full" 
                                value={inputLabel}
                                onChange={(e) => setInputLabel(e.target.value)} 
                            />
                            <Button 
                                variant={collector.collecting ? "destructive" : "secondary"} 
                                onClick={() => collector.collecting ? collector.stopCollecting() : collector.startCollecting(inputLabel)}
                            >
                                {collector.collecting ? "Stop" : "Record"}
                            </Button>
                            <span className="whitespace-nowrap">{collector.sampleCount} samples</span>
                            <Button variant="ghost" size="icon" onClick={collector.saveDataset} title="Download JSON">
                                <RefreshCcw className="w-4 h-4 rotate-180" />
                            </Button>
                        </div>
                    </details>
                </div>
          </div>

          {/* Right Column: Live Translation */}
          <div className="xl:col-span-1 h-full">
             <SentenceDisplay 
                sentence={sentence} 
                onClear={() => setSentence("")} 
             />
          </div>
      </div>
    </div>
  );
}

// --- Sub-Components ---

function VideoStage({ videoRef, canvasRef, detectedLabel, confidence, isLocked, settings }: VideoStageProps) {
  const videoStyle = settings ? {
      filter: `brightness(${settings.brightness}%) contrast(${settings.contrast}%) saturate(${settings.saturation}%)`
  } : {};

  return (
    <Card className="w-full aspect-[4/3] max-w-[640px] shadow-lg border rounded-lg overflow-hidden relative bg-black shrink-0 group">
      <video 
        ref={videoRef} 
        className="absolute inset-0 w-full h-full object-cover opacity-60 transition-all duration-200" 
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
  const info = GESTURE_LIBRARY[label] || { emoji: "👋", title: "Start Signing", description: "Hold your hand up to the camera. Use the settings below if needed." };
  
  if (label && !GESTURE_LIBRARY[label]) {
     info.title = label;
     info.description = "Custom or Unknown Gesture";
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
    <Card className="h-full min-h-[400px] xl:min-h-[auto] flex flex-col relative overflow-hidden">
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
                <ScanFace className="w-12 h-12 mb-2 opacity-20" />
                <p>Do a sign to start translating...</p>
                <p className="text-xs mt-2">Words added every 2s</p>
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