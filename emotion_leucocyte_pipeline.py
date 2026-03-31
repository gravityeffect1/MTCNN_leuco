
import cv2
import numpy as np
import os
import time
import threading
from collections import Counter
from fer.fer import FER


class EmotionLeucocytePipeline:
    CANVAS_W, CANVAS_H = 1280, 720
    HALF_W = CANVAS_W // 2                         

    SUBMIT_EVERY_N_FRAMES = 3

    EMOTION_BUFFER_SIZE = 12

    EMOTION_MAP = {
        "happy":   "leucocyte_1.jpg",
        "sad":     "leucocyte_2.jpg",
        "angry":   "leucocyte_3.jpg",
        "neutral": "leucocyte_4.jpg",
    }

    SCORE_BLEND: dict[str, dict[str, float]] = {
        "happy":   {"happy": 1.0, "surprise": 0.20},
        "sad":     {"sad": 1.0,   "disgust": 0.35, "fear": 0.15},
        "angry":   {"angry": 1.0, "disgust": 0.25},
        "neutral": {"neutral": 1.0},
    }

    SCORE_CALIBRATION: dict[str, float] = {
        "happy":   1.0,
        "sad":     1.6,
        "angry":   1.1,
        "neutral": 0.75,
    }

    MIN_CONFIDENCE_PER_EMOTION: dict[str, float] = {
        "happy":   0.35,
        "sad":     0.18,
        "angry":   0.28,
        "neutral": 0.20,
    }

    MIN_FACE_PX = 60

    BBOX_COLOR   = (0, 255, 128)
    LABEL_BG     = (30, 30, 30)
    LABEL_FG     = (255, 255, 255)
    PANEL_BG     = (20, 20, 20)
    DIVIDER_CLR  = (80, 80, 80)

    def __init__(self, image_dir: str = "images"):
        self.image_dir = image_dir
        self.detector = FER(mtcnn=True)
        self.cap = None
        self.current_emotion: str = "neutral"
        self.frame_counter: int = 0
        self._image_cache: dict[str, np.ndarray] = {}

        self._running = False
        self._pending_frame = None
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._result = None
        self._result_lock = threading.Lock()
        self._emotion_buffer: list[str] = []

        print("[INIT] FER detector loaded (MTCNN backend).")
        print(f"[INIT] Medical image dir : '{os.path.abspath(image_dir)}'")
        print(f"[INIT] Smoothing buffer  : {self.EMOTION_BUFFER_SIZE} frames")
        print(f"[INIT] Min confidence    : {self.MIN_CONFIDENCE_PER_EMOTION}")

    @staticmethod
    def _letterbox(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas

    def get_medical_image(self, emotion_label: str) -> np.ndarray:
    
        emotion_label = emotion_label.lower()
        filename = self.EMOTION_MAP.get(emotion_label, "leucocyte_4.jpg")

        filepath = os.path.join(self.image_dir, filename)
        if filepath in self._image_cache:
            return self._image_cache[filepath].copy()

        if os.path.isfile(filepath):
            img = cv2.imread(filepath)
            if img is not None:
                img = cv2.resize(img, (self.HALF_W, self.CANVAS_H))
                self._image_cache[filepath] = img
                return img.copy()
            else:
                print(f"[WARN] Could not decode {filepath}")

    def _blend_and_calibrate(self, raw: dict) -> dict:
    
        blended = {}
        for emotion, sources in self.SCORE_BLEND.items():
            blended[emotion] = sum(raw.get(src, 0.0) * w
                                   for src, w in sources.items())
        calibrated = {e: blended[e] * self.SCORE_CALIBRATION[e]
                      for e in blended}
        total = sum(calibrated.values()) or 1.0
        return {e: v / total for e, v in calibrated.items()}

    def _apply_clahe(self, frame: np.ndarray, clahe) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = clahe.apply(l_ch)
        return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    def _analyse_frame(self, frame: np.ndarray, clahe):
    
        enhanced = self._apply_clahe(frame, clahe)
        results  = self.detector.detect_emotions(enhanced)
        if not results:
            return None, None
        best = max(results, key=lambda r: r["box"][2] * r["box"][3])
        return self._blend_and_calibrate(best["emotions"]), best

    def _weighted_vote(self, buffer: list) -> str:
  
        weights: dict = {}
        decay = 0.85
        for i, emotion in enumerate(reversed(buffer)):
            w = decay ** i
            weights[emotion] = weights.get(emotion, 0.0) + w
        return max(weights, key=weights.__getitem__)


    def _analysis_worker(self):
    
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        while self._running:
            triggered = self._frame_ready.wait(timeout=0.5)
            if not triggered:
                continue
            self._frame_ready.clear()

            with self._frame_lock:
                frame = self._pending_frame
                self._pending_frame = None
            if frame is None:
                continue

            scores, best = self._analyse_frame(frame, clahe)
            if scores is None:
                with self._result_lock:
                    self._result = (self.current_emotion, None)
                continue

            x, y, w, h = best["box"]
            if w >= self.MIN_FACE_PX and h >= self.MIN_FACE_PX:
                pad    = int(0.15 * max(w, h))
                fh, fw = frame.shape[:2]
                x1, y1 = max(0, x - pad), max(0, y - pad)
                x2, y2 = min(fw, x + w + pad), min(fh, y + h + pad)
                roi    = frame[y1:y2, x1:x2]

                scale = max(1.0, 96 / min(roi.shape[:2]))
                if scale > 1.0:
                    roi = cv2.resize(roi, (0, 0), fx=scale, fy=scale,
                                     interpolation=cv2.INTER_CUBIC)

                roi_scores, _ = self._analyse_frame(roi, clahe)
                if roi_scores is not None:
                    scores = {e: (scores[e] + roi_scores[e]) / 2.0
                              for e in scores}

            top_emotion = max(scores, key=scores.get)
            if scores[top_emotion] < self.MIN_CONFIDENCE_PER_EMOTION[top_emotion]:
                top_emotion = "neutral"

            self._emotion_buffer.append(top_emotion)
            if len(self._emotion_buffer) > self.EMOTION_BUFFER_SIZE:
                self._emotion_buffer.pop(0)
            smoothed = self._weighted_vote(self._emotion_buffer)

            with self._result_lock:
                self._result = (smoothed, best)

    @staticmethod
    def _draw_bbox(frame, box, emotion, confidence):

        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      EmotionLeucocytePipeline.BBOX_COLOR, 2)

        label = f"{emotion.upper()} ({confidence:.0%})"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)

        label_top = max(0, y - th - baseline - 8)
        cv2.rectangle(frame, (x, label_top), (x + tw + 8, y),
                      EmotionLeucocytePipeline.LABEL_BG, -1)
        cv2.putText(frame, label,
                    (x + 4, max(th + baseline + 4, y - baseline - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    EmotionLeucocytePipeline.LABEL_FG, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_hud(frame, fps: float):
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

    def run(self):
        
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam (index 0).")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[RUN]  Webcam resolution : {actual_w}x{actual_h}")

        window_name = "Emotion-Leucocyte Pipeline"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.CANVAS_W, self.CANVAS_H)

        last_bbox = None
        fps_timer = time.time()
        fps_value = 0.0
        fps_alpha = 0.1

        self._running = True
        worker = threading.Thread(target=self._analysis_worker, daemon=True)
        worker.start()

        print("[RUN]  Pipeline started - press 'q' or ESC to quit.\n")

        try:
            while True:
                ret, raw_frame = self.cap.read()
                if not ret:
                    print("[ERR]  Frame grab failed; exiting.")
                    break

                cam_frame = self._letterbox(raw_frame, self.HALF_W, self.CANVAS_H)

                self.frame_counter += 1
                if self.frame_counter % self.SUBMIT_EVERY_N_FRAMES == 0:
                    with self._frame_lock:
                        self._pending_frame = cam_frame.copy()
                    self._frame_ready.set()

                with self._result_lock:
                    if self._result is not None:
                        self.current_emotion, last_bbox = self._result
                        self._result = None

                if last_bbox is not None:
                    scores = {k: last_bbox["emotions"].get(k, 0.0)
                              for k in ("happy", "sad", "angry", "neutral")}
                    conf = scores[self.current_emotion]
                    self._draw_bbox(cam_frame, last_bbox["box"],
                                    self.current_emotion, conf)

                now = time.time()
                instant = 1.0 / max(now - fps_timer, 1e-6)
                fps_value = fps_alpha * instant + (1 - fps_alpha) * fps_value
                fps_timer = now
                self._draw_hud(cam_frame, fps_value)

                left_panel = self.get_medical_image(self.current_emotion)
                canvas = np.hstack((left_panel, cam_frame))
                cv2.line(canvas, (self.HALF_W, 0), (self.HALF_W, self.CANVAS_H),
                         self.DIVIDER_CLR, 2)

                cv2.imshow(window_name, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[EXIT] Key pressed.")
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("[EXIT] Window closed.")
                    break

        finally:
            self._running = False
            self._frame_ready.set()
            self._cleanup()

    def _cleanup(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("[EXIT] Resources released. Goodbye.")

if __name__ == "__main__":
    pipeline = EmotionLeucocytePipeline(image_dir="Images")
    pipeline.run()
