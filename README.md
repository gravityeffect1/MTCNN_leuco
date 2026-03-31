# Emotion–Leucocyte Pipeline

> **Real-time facial emotion recognition mapped to leucocyte microscopy images.**
> A  demonstration tool built with OpenCV, FER, and MTCNN.

---

## Overview

This pipeline captures a live webcam feed, classifies the viewer's facial emotion into one of four affective states (Happy · Sad · Angry · Neutral), and immediately displays a corresponding leucocyte (white blood cell) microscopy image on the left half of a split-screen window.


```
┌─────────────────────────────────────────────────┐
│  LEFT PANEL (640 × 720)  │  RIGHT PANEL (640 × 720)  │
│  Leucocyte image matched │  Live webcam feed          │
│  to detected emotion     │  Face bbox + emotion label │
└─────────────────────────────────────────────────┘
```

---

## Scientific Background

| Emotion | Neuroendocrine signal | Leucocyte effect (literature) |
|---------|----------------------|-------------------------------|
| Happy / Positive | ↓ Cortisol, ↑ DHEA | NK cell activity ↑, balanced lymphocyte counts |
| Sad / Depressive | ↑ CRH, HPA axis activation | Lymphopenia, neutrophil predominance |
| Angry / Stress | ↑ Catecholamines, cortisol | Transient leukocytosis, neutrophilia |
| Neutral | Baseline HPA tone | Reference leucocyte morphology |


---

## Repository Structure

```
emotion-leucocyte-pipeline/
├── emotion_leucocyte_pipeline.py   
├── requirements.txt
├── .gitignore
├── images/                       
│   ├── leucocyte_1.jpg             
│   ├── leucocyte_2.jpg             
│   ├── leucocyte_3.jpg            
│   └── leucocyte_4.jpg             
└────────    
```

---

## Quickstart

### 1. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/emotion-leucocyte-pipeline.git
cd emotion-leucocyte-pipeline
pip install -r requirements.txt
```


### 2. Run the pipeline

```bash
python emotion_leucocyte_pipeline.py
```

Press **`q`** or **`ESC`** to quit.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Webcam capture, image rendering, CLAHE preprocessing |
| `fer` | Facial Emotion Recognition (FER2013-based CNN) |
| `tensorflow` | FER backend |
| `mtcnn` | Multi-Task CNN face detector (used by FER) |
| `numpy` | Array operations, image manipulation |

---

## Technical Highlights

- **Two-pass ROI ensemble** — FER runs on the full frame *and* an upscaled face crop; scores are averaged for improved accuracy on small faces.
- **Score blending** — FER's 7 raw emotion classes are collapsed into 4 target states using clinically motivated weights (e.g., `sad` absorbs partial `disgust` and `fear` signal).
- **Dataset bias calibration** — Multipliers correct the known FER2013 tendency to over-predict `neutral` and under-predict `sad`.
- **Recency-weighted temporal smoothing** — An exponentially decaying vote buffer prevents flickering while remaining responsive to genuine emotion changes.
- **Background threading** — FER inference runs in a daemon thread; the main loop always renders at native webcam FPS with the most recent available result.
- **CLAHE preprocessing** — Contrast Limited Adaptive Histogram Equalization in LAB colour space improves detection under variable lighting.

---

## Known Limitations

- Trained on FER2013: predominantly frontal, grayscale, internet-scraped faces — performance degrades on extreme head poses, low-resolution captures, or heavily occluded faces.
- Single-face pipeline: only the largest detected face is analysed.
- Emotion→leucocyte mapping is a **simplified illustration**; real PNI effects are chronic, multifactorial, and not reducible to a single image.

---

## License

MIT License — see `LICENSE`.

---

## Acknowledgements

- [FER library](https://github.com/justinshenk/fer) — Justin Shenk et al.
- [MTCNN](https://github.com/ipazc/mtcnn) — Iván de Paz Centeno
- FER2013 dataset — Goodfellow et al., 2013
