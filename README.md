# Real-Time Fatigue, Activeness & Productivity Analyzer

A real-time, webcam-based monitoring project that detects **fatigue / activeness / distraction** signals from facial landmarks and logs session metrics (blinks, yawns, efficiency, etc.). It also includes a **Streamlit dashboard** to visualize productivity sessions over time.

## Features

- **Real-time detection (webcam)** using:
  - OpenCV for camera capture
  - MediaPipe Face Mesh for facial landmarks
  - A trained ML classifier (RandomForest) to classify states such as *active*, *fatigued*, *distracted*, *no_face*
- **Session logging** to CSV (timestamps, EAR/MAR, head pose, blink/yawn counts, efficiency, predictions, confidence)
- **Dashboard** with Streamlit + Plotly:
  - Latest session summary
  - Session timeline
  - Efficiency trends
  - Active vs inactive time
  - Blink & yawn analytics
  - Model confidence views

## Repository Contents (key files)

- `real_time_detection.py` — runs webcam detection and logs session data to CSV.
- `streamlit_app.py` — Streamlit dashboard reading the logged CSV and plotting charts.
- `train_model.py` — training script (feature extraction + model training pipeline).
- `model.pkl` — saved model artifact (present in repo).
- `model_performance.txt` — stored evaluation metrics/report.

> Note: `real_time_detection.py` currently tries to load `enhanced_model.pkl` (but the repo contains `model.pkl`). You may need to rename the file or update the script to match.

## Setup

### 1) Clone the repo
```bash
git clone https://github.com/BhumiLodaya/Real-Time-Fatigue-Activeness-Productivity-Analyzer.git
cd Real-Time-Fatigue-Activeness-Productivity-Analyzer
```

### 2) Create and activate a virtual environment (recommended)

**Windows (PowerShell):**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
This repo includes a `requirements.txt`, but it may be empty. If yours is empty, install dependencies manually:

```bash
pip install opencv-python mediapipe numpy pandas scikit-learn imbalanced-learn joblib streamlit plotly tqdm
```

(If you later populate `requirements.txt`, you can use:)
```bash
pip install -r requirements.txt
```

## Run

### Option A — Run real-time webcam detection (data logger)
This will open your webcam and log session metrics into a CSV file.

```bash
python real_time_detection.py
```

**Important:** The script uses a hard-coded CSV output path (currently pointing to a local Windows folder).  
Update it inside `real_time_detection.py` to a relative path such as:

- `data/final_fatigue_dataset.csv`

Create the folder if needed:
```bash
mkdir data
```

### Option B — Run the Streamlit dashboard
The dashboard reads the CSV file and visualizes productivity sessions.

```bash
streamlit run streamlit_app.py
```

**Important:** `streamlit_app.py` also uses a hard-coded CSV path. Update it to match where your CSV is being written.

## Training (optional)

If you want to retrain the model (requires datasets arranged under the expected `data/...` folders):

```bash
python train_model.py
```

The training script extracts features such as:
- EAR (eye aspect ratio)
- MAR (mouth aspect ratio)
- head pose angles (pitch/yaw/roll)
- symmetry / face confidence / derived metrics

…and trains a classifier (RandomForest) with evaluation output similar to what’s in `model_performance.txt`.

## Notes / Known Issues

- **Hard-coded local file paths:** Update CSV input/output paths to be portable.
- **Model filename mismatch:** detection code loads `enhanced_model.pkl` while repo includes `model.pkl`.
- **Webcam permission:** on macOS/Linux you may need to allow camera access for Python.

## Tech Stack

- Python
- OpenCV
- MediaPipe FaceMesh
- scikit-learn (+ imbalanced-learn/SMOTE)
- Streamlit + Plotly
- pandas, numpy, joblib

## License

Add a license if you plan to make this reusable by others (MIT, Apache-2.0, etc.).

## Author

- **BhumiLodaya**
