import os
import cv2
import csv
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import joblib
import mediapipe as mp
import math

# Paths to your datasets
dataset_paths = {
    'fatigued': 'data/yawn_dataset/Yawn',
    'active': 'data/yawn_dataset/NoYawn',
    'distracted': 'data/cew_dataset/Closed_Eyes',
    'active2': 'data/cew_dataset/Open_Eyes',
}

# Setup Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Improved feature extraction
def eye_aspect_ratio(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio (EAR)"""
    A = np.linalg.norm(landmarks[eye_indices[1]] - landmarks[eye_indices[5]])
    B = np.linalg.norm(landmarks[eye_indices[2]] - landmarks[eye_indices[4]])
    C = np.linalg.norm(landmarks[eye_indices[0]] - landmarks[eye_indices[3]])
    return (A + B) / (2.0 * C) if C != 0 else 0

def mouth_aspect_ratio(landmarks):
    """Calculate Mouth Aspect Ratio (MAR)"""
    # Vertical mouth landmarks
    A = np.linalg.norm(landmarks[13] - landmarks[14])  # Upper lip to lower lip
    B = np.linalg.norm(landmarks[12] - landmarks[15])  # Another vertical measurement
    # Horizontal mouth landmarks
    C = np.linalg.norm(landmarks[78] - landmarks[308])  # Left corner to right corner
    return (A + B) / (2.0 * C) if C != 0 else 0

def perclos(ear_values, threshold=0.25):
    """Calculate PERCLOS (Percentage of Eyelid Closure)"""
    if len(ear_values) == 0:
        return 0
    closed_frames = sum(1 for ear in ear_values if ear < threshold)
    return closed_frames / len(ear_values)

def estimate_head_pose(landmarks):
    """Estimate head pose using facial landmarks"""
    # Key facial points for pose estimation
    nose_tip = landmarks[1]
    chin = landmarks[17]
    left_eye_corner = landmarks[33]
    right_eye_corner = landmarks[263]
    left_mouth_corner = landmarks[61]
    right_mouth_corner = landmarks[291]
    
    # Calculate angles
    eye_center = (left_eye_corner + right_eye_corner) / 2
    mouth_center = (left_mouth_corner + right_mouth_corner) / 2
    
    # Yaw (left-right rotation)
    eye_width = np.linalg.norm(right_eye_corner - left_eye_corner)
    left_eye_dist = np.linalg.norm(nose_tip - left_eye_corner)
    right_eye_dist = np.linalg.norm(nose_tip - right_eye_corner)
    yaw = math.atan2(right_eye_dist - left_eye_dist, eye_width) * 180 / math.pi
    
    # Pitch (up-down rotation)
    nose_to_eye = np.linalg.norm(nose_tip - eye_center)
    nose_to_mouth = np.linalg.norm(nose_tip - mouth_center)
    pitch = math.atan2(nose_to_eye, nose_to_mouth) * 180 / math.pi - 90
    
    # Roll (tilt)
    eye_slope = (right_eye_corner[1] - left_eye_corner[1]) / (right_eye_corner[0] - left_eye_corner[0])
    roll = math.atan(eye_slope) * 180 / math.pi
    
    return pitch, yaw, roll

def calculate_blink_frequency(ear_values, threshold=0.25):
    """Calculate blink frequency"""
    if len(ear_values) < 2:
        return 0
    
    blinks = 0
    was_closed = False
    
    for ear in ear_values:
        is_closed = ear < threshold
        if was_closed and not is_closed:
            blinks += 1
        was_closed = is_closed
    
    return blinks / len(ear_values)  # Normalized blink rate

def extract_enhanced_features(image):
    """Extract comprehensive facial features for fatigue detection"""
    h, w = image.shape[:2]
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        return None, False  # Return None and face_detected flag

    face = results.multi_face_landmarks[0]
    coords = np.array([(lm.x * w, lm.y * h) for lm in face.landmark])

    # Eye landmark indices (MediaPipe)
    left_eye_idx = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    right_eye_idx = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    # Calculate EAR for both eyes
    left_ear = eye_aspect_ratio(coords, [33, 160, 158, 133, 153, 144])
    right_ear = eye_aspect_ratio(coords, [362, 385, 387, 263, 373, 380])
    avg_ear = (left_ear + right_ear) / 2.0
    
    # Calculate MAR
    mar = mouth_aspect_ratio(coords)
    
    # Head pose estimation
    pitch, yaw, roll = estimate_head_pose(coords)
    
    # Additional features
    eye_distance = np.linalg.norm(coords[33] - coords[263])  # Distance between eye corners
    face_symmetry = abs(np.linalg.norm(coords[33] - coords[1]) - np.linalg.norm(coords[263] - coords[1]))
    
    # Eye opening metrics
    left_eye_height = np.linalg.norm(coords[159] - coords[145])
    right_eye_height = np.linalg.norm(coords[386] - coords[374])
    avg_eye_height = (left_eye_height + right_eye_height) / 2.0
    
    # Mouth metrics
    mouth_width = np.linalg.norm(coords[61] - coords[291])
    mouth_height = np.linalg.norm(coords[13] - coords[14])
    
    # Face detection confidence (based on feature quality)
    face_confidence = min(1.0, (avg_ear + mar + avg_eye_height/100) / 3.0)
    
    features = [
        avg_ear,           # Average Eye Aspect Ratio
        mar,               # Mouth Aspect Ratio
        pitch,             # Head pitch
        yaw,               # Head yaw
        roll,              # Head roll
        eye_distance,      # Distance between eyes
        face_symmetry,     # Face symmetry measure
        avg_eye_height,    # Average eye height
        mouth_width,       # Mouth width
        mouth_height,      # Mouth height
        left_ear,          # Left eye EAR
        right_ear,         # Right eye EAR
        face_confidence    # Face detection confidence
    ]
    
    return features, True  # Return features and face_detected flag

def get_no_face_features():
    """Return default features when no face is detected"""
    return [
        0.0,    # avg_ear
        0.0,    # mar
        0.0,    # pitch
        0.0,    # yaw
        0.0,    # roll
        0.0,    # eye_distance
        0.0,    # face_symmetry
        0.0,    # avg_eye_height
        0.0,    # mouth_width
        0.0,    # mouth_height
        0.0,    # left_ear
        0.0,    # right_ear
        0.0     # face_confidence (0 indicates no face)
    ]

def balance_dataset(X, y, strategy='hybrid'):
    """Balance the dataset using multiple strategies"""
    if strategy == 'oversample':
        # Oversample minority classes
        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
    elif strategy == 'undersample':
        # Undersample majority class
        from collections import Counter
        counter = Counter(y)
        min_samples = min(counter.values())
        
        X_balanced = []
        y_balanced = []
        
        for class_label in counter.keys():
            class_indices = [i for i, label in enumerate(y) if label == class_label]
            sampled_indices = resample(class_indices, n_samples=min_samples, random_state=42)
            X_balanced.extend([X[i] for i in sampled_indices])
            y_balanced.extend([y[i] for i in sampled_indices])
        
        X_balanced = np.array(X_balanced)
        y_balanced = np.array(y_balanced)
    else:  # hybrid
        # Combine oversampling minority and slight undersampling majority
        from collections import Counter
        counter = Counter(y)
        median_samples = sorted(counter.values())[len(counter.values())//2]
        
        X_balanced = []
        y_balanced = []
        
        for class_label in counter.keys():
            class_indices = [i for i, label in enumerate(y) if label == class_label]
            current_count = len(class_indices)
            
            if current_count < median_samples:
                # Oversample minority classes
                target_samples = median_samples
                sampled_indices = resample(class_indices, n_samples=target_samples, random_state=42)
            else:
                # Keep majority class as is or slightly undersample
                target_samples = min(current_count, median_samples * 2)
                sampled_indices = resample(class_indices, n_samples=target_samples, random_state=42)
            
            X_balanced.extend([X[i] for i in sampled_indices])
            y_balanced.extend([y[i] for i in sampled_indices])
        
        X_balanced = np.array(X_balanced)
        y_balanced = np.array(y_balanced)
    
    return X_balanced, y_balanced

# Clean old dataset
csv_path = 'data/fatigue_dataset_enhanced.csv'
if os.path.exists(csv_path):
    os.remove(csv_path)
    print(f"[INFO] Old {csv_path} removed.")

# Dataset collection with enhanced features
features = []
labels = []

print("[INFO] Extracting enhanced features from image dataset...")

for label, folder in dataset_paths.items():
    actual_label = 'active' if label == 'active2' else label
    folder_path = os.path.join(folder)
    
    if not os.path.exists(folder_path):
        print(f"[WARNING] Folder {folder_path} does not exist. Skipping...")
        continue
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    successful_extractions = 0

    for img_name in tqdm(image_files, desc=f"Processing {label}"):
        img_path = os.path.join(folder_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        try:
            feats, face_detected = extract_enhanced_features(image)
            if feats and all(not math.isnan(f) and not math.isinf(f) for f in feats):
                features.append(feats)
                labels.append(actual_label)
                successful_extractions += 1
        except Exception as e:
            continue
    
    print(f"[INFO] Successfully extracted {successful_extractions} features from {actual_label}")

# Add "no_face" samples to the dataset
print("[INFO] Adding no-face samples to dataset...")
no_face_samples = 200  # Add samples representing no face detected
no_face_features = get_no_face_features()

for _ in range(no_face_samples):
    # Add slight variations to avoid identical samples
    varied_features = [f + np.random.normal(0, 0.001) for f in no_face_features]
    features.append(varied_features)
    labels.append('no_face')

print(f"[INFO] Added {no_face_samples} no-face samples")

# Save enhanced features to CSV
os.makedirs('data', exist_ok=True)
print(f"[INFO] Saving enhanced features to {csv_path}")

feature_names = [
    'avg_ear', 'mar', 'pitch', 'yaw', 'roll', 'eye_distance', 
    'face_symmetry', 'avg_eye_height', 'mouth_width', 'mouth_height',
    'left_ear', 'right_ear', 'face_confidence', 'label'
]

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(feature_names)
    for row, label in zip(features, labels):
        writer.writerow(row + [label])

print(f"[INFO] Total samples collected: {len(features)}")
print(f"[INFO] Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

# Encode labels and prepare data
le = LabelEncoder()
y = le.fit_transform(labels)
X = np.array(features)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance the dataset
print("[INFO] Balancing dataset...")
X_balanced, y_balanced = balance_dataset(X_scaled, y, strategy='hybrid')
print(f"[INFO] Balanced dataset size: {len(X_balanced)}")
print(f"[INFO] Balanced class distribution: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# Enhanced model with better parameters
print("[INFO] Training enhanced Random Forest model...")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'  # Handle remaining class imbalance
)

clf.fit(X_train, y_train)

# Cross-validation
cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
print(f"[INFO] Cross-validation scores: {cv_scores}")
print(f"[INFO] Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Evaluate
print("[INFO] Classification Report:")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
print("\n[INFO] Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature importance
feature_importance = clf.feature_importances_
feature_names_only = feature_names[:-1]  # Remove 'label'
importance_pairs = list(zip(feature_names_only, feature_importance))
importance_pairs.sort(key=lambda x: x[1], reverse=True)

print("\n[INFO] Feature Importance:")
for feature, importance in importance_pairs:
    print(f"{feature}: {importance:.4f}")

# Save enhanced model and components
model_data = {
    'classifier': clf,
    'label_encoder': le,
    'scaler': scaler,
    'feature_names': feature_names_only,
    'no_face_features': no_face_features  # Add reference for no-face detection
}

joblib.dump(model_data, 'enhanced_model.pkl')
print("[✅] Enhanced model saved as enhanced_model.pkl")

# Save model performance metrics
with open('model_performance.txt', 'w') as f:
    f.write("Model Performance Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Cross-validation scores: {cv_scores}\n")
    f.write(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, y_pred, target_names=le.classes_))
    f.write("\n\nFeature Importance:\n")
    for feature, importance in importance_pairs:
        f.write(f"{feature}: {importance:.4f}\n")

print("[✅] Performance report saved as model_performance.txt")

# Test function for real-time use
def predict_fatigue_state(image, model_data):
    """
    Predict fatigue state from image with proper no-face handling
    """
    feats, face_detected = extract_enhanced_features(image)
    
    if not face_detected:
        return {
            'prediction': 'no_face',
            'confidence': 0.0,
            'probabilities': {},
            'face_detected': False
        }
    
    # Scale features
    feats_scaled = model_data['scaler'].transform([feats])
    
    # Predict
    prediction = model_data['classifier'].predict(feats_scaled)[0]
    probabilities = model_data['classifier'].predict_proba(feats_scaled)[0]
    
    # Get class names
    class_names = model_data['label_encoder'].classes_
    prob_dict = dict(zip(class_names, probabilities))
    
    predicted_class = model_data['label_encoder'].inverse_transform([prediction])[0]
    confidence = max(probabilities)
    
    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': prob_dict,
        'face_detected': True
    }

print("\n[✅] Training completed! Use predict_fatigue_state() function for real-time detection.")