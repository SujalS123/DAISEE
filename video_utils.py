import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

mp_face_mesh = mp.solutions.face_mesh

def extract_features_from_video(video_path, target_dim=2836):
    """
    Extracts facial features from video using MediaPipe and maps them to a target_dim vector.
    Approximates the OpenFace (mean, std, min, max) structure.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    all_frames_features = []
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0]
                
                # Extract 468 landmarks (x, y, z)
                # We normalize them to [0, 1] relative to the image
                frame_pts = []
                for pt in landmarks.landmark:
                    frame_pts.extend([pt.x, pt.y, pt.z])
                
                # frame_pts is now 468 * 3 = 1404 features
                # We need to map this to 709 features (before the 4x expansion)
                # We'll take the first 709 or pad with zeros
                if len(frame_pts) > 709:
                    frame_pts = frame_pts[:709]
                else:
                    frame_pts.extend([0] * (709 - len(frame_pts)))
                
                all_frames_features.append(frame_pts)

    cap.release()

    if not all_frames_features:
        return None

    # Convert to numpy array (N_frames, 709)
    data = np.array(all_frames_features)

    # Calculate statistics: Mean, Std, Min, Max
    means = np.mean(data, axis=0)
    stds  = np.std(data, axis=0)
    mins  = np.min(data, axis=0)
    maxs  = np.max(data, axis=0)

    # Concatenate to get 2836 features
    final_vector = np.concatenate([means, stds, mins, maxs])
    
    # Ensure exact dimension
    if len(final_vector) < target_dim:
        final_vector = np.pad(final_vector, (0, target_dim - len(final_vector)))
    elif len(final_vector) > target_dim:
        final_vector = final_vector[:target_dim]

    return final_vector
