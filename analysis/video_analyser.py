import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Dict
from collections import Counter
from models.load_models import model_loader

class VideoAnalyzer:
    """Simple, reliable video analyzer for hackathon demo"""
    
    def __init__(self):
        self.model, self.processor = model_loader.load_video_model()
        self.device = model_loader.get_device()
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_face(self, frame: np.ndarray) -> Dict:
        """Detect face in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_crop = frame[y:y+h, x:x+w]
            
            return {
                'detected': True,
                'bbox': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)},
                'face_crop': face_crop
            }
        
        return {'detected': False, 'bbox': None, 'face_crop': None}
    
    def predict_deepfake(self, frame: np.ndarray) -> Dict:
        """Predict if frame is deepfake"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        inputs = self.processor(images=pil_img, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
        
        fake_prob = probs[0][1].item() if probs.shape[1] > 1 else probs[0][0].item()
        confidence = max(probs[0]).item()
        
        return {
            'fake_score': round(fake_prob, 3),
            'confidence': round(confidence, 3),
            'label': 'fake' if fake_prob > 0.5 else 'real'
        }
    
    def detect_suspicious_regions(self, face: np.ndarray, fake_score: float) -> List[str]:
        try:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            suspicious_regions = []
            
            regions = {
                'eyes': (int(h*0.25), int(h*0.45), int(w*0.15), int(w*0.85)),
                'nose': (int(h*0.40), int(h*0.65), int(w*0.35), int(w*0.65)),
                'mouth': (int(h*0.60), int(h*0.80), int(w*0.30), int(w*0.70)),
                'forehead': (int(h*0.08), int(h*0.28), int(w*0.25), int(w*0.75)),
                'cheeks': (int(h*0.45), int(h*0.70), int(w*0.15), int(w*0.85)),
                'chin': (int(h*0.75), int(h*0.95), int(w*0.30), int(w*0.70))
            }
            
            for region_name, (y1, y2, x1, x2) in regions.items():
                region = gray[y1:y2, x1:x2]
                
                if region.size == 0:
                    continue
                
                suspicious = False
                
                variance = np.var(region)
                if variance < 200 or variance > 8000:
                    suspicious = True
                
                edges = cv2.Canny(region, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                if edge_density < 0.05:
                    suspicious = True
                
                if fake_score > 0.7 and variance < 400:
                    suspicious = True
                
                if suspicious:
                    suspicious_regions.append(region_name)
            
            left_half = gray[:, :w//2]
            right_half = np.fliplr(gray[:, w//2:])
            
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            symmetry_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
            
            if symmetry_diff < 10:
                suspicious_regions.append('unnatural_symmetry')
            
            return suspicious_regions if suspicious_regions else ['none']
        
        except Exception as e:
            print(f"Region detection error: {e}")
            return ['analysis_error']
    
    def analyze_interval(self, interval_data: Dict) -> Dict:
        """Analyze all frames in an interval"""
        frames_data = interval_data['video_data']
        
        if not frames_data:
            return {
                'interval_id': interval_data['interval_id'],
                'interval': interval_data['interval'],
                'fake_score': 0.0,
                'confidence': 0.0,
                'suspicious_regions': [],
                'face_detected': False,
                'frame_results': []
            }
        
        frame_results = []
        total_fake_score = 0
        faces_detected = 0
        all_regions = []
        
        for frame_data in frames_data:
            frame = frame_data['frame']
            timestamp = frame_data['timestamp']
            
            face_info = self.detect_face(frame)
            
            if face_info['detected']:
                faces_detected += 1
                pred = self.predict_deepfake(face_info['face_crop'])
                regions = self.detect_suspicious_regions(face_info['face_crop'], pred['fake_score'])
            else:
                pred = self.predict_deepfake(frame)
                regions = ['no_face_detected']
            
            total_fake_score += pred['fake_score']
            all_regions.extend(regions)
            
            frame_results.append({
                'timestamp': timestamp,
                'fake_score': pred['fake_score'],
                'confidence': pred['confidence'],
                'face_detected': face_info['detected'],
                'regions': regions
            })
        
        avg_fake_score = total_fake_score / len(frames_data)
        
        region_counts = Counter(all_regions)
        threshold = len(frames_data) * 0.5
        
        consistent_regions = [
            region for region, count in region_counts.items()
            if count >= threshold and region not in ['none', 'no_face_detected', 'analysis_error']
        ]
        
        return {
            'interval_id': interval_data['interval_id'],
            'interval': interval_data['interval'],
            'fake_score': round(avg_fake_score, 3),
            'confidence': round(np.mean([f['confidence'] for f in frame_results]), 3),
            'suspicious_regions': consistent_regions if consistent_regions else list(set(all_regions)),
            'face_detected': faces_detected > 0,
            'frame_count': len(frames_data),
            'frames_with_faces': faces_detected,
            'frame_results': frame_results
        }