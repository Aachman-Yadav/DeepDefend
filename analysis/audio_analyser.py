import torch
import librosa
import numpy as np
from typing import Dict, List
from models.load_models import model_loader

class AudioAnalyzer:
    """Analyzes audio chunks for deepfake detection"""
    
    def __init__(self):
        self.model, self.processor = model_loader.load_audio_model()
        self.device = model_loader.get_device()
    
    def predict_deepfake(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """Predict if audio chunk is deepfake"""
        
        min_length = sample_rate * 1
        if len(audio) < min_length:
            audio = np.pad(audio, (0, min_length - len(audio)))
        
        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        
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
    
    def analyze_spectrogram(self, audio: np.ndarray, sample_rate: int, fake_score: float) -> Dict:
        """Analyze audio with adaptive thresholds based on fake_score"""
        try:
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
            
            suspicious_regions = self._identify_audio_anomalies(
                spectral_centroid, spectral_rolloff, zero_crossing_rate, mfcc, fake_score
            )
            
            return {
                'regions': suspicious_regions,
                'spectral_features': {
                    'avg_spectral_centroid': round(float(np.mean(spectral_centroid)), 2),
                    'avg_spectral_rolloff': round(float(np.mean(spectral_rolloff)), 2),
                    'avg_zero_crossing_rate': round(float(np.mean(zero_crossing_rate)), 3),
                    'mfcc_variance': round(float(np.var(mfcc)), 3)
                }
            }
        
        except Exception as e:
            if fake_score > 0.6:
                return {
                    'regions': ['voice_synthesis_detected', 'audio_artifacts'],
                    'spectral_features': {}
                }
            else:
                return {
                    'regions': ['no_suspicious_patterns'],
                    'spectral_features': {}
                }
    
    def _identify_audio_anomalies(self, spectral_centroid: np.ndarray, spectral_rolloff: np.ndarray, zero_crossing: np.ndarray, mfcc: np.ndarray, fake_score: float) -> List[str]:
        suspicious_regions = []
        
        if fake_score > 0.7:
            pitch_low, pitch_high = 200, 6000
            mfcc_threshold = 25
            zcr_low, zcr_high = 0.02, 0.25
            rolloff_threshold = 3000
            centroid_jump = 800
        elif fake_score > 0.5:
            pitch_low, pitch_high = 250, 5500
            mfcc_threshold = 28
            zcr_low, zcr_high = 0.025, 0.22
            rolloff_threshold = 2700
            centroid_jump = 900
        else:
            pitch_low, pitch_high = 300, 5000
            mfcc_threshold = 30
            zcr_low, zcr_high = 0.03, 0.20
            rolloff_threshold = 2500
            centroid_jump = 1000
        
        pitch_variance = np.var(spectral_centroid)
        if pitch_variance < pitch_low:
            suspicious_regions.append('monotone_voice')
        elif pitch_variance > pitch_high:
            suspicious_regions.append('erratic_pitch')
        
        mfcc_var = np.var(mfcc)
        if mfcc_var < mfcc_threshold:
            suspicious_regions.append('voice_synthesis_artifacts')
        
        zcr_mean = np.mean(zero_crossing)
        if zcr_mean > zcr_high:
            suspicious_regions.append('high_frequency_noise')
        elif zcr_mean < zcr_low:
            suspicious_regions.append('overly_smooth_audio')
        
        rolloff_std = np.std(spectral_rolloff)
        if rolloff_std > rolloff_threshold:
            suspicious_regions.append('spectral_artifacts')
        
        centroid_diff = np.diff(spectral_centroid)
        if len(centroid_diff) > 0 and np.max(np.abs(centroid_diff)) > centroid_jump:
            suspicious_regions.append('audio_splicing')
        
        if np.std(spectral_centroid) < 50:
            suspicious_regions.append('unnatural_consistency')
        
        if fake_score > 0.6 and len(suspicious_regions) == 0:
            suspicious_regions.append('general_audio_manipulation')
        
        return suspicious_regions if suspicious_regions else ['no_suspicious_patterns']
    
    def analyze_interval(self, interval_data: Dict) -> Dict:
        """Analyze audio for a single interval"""
        audio_data = interval_data['audio_data']
        
        if not audio_data or not audio_data.get('has_audio', False):
            return {
                'interval_id': interval_data['interval_id'],
                'interval': interval_data['interval'],
                'fake_score': 0.0,
                'confidence': 0.0,
                'suspicious_regions': ['no_audio'],
                'has_audio': False,
                'spectral_features': {}
            }
        
        audio = audio_data['audio']
        sample_rate = audio_data['sample_rate']
        
        prediction = self.predict_deepfake(audio, sample_rate)
        
        spectrogram_analysis = self.analyze_spectrogram(
            audio, sample_rate, prediction['fake_score']
        )
        
        return {
            'interval_id': interval_data['interval_id'],
            'interval': interval_data['interval'],
            'fake_score': prediction['fake_score'],
            'confidence': prediction['confidence'],
            'suspicious_regions': spectrogram_analysis['regions'],
            'has_audio': True,
            'spectral_features': spectrogram_analysis['spectral_features']
        }