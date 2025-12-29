import os
os.environ['LIBROSA_CACHE_DIR'] = '/tmp'
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
os.environ['NUMBA_CACHE_DIR'] = '/tmp'

from typing import Dict
from extraction.media_extractor import MediaExtractor
from analysis.video_analyser import VideoAnalyzer
from analysis.audio_analyser import AudioAnalyzer
from analysis.llm_analyser import LLMFusion

class DeepfakeDetectionPipeline:
    """Complete deepfake detection pipeline"""
    
    def __init__(self):
        self.media_extractor = MediaExtractor(frames_per_interval=5)
        self.video_analyzer = VideoAnalyzer()
        self.audio_analyzer = AudioAnalyzer()
        self.llm_fusion = LLMFusion()
    
    def analyze_video(self, video_path: str, interval_duration: float = 2.0) -> Dict:
        
        timeline, video_info = self.media_extractor.extract_all(video_path, interval_duration)
        
        for i, interval in enumerate(timeline):
            video_results = self.video_analyzer.analyze_interval(interval)
            interval['video_results'] = video_results
        
        for i, interval in enumerate(timeline):
            audio_results = self.audio_analyzer.analyze_interval(interval)
            interval['audio_results'] = audio_results
        
        final_report = self.llm_fusion.generate_report(timeline, video_info)
        
        return {
            'video_info': video_info,
            'timeline': timeline,
            'final_report': final_report,
            'summary': self.llm_fusion.generate_report(timeline, video_info)
        }