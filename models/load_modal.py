from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch

class ModelLoader:
    
    _instance = None
    _video_model = None
    _video_processor = None
    _audio_model = None
    _audio_processor = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance
    
    def load_video_model(self):
        if self._video_model is None:
            self._video_model = AutoModelForImageClassification.from_pretrained("./models/video_model")
            self._video_processor = AutoImageProcessor.from_pretrained("./models/video_model")
            
            self._video_model.eval()
            
            if torch.cuda.is_available():
                self._video_model = self._video_model.cuda()
                
            print("Video model loaded!")
        
        return self._video_model, self._video_processor
    
    def load_audio_model(self):
        if self._audio_model is None:
            self._audio_model = AutoModelForAudioClassification.from_pretrained("./models/audio_model")
            self._audio_processor = AutoFeatureExtractor.from_pretrained("./models/audio_model")
            
            self._audio_model.eval()
            
            if torch.cuda.is_available():
                self._audio_model = self._audio_model.cuda()
                
            print("Audio model loaded!")
        
        return self._audio_model, self._audio_processor
    
    def get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

model_loader = ModelLoader()