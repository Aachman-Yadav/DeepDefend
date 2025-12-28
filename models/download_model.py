from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import os

def download_models():
    
    os.makedirs("./models/video_model", exist_ok=True)
    os.makedirs("./models/audio_model", exist_ok=True)
    
    print("Downloading video deepfake detection model...")
    video_model_name = "rushild25/DeepDefend"
    
    video_model = AutoModelForImageClassification.from_pretrained(video_model_name)
    video_processor = AutoImageProcessor.from_pretrained(video_model_name)
    
    video_model.save_pretrained("./models/video_model")
    video_processor.save_pretrained("./models/video_model")
    print("Video model saved to ./models/video_model")
    
    print("\nDownloading audio deepfake detection model...")
    audio_model_name = "mo-thecreator/Deepfake-audio-detection" 
    
    audio_model = AutoModelForAudioClassification.from_pretrained(audio_model_name)
    audio_processor = AutoFeatureExtractor.from_pretrained(audio_model_name)
    
    audio_model.save_pretrained("./models/audio_model")
    audio_processor.save_pretrained("./models/audio_model")
    print("Audio model saved to ./models/audio_model")
    
    print("\nAll models downloaded successfully!")

if __name__ == "__main__":
    download_models()
