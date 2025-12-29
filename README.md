# DeepDefend

> **Multi-Modal Deepfake Detection System**  
> Detect AI-generated deepfakes in videos using computer vision and audio analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.5-009688.svg)](https://fastapi.tiangolo.com)

## Overview

DeepDefend is a comprehensive deepfake detection system that combines **video frame analysis** and **audio analysis** to identify AI-generated synthetic media. Using machine learning models and AI-powered evidence fusion, it provides detailed, interval-by-interval analysis with explainable results.

### Why DeepDefend?

- **Multi-Modal Analysis**: Combines video and audio detection for higher accuracy
- **AI-Powered Fusion**: Uses LLM to generate human-readable reports
- **Interval Breakdown**: Shows exactly which parts of the video are suspicious
- **REST API**: Easy integration with any frontend or application

## Features

### Core Detection Capabilities

- **Video Analysis**
  - Frame-by-frame deepfake detection using pre-trained models
  - Face detection and region-specific analysis
  - Suspicious region identification (eyes, mouth, face boundaries)
  - Confidence scoring per frame

- **Audio Analysis**
  - Voice synthesis detection
  - Spectrogram analysis for audio artifacts
  - Frequency pattern recognition
  - Audio splicing detection

- **AI-Powered Reporting**
  - LLM-based evidence fusion (Google Gemini)
  - Natural language explanation of findings
  - Verdict with confidence percentage
  - Timestamped suspicious intervals

### Processing Pipeline

```
Video Input
    ↓
┌───────────────────┐
│ Media Extraction  │ → Extract frames (5 per interval)
│                   │ → Extract audio chunks
└────────┬──────────┘
         │
         ├──────────────────────┬──────────────────────┐
         ▼                      ▼                      ▼
┌─────────────────┐   ┌─────────────────┐   ┌────────────────┐
│ Video Analysis  │   │ Audio Analysis  │   │ Timeline Gen   │
│ • Face detect   │   │ • Spectrogram   │   │ • 2s intervals │
│ • Region scan   │   │ • Voice synth   │   │ • Metadata     │
│ • Fake score    │   │ • Artifacts     │   │                │
└────────┬────────┘   └────────┬────────┘   └────────┬───────┘
         │                     │                     │
         └──────────────┬──────────────┬─────────────┘
                        ▼              ▼
                ┌──────────────────────────┐
                │   LLM Fusion Engine      │
                │ • Combine evidence       │
                │ • Generate verdict       │
                │ • Natural language report│
                └────────────┬─────────────┘
                             ▼
                      Final Report
                    (JSON Response)
```

## Demo

### Live Demo
**API**: [https://deepdefend-api.hf.space](https://huggingface.co/spaces/nishchandel/deepdefend-api)  
**Docs**: [https://deepdefend-api.hf.space/docs](https://nishchandel-deepdefend-api.hf.space/docs)

### Example Analysis

<details>
<summary>Click to see sample output</summary>

```json
{
  "verdict": "DEEPFAKE",
  "confidence": 87.5,
  "overall_scores": {
    "overall_video_score": 0.823,
    "overall_audio_score": 0.756,
    "overall_combined_score": 0.789
  },
  "detailed_analysis": "This video shows strong indicators of deepfake manipulation...",
  "suspicious_intervals": [
    {
      "interval": "4.0-6.0",
      "video_score": 0.891,
      "audio_score": 0.834,
      "video_regions": ["eyes", "mouth"],
      "audio_regions": ["voice_synthesis_artifacts"]
    }
  ],
  "total_intervals_analyzed": 15,
  "video_info": {
    "duration": 12.498711111111112,
    "fps": 29.923085402583734,
    "total_frames": 374,
    "file_size_mb": 31.36
  },
  "analysis_id": "4cd98ea5-8c14-4cae-8da4-689345b0aabc",
  "timestamp": "2025-10-10T23:34:35.724916"
}
```
</details>

## Installation

### Prerequisites

- Python 3.10 or higher
- FFmpeg installed on your system
- Google Gemini API key 

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/deepdefend.git
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download ML models**
```bash
python models/download_model.py
```
*This will download ~2GB of models from Hugging Face*

5. **Configure environment**
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

6. **Run the server**
```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### Docker Setup

```bash
# Build image
docker build -t deepdefend .

# Run container
docker run -p 8000:8000 -e GOOGLE_API_KEY=your_key deepdefend
```

## Tech Stack
### Frontend
- **Frameworks** : Next.js
- **Code** : [nu11Orbit/DeepDefend](https://github.com/nu11Orbit/DeepDefend)
  
### Backend
- **Framework**: FastAPI 0.109.0
- **Server**: Uvicorn
- **ML Framework**: PyTorch 2.3.1
- **Transformers**: Hugging Face Transformers 4.36.2

### ML Models
- **Video Detection**: [rushild25/DeepDefend](https://huggingface.co/rushild25/DeepDefend)
- **Audio Detection**: [mo-thecreator/Deepfake-audio-detection](https://huggingface.co/mo-thecreator/Deepfake-audio-detection)
- **LLM Fusion**: Google Gemini 2.5 Flash

### Processing
- **Computer Vision**: OpenCV, Pillow
- **Audio Processing**: Librosa, SoundFile
- **Video Processing**: FFmpeg

### Deployment
- **Container**: Docker
- **Platforms**: Hugging Face Spaces

## Project Structure

```
deepdefend/
│   
│── extraction/
│   ├── media_extractor.py     # Frame & audio extraction
│   └── timeline_generator.py  # Timeline creation
│
│── analysis/
│   ├── video_analyser.py      # Video deepfake detection
│   ├── audio_analyser.py      # Audio deepfake detection
│   ├── llm_analyser.py        # LLM-based fusion
│   └── prompt.py              # LLM prompts
│ 
│── models/
│   ├── download_model.py      # Model downloader
│   ├── load_models.py         # Model loader
│   ├── video_model/           # (Downloaded)
│   └── audio_model/           # (Downloaded)
│
│── main.py                    # FastAPI application
│── pipeline.py                # Main detection pipeline
│── requirements.txt           # Python dependencies
│── Dockerfile                 # Container configuration
├── .gitignore
└── README.md
```
