import os
os.environ['HF_HOME'] = '/tmp/hf_home'
os.environ['LIBROSA_CACHE_DIR'] = '/tmp'
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
os.environ['NUMBA_CACHE_DIR'] = '/tmp'

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
import uuid
import shutil
import json
from pathlib import Path
from datetime import datetime
from pipeline import DeepfakeDetectionPipeline

analysis_history = []
MAX_HISTORY = 10

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(
    title="DeepDefend API",
    description="Advanced Deepfake Detection System with Multi-Modal Analysis",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("/tmp/deepdefend_uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        print("Loading DeepDefend Pipeline...")
        pipeline = DeepfakeDetectionPipeline()
    return pipeline

class AnalysisResult(BaseModel):
    verdict: str
    confidence: float
    overall_scores: dict
    detailed_analysis: str
    suspicious_intervals: list
    total_intervals_analyzed: int
    video_info: dict
    analysis_id: str
    timestamp: str

class HistoryItem(BaseModel):
    analysis_id: str
    filename: str
    verdict: str
    confidence: float
    timestamp: str
    video_duration: float

class StatsResponse(BaseModel):
    total_analyses: int
    deepfakes_detected: int
    real_videos: int
    avg_confidence: float
    avg_video_score: float
    avg_audio_score: float

class IntervalDetail(BaseModel):
    interval_id: int
    time_range: str
    video_score: float
    audio_score: float
    verdict: str
    suspicious_regions: dict

def add_to_history(analysis_data: dict):
    """Add analysis to history"""
    history_item = {
        "analysis_id": analysis_data["analysis_id"],
        "filename": analysis_data["filename"],
        "verdict": analysis_data["verdict"],
        "confidence": analysis_data["confidence"],
        "timestamp": analysis_data["timestamp"],
        "video_duration": analysis_data["video_info"]["duration"],
        "overall_scores": analysis_data["overall_scores"]
    }
    
    analysis_history.insert(0, history_item)
    
    if len(analysis_history) > MAX_HISTORY:
        analysis_history.pop()

@app.get("/")
async def root():
    return {
        "service": "DeepDefend API",
        "version": "1.0.0",
        "status": "online",
        "description": "Advanced Multi-Modal Deepfake Detection",
        "features": [
            "Video frame-by-frame analysis",
            "Audio deepfake detection",
            "AI-powered evidence fusion",
            "Frame-level heatmap generation",
            "Interval breakdown analysis",
            "Analysis history tracking"
        ],
        "endpoints": {
            "analyze": "POST /api/analyze",
            "history": "GET /api/history",
            "stats": "GET /api/stats",
            "intervals": "GET /api/intervals/{analysis_id}",
            "compare": "GET /api/compare",
            "health": "GET /api/health"
        }
    }

@app.get("/api/health")
async def health():
    """Health check with system info"""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "total_analyses": len(analysis_history),
        "storage_used_mb": sum(
            f.stat().st_size for f in UPLOAD_DIR.glob('*') if f.is_file()
        ) / (1024 * 1024) if UPLOAD_DIR.exists() else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/analyze", response_model=AnalysisResult)
async def analyze_video(
    file: UploadFile = File(...),
    interval_duration: float = Query(default=2.0, ge=1.0, le=5.0)
):
    """
    Upload and analyze video for deepfakes
    
    Returns complete analysis with:
    - Overall verdict and confidence
    - Video/audio scores
    - Suspicious intervals
    - AI-generated detailed analysis
    """
    
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > 250 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max: 250MB")
    
    if file_size < 100 * 1024:
        raise HTTPException(status_code=400, detail="File too small. Min: 100KB")
    
    analysis_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{analysis_id}{file_ext}"
    
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        pipe = get_pipeline()
        
        print(f"\nAnalyzing: {file.filename}")
        results = pipe.analyze_video(str(video_path), interval_duration)
        
        final_report = results['final_report']
        video_info = results['video_info']
        
        analysis_data = {
            "analysis_id": analysis_id,
            "filename": file.filename,
            "verdict": final_report['verdict'],
            "confidence": final_report['confidence'],
            "overall_scores": final_report['overall_scores'],
            "detailed_analysis": final_report['detailed_analysis'],
            "suspicious_intervals": final_report['suspicious_intervals'],
            "total_intervals_analyzed": final_report['total_intervals_analyzed'],
            "video_info": {
                "duration": video_info['duration'],
                "fps": video_info['fps'],
                "total_frames": video_info['total_frames'],
                "file_size_mb": round(file_size / (1024 * 1024), 2)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        add_to_history(analysis_data)
        