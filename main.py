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

        interval_data = {
            'analysis_id': analysis_id,
            'timeline': [
                {
                    'interval_id': interval['interval_id'],
                    'interval': interval['interval'],
                    'start': interval['start'],
                    'end': interval['end'],
                    'video_results': interval.get('video_results'),
                    'audio_results': interval.get('audio_results')
                }
                for interval in results.get('timeline', [])
            ]
        }
        
        results_path = UPLOAD_DIR / f"{analysis_id}_results.json"
        with open(results_path, 'w') as f:
            json.dump(interval_data, f, indent=2)
        
        return AnalysisResult(**analysis_data)
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        if video_path.exists():
            os.remove(video_path)

@app.get("/api/history", response_model=List[HistoryItem])
async def get_history(limit: int = Query(default=10, ge=1, le=50)):
    """Get recent analysis history"""
    return [
        HistoryItem(
            analysis_id=item["analysis_id"],
            filename=item["filename"],
            verdict=item["verdict"],
            confidence=item["confidence"],
            timestamp=item["timestamp"],
            video_duration=item["video_duration"]
        )
        for item in analysis_history[:limit]
    ]

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get overall statistics"""
    
    if not analysis_history:
        return StatsResponse(
            total_analyses=0,
            deepfakes_detected=0,
            real_videos=0,
            avg_confidence=0.0,
            avg_video_score=0.0,
            avg_audio_score=0.0
        )
    
    deepfakes = sum(1 for item in analysis_history if item["verdict"] == "DEEPFAKE")
    real = len(analysis_history) - deepfakes
    
    avg_confidence = sum(item["confidence"] for item in analysis_history) / len(analysis_history)
    avg_video = sum(item["overall_scores"]["overall_video_score"] for item in analysis_history) / len(analysis_history)
    avg_audio = sum(item["overall_scores"]["overall_audio_score"] for item in analysis_history) / len(analysis_history)
    
    return StatsResponse(
        total_analyses=len(analysis_history),
        deepfakes_detected=deepfakes,
        real_videos=real,
        avg_confidence=round(avg_confidence, 2),
        avg_video_score=round(avg_video, 3),
        avg_audio_score=round(avg_audio, 3)
    )

@app.get("/api/intervals/{analysis_id}")
async def get_interval_details(analysis_id: str):
    """Get detailed interval-by-interval breakdown"""
    
    results_path = UPLOAD_DIR / f"{analysis_id}_results.json"
    
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    with open(results_path, 'r') as f:
        interval_data = json.load(f)
    
    timeline = interval_data.get('timeline', [])
    
    intervals = []
    for interval in timeline:
        video_res = interval.get('video_results', {})
        audio_res = interval.get('audio_results', {})
        
        avg_score = (video_res.get('fake_score', 0) + audio_res.get('fake_score', 0)) / 2
        
        intervals.append({
            "interval_id": interval['interval_id'],
            "time_range": interval['interval'],
            "start": interval['start'],
            "end": interval['end'],
            "video_score": video_res.get('fake_score', 0),
            "audio_score": audio_res.get('fake_score', 0),
            "combined_score": round(avg_score, 3),
            "verdict": "SUSPICIOUS" if avg_score > 0.6 else "NORMAL",
            "suspicious_regions": {
                "video": video_res.get('suspicious_regions', []),
                "audio": audio_res.get('suspicious_regions', [])
            },
            "has_face": video_res.get('face_detected', False),
            "has_audio": audio_res.get('has_audio', False)
        })
    
    return {
        "analysis_id": analysis_id,
        "total_intervals": len(intervals),
        "intervals": intervals
    }

@app.get("/api/compare")
async def compare_scores():
    """Compare video vs audio detection rates"""
    
    if not analysis_history:
        return {
            "message": "No analysis data available",
            "comparison": None
        }
    
    video_higher = 0
    audio_higher = 0
    equal = 0
    
    for item in analysis_history:
        scores = item["overall_scores"]
        v_score = scores["overall_video_score"]
        a_score = scores["overall_audio_score"]
        
        if v_score > a_score:
            video_higher += 1
        elif a_score > v_score:
            audio_higher += 1
        else:
            equal += 1
    
    return {
        "total_analyses": len(analysis_history),
        "comparison": {
            "video_better_detection": video_higher,
            "audio_better_detection": audio_higher,
            "equal_detection": equal
        },
        "percentages": {
            "video_dominant": round((video_higher / len(analysis_history)) * 100, 1),
            "audio_dominant": round((audio_higher / len(analysis_history)) * 100, 1),
            "balanced": round((equal / len(analysis_history)) * 100, 1)
        }
    }

@app.get("/api/recent-verdict")
async def get_recent_verdict_distribution(limit: int = Query(default=20, ge=5, le=50)):
    """Get verdict distribution for recent analyses"""
    
    recent = analysis_history[:limit]
    
    if not recent:
        return {
            "total": 0,
            "deepfakes": 0,
            "real": 0,
            "distribution": []
        }
    
    deepfakes = sum(1 for item in recent if item["verdict"] == "DEEPFAKE")
    real = len(recent) - deepfakes
    
    distribution = {
        "very_confident": 0,  
        "confident": 0,        
        "moderate": 0,         
        "low": 0               
    }
    
    for item in recent:
        conf = item["confidence"]
        if conf >= 80:
            distribution["very_confident"] += 1
        elif conf >= 60:
            distribution["confident"] += 1
        elif conf >= 40:
            distribution["moderate"] += 1
        else:
            distribution["low"] += 1
    
    return {
        "total": len(recent),
        "deepfakes": deepfakes,
        "real": real,
        "deepfake_rate": round((deepfakes / len(recent)) * 100, 1),
        "confidence_distribution": distribution
    }

@app.delete("/api/clear-history")
async def clear_history():
    """Clear analysis history (for demo reset)"""
    global analysis_history
    
    count = len(analysis_history)
    analysis_history.clear()
    
    for file in UPLOAD_DIR.glob("*_results.json"):
        os.remove(file)
    
    return {
        "message": "History cleared",
        "items_removed": count
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"Error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )
        