from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from analysis.prompt import _create_analysis_prompt
from dotenv import load_dotenv
import re
load_dotenv()

class LLMFusion:
    """Fuses video and audio analysis results using LLM to generate human-readable report"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    def prepare_analysis_json(self, timeline: List[Dict]) -> Dict:
        analysis_data = {
            'total_intervals': len(timeline),
            'intervals': []
        }
        
        for interval in timeline:
            interval_summary = {
                'interval_id': interval['interval_id'],
                'time_range': interval['interval'],
                'video_analysis': interval.get('video_results', {}),
                'audio_analysis': interval.get('audio_results', {})
            }
            analysis_data['intervals'].append(interval_summary)
        
        return analysis_data
    
    def calculate_overall_scores(self, timeline: List[Dict]) -> Dict:
        """Calculate overall video and audio fake scores"""
        video_scores = []
        audio_scores = []
        
        for interval in timeline:
            if interval.get('video_results') and 'fake_score' in interval['video_results']:
                video_scores.append(interval['video_results']['fake_score'])
            
            if interval.get('audio_results') and 'fake_score' in interval['audio_results']:
                audio_scores.append(interval['audio_results']['fake_score'])
        
        overall_video = round(sum(video_scores) / len(video_scores), 3) if len(video_scores) > 0 else 0.0
        overall_audio = round(sum(audio_scores) / len(audio_scores), 3) if len(audio_scores) > 0 else 0.0
        
        if overall_video > 0 and overall_audio > 0:
            overall_combined = round((overall_video + overall_audio) / 2, 3)
        elif overall_video > 0:
            overall_combined = overall_video
        elif overall_audio > 0:
            overall_combined = overall_audio
        else:
            overall_combined = 0.0
        
        return {
            'overall_video_score': overall_video,
            'overall_audio_score': overall_audio,
            'overall_combined_score': overall_combined
        }
    
    def generate_report(self, timeline: List[Dict], video_info: Dict) -> Dict:
        analysis_json = self.prepare_analysis_json(timeline)
        overall_scores = self.calculate_overall_scores(timeline)
        
        prompt = _create_analysis_prompt(analysis_json, overall_scores, video_info)
        
        try:
            response = self.llm.invoke(prompt)
            llm_response = response.content 
        except Exception as e:
            print(f"LLM failed: {e}")
            llm_response = "Analysis failed."
        
        report = self._structure_report(llm_response, overall_scores, analysis_json)
        return report
    
    def _structure_report(self, llm_response: str, overall_scores: Dict, analysis_json: Dict) -> Dict:
        """Extract structured information from LLM response"""
        
        verdict = "DEEPFAKE" if overall_scores['overall_combined_score'] > 0.5 else "REAL"
        
        confidence = 75.0
        conf_match = re.search(r'(\d+)\s*%', llm_response)
        if conf_match:
            confidence = float(conf_match.group(1))
        
        suspicious_intervals = []
        for interval_data in analysis_json['intervals']:
            video_score = interval_data.get('video_analysis', {}).get('fake_score', 0)
            audio_score = interval_data.get('audio_analysis', {}).get('fake_score', 0)
            
            if video_score > 0.6 or audio_score > 0.6:
                suspicious_intervals.append({
                    'interval': interval_data['time_range'],
                    'video_score': video_score,
                    'audio_score': audio_score,
                    'video_regions': interval_data.get('video_analysis', {}).get('suspicious_regions', []),
                    'audio_regions': interval_data.get('audio_analysis', {}).get('suspicious_regions', [])
                })
        
        return {
            'verdict': verdict,
            'confidence': confidence,
            'overall_scores': overall_scores,
            'detailed_analysis': llm_response,
            'suspicious_intervals': suspicious_intervals,
            'total_intervals_analyzed': analysis_json['total_intervals']
        }