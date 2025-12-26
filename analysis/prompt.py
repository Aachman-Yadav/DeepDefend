import json
from typing import List, Dict

def _create_analysis_prompt(analysis_json: Dict, overall_scores: Dict, video_info: Dict) -> str:
    """Create prompt for LLM with proper score interpretation"""
    
    prompt = f"""You are a deepfake detection expert. Analyze the following video analysis results and generate a human-readable report.

VIDEO INFORMATION:
- Duration: {video_info['duration']:.2f} seconds
- Total Intervals Analyzed: {analysis_json['total_intervals']}

OVERALL SCORES (CRITICAL - READ CAREFULLY):
- Video Deepfake Score: {overall_scores['overall_video_score']}
- Audio Deepfake Score: {overall_scores['overall_audio_score']}
- Averaged Combined Score: {overall_scores['overall_combined_score']}

SCORE INTERPRETATION GUIDE:
- Scores range from 0.0 to 1.0
- 0.0 - 0.3: LIKELY REAL (low probability of manipulation)
- 0.3 - 0.5: POSSIBLY REAL (some minor artifacts, but probably authentic)
- 0.5 - 0.7: POSSIBLY FAKE (suspicious patterns detected)
- 0.7 - 1.0: LIKELY FAKE (high probability of deepfake)

IMPORTANT: The numerical scores are the PRIMARY evidence. Suspicious regions are secondary indicators that provide detail about WHERE issues were detected, but should NOT override low scores.

INTERVAL-BY-INTERVAL ANALYSIS:
{json.dumps(analysis_json['intervals'], indent=2)}

ANALYSIS RULES:
1. If average score < 0.5, you should lean towards "REAL" verdict unless there is overwhelming contradictory evidence
2. If average score > 0.5, you should lean towards "DEEPFAKE" verdict
3. Suspicious regions (like "monotone_voice" or "eyes") only matter if the scores also indicate manipulation
4. A low score with suspicious regions = detection system being cautious, likely still REAL
5. Base your confidence on how far the scores are from 0.5 threshold

TASK:
Based on the analysis above, provide:

1. **VERDICT**: State clearly if this is "REAL" or "DEEPFAKE"
   - Must align with the overall scores
   - If avg score < 0.5, verdict should typically be REAL
   - If avg score > 0.5, verdict should typically be DEEPFAKE

2. **CONFIDENCE**: Your confidence level (0-100%)
   - Base this on how definitive the scores are
   - Score near 0.0 or 1.0 = high confidence
   - Score near 0.5 = low confidence

3. **KEY FINDINGS**: Summarize the most important patterns found
   - Focus on intervals with scores > 0.6 (those are actually suspicious)
   - Mention if scores are consistently low (indicates authentic content)

4. **SUSPICIOUS INTERVALS**: Only list intervals where fake_score > 0.6
   - If no intervals exceed 0.6, state "No highly suspicious intervals detected"

5. **EVIDENCE SUMMARY**: 
   - Video evidence: Mention specific facial regions only if video score > 0.5
   - Audio evidence: Mention audio patterns only if audio score > 0.5
   - If scores are low, acknowledge the content appears authentic

6. **EXPLANATION**: In 2-3 sentences, explain your verdict
   - Reference the numerical scores explicitly
   - Explain in simple terms what the scores mean for this video

CRITICAL REMINDER: Your verdict MUST be consistent with the numerical scores. Do not declare something a deepfake if the scores indicate it's real (< 0.5).

Format your response as a clear, structured analysis that a non-technical person could understand."""

    return prompt