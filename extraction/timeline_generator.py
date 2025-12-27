import numpy as np
from typing import List, Dict

class TimelineGenerator:
    
    def __init__(self, interval_duration: float = 2.0):
        self.interval_duration = interval_duration
        
    def create_timeline(self, video_duration: float) -> List[Dict]:
        
        num_intervals = int(np.ceil(video_duration / self.interval_duration))
        
        timeline = []
        for i in range(num_intervals):
            start_time = i * self.interval_duration
            end_time = min((i + 1) * self.interval_duration, video_duration)
            
            timeline.append({
                'interval_id': i,
                'start': round(start_time, 2),
                'end': round(end_time, 2),
                'interval': f"{start_time:.1f}-{end_time:.1f}",
                'duration': round(end_time - start_time, 2),
                'video_data': [],  
                'audio_data': None,  
                'video_results': None,  
                'audio_results': None
            })
            
        return timeline
    
    def get_interval_for_timestamp(self, timeline: List[Dict], timestamp: float) -> Dict:
        
        for interval in timeline:
            if interval['start'] <= timestamp < interval['end']:
                return interval
            
        return timeline[-1]