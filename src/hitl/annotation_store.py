import json
import os
from pathlib import Path
from src.config import ANNOTATIONS_FILE

class AnnotationStore:
    def __init__(self, file_path=ANNOTATIONS_FILE):
        self.file_path = Path(file_path)
        self.annotations = self._load()
        
    def _load(self):
        if not self.file_path.exists():
            return []
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return []
            
    def save(self):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, 'w') as f:
            json.dump(self.annotations, f, indent=2)
            
    def add_annotation(self, filename, label_id, label_name):
        # Check if already exists
        for ann in self.annotations:
            if ann['filename'] == filename:
                ann['label_id'] = label_id
                ann['label_name'] = label_name
                self.save()
                return
                
        self.annotations.append({
            "filename": filename,
            "label_id": label_id,
            "label_name": label_name
        })
        self.save()
        
    def get_count(self):
        return len(self.annotations)
        
    def is_annotated(self, filename):
        return any(ann['filename'] == filename for ann in self.annotations)
