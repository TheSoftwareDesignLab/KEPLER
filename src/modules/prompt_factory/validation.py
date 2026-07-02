import os
import json
import urllib.request
import numpy as np
import re
from typing import List, Dict, Tuple, Any

class SemanticValidator:
    def __init__(
        self, 
        sensor_categories: Dict[str, str], 
        priority_categories: List[str], 
        days_categories: List[str],
        hours_categories: List[str],
        embedding_model: str = "mxbai-embed-large"
    ):
        self.sensor_categories = sensor_categories
        self.priority_categories = priority_categories
        self.days_categories = days_categories
        self.hours_categories = hours_categories
        self.model_name = embedding_model
        
        self.sensor_dict = self._initialize_category_dict(self.sensor_categories)
        self.priority_dict = self._initialize_list_dict(self.priority_categories)
        self.days_dict = self._initialize_list_dict(self.days_categories)
        self.hours_dict = self._initialize_list_dict(self.hours_categories)

    def _get_ollama_embeddings(self, input_data: Any) -> List[List[float]]:
        raw_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        clean_url = raw_url.replace("[", "").replace("]", "").split("(")[0].strip()
        target_endpoint = f"{clean_url}/api/embed"
        
        if isinstance(input_data, str):
            input_data = [input_data]
            
        payload = {
            "model": self.model_name,
            "input": input_data
        }
        
        try:
            req = urllib.request.Request(
                target_endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=15) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["embeddings"]
        except Exception as e:
            print(f"[OLLAMA EMBED ERROR] Telemetry recovery failed: {e}")
            return []

    def _initialize_category_dict(self, categories: Dict[str, str]) -> Dict[str, np.ndarray]:
        if not categories:
            return {}
        tags = list(categories.keys())
        phrases = list(categories.values())
        
        embeddings = self._get_ollama_embeddings(phrases)
        if not embeddings:
            return {tag: np.zeros(1024) for tag in tags}
            
        category_dict = {}
        for tag, vector in zip(tags, embeddings):
            category_dict[tag] = np.array(vector)
        return category_dict

    def _initialize_list_dict(self, categories: List[str]) -> Dict[str, np.ndarray]:
        if not categories:
            return {}
        
        embeddings = self._get_ollama_embeddings(categories)
        if not embeddings:
            return {text_anchor: np.zeros(1024) for text_anchor in categories}
            
        category_dict = {}
        for text_anchor, vector in zip(categories, embeddings):
            category_dict[text_anchor] = np.array(vector)
        return category_dict

    def _find_closest_tag(self, target_vector: np.ndarray, embeddings_dict: Dict[str, np.ndarray]) -> Tuple[str, float]:
        if not embeddings_dict or target_vector is None or target_vector.size == 0:
            return "", 0.0

        valid_keys = list(embeddings_dict.keys())
        embeddings_matrix = np.array([embeddings_dict[k] for k in valid_keys])

        target_norm = np.linalg.norm(target_vector)
        if target_norm == 0:
            return valid_keys[0], 0.0
        normalized_target = target_vector / target_norm

        matrix_norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        matrix_norms[matrix_norms == 0] = 1.0 
        normalized_matrix = embeddings_matrix / matrix_norms

        all_similarities = np.dot(normalized_matrix, normalized_target)
        
        best_index = int(np.argmax(all_similarities))
        return valid_keys[best_index], float(all_similarities[best_index])

    def _to_vector(self, embeddings: Any) -> np.ndarray:
        if not embeddings or not isinstance(embeddings, list):
            return np.zeros(1024)
        first_element = embeddings[0]
        if isinstance(first_element, list):
            return np.array(first_element)
        return np.array(embeddings)

    def validate_generated_request(self, generated_text: str) -> Dict[str, Any]:
        if not generated_text or len(generated_text.strip()) == 0:
            return {
                "predicted_sensor": "N/A", "sensor_similarity": 0.0,
                "predicted_priority": "N/A", "priority_similarity": 0.0,
                "predicted_day": "N/A", "day_similarity": 0.0,
                "predicted_hour": "N/A", "hour_similarity": 0.0
            }

        day_pattern = r'\b(today|tomorrow|after\s+tomorrow|three\s+later|four\s+later)\b'
        hour_pattern = r'\b(morning|mid-day|noon|afternoon|evening|overnight|tonight|night)\b'

        day_found = re.search(day_pattern, generated_text, re.IGNORECASE)
        hour_found = re.search(hour_pattern, generated_text, re.IGNORECASE)

        day_token = day_found.group(0).strip() if day_found else ""
        hour_token = hour_found.group(0).strip() if hour_found else ""

        time_snippets = []
        if day_token:
            time_snippets.append(day_token)
        if hour_token:
            time_snippets.append(hour_token)

        time_snippet = " ".join(time_snippets) if time_snippets else generated_text

        full_text_embeddings = self._get_ollama_embeddings(generated_text)
        time_text_embeddings = self._get_ollama_embeddings(time_snippet)

        global_vector = self._to_vector(full_text_embeddings)
        temporal_vector = self._to_vector(time_text_embeddings)
        
        predicted_sensor, sensor_sim = self._find_closest_tag(global_vector, self.sensor_dict)
        predicted_priority, priority_sim = self._find_closest_tag(global_vector, self.priority_dict)
        predicted_day, day_sim = self._find_closest_tag(temporal_vector, self.days_dict)
        predicted_hour, hour_sim = self._find_closest_tag(temporal_vector, self.hours_dict)
        
        return {
            "predicted_sensor": predicted_sensor,
            "sensor_similarity": sensor_sim,
            "predicted_priority": predicted_priority,
            "priority_similarity": priority_sim,
            "predicted_day": predicted_day,
            "day_similarity": day_sim,
            "predicted_hour": predicted_hour,
            "hour_similarity": hour_sim
        }