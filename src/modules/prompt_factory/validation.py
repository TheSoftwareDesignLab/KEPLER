import os
import json
import urllib.request
import numpy as np
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
        
        payload = {
            "model": self.model_name,
            "input": input_data
        }
        
        req = urllib.request.Request(
            target_endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["embeddings"]

    def _initialize_category_dict(self, categories: Dict[str, str]) -> Dict[str, np.ndarray]:
        if not categories:
            return {}
        tags = list(categories.keys())
        phrases = list(categories.values())
        
        embeddings = self._get_ollama_embeddings(phrases)
        
        category_dict = {}
        for tag, vector in zip(tags, embeddings):
            category_dict[tag] = np.array(vector)
        return category_dict

    def _initialize_list_dict(self, categories: List[str]) -> Dict[str, np.ndarray]:
        if not categories:
            return {}
        
        embeddings = self._get_ollama_embeddings(categories)
        
        category_dict = {}
        for text_anchor, vector in zip(categories, embeddings):
            category_dict[text_anchor] = np.array(vector)
        return category_dict

    def _find_closest_tag(self, target_vector: np.ndarray, embeddings_dict: Dict[str, np.ndarray]) -> Tuple[str, float]:
        if not embeddings_dict:
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

    def validate_generated_request(self, generated_text: str) -> Dict[str, Any]:
        embeddings = self._get_ollama_embeddings(generated_text)
        sentence_vector = np.array(embeddings[0])
        
        predicted_sensor, sensor_sim = self._find_closest_tag(sentence_vector, self.sensor_dict)
        predicted_priority, priority_sim = self._find_closest_tag(sentence_vector, self.priority_dict)
        predicted_day, day_sim = self._find_closest_tag(sentence_vector, self.days_dict)
        predicted_hour, hour_sim = self._find_closest_tag(sentence_vector, self.hours_dict)
        
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