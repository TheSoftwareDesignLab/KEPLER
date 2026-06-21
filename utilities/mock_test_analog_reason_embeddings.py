import sys
import os
import numpy as np
import urllib.request
import json
from typing import List, Dict, Any

class TargetTask:
    def __init__(self, data: dict):
        self.data = data

def find_closest_word(target_vector: np.ndarray, embeddings_dict: dict, ignore_words: list = None) -> list:
    if ignore_words is None:
        ignore_words = set()
    else:
        ignore_words = set(ignore_words)

    valid_keys = [k for k in embeddings_dict.keys() if k not in ignore_words]
    
    if not valid_keys:
        return []

    embeddings_matrix = np.array([embeddings_dict[k] for k in valid_keys])

    target_norm = np.linalg.norm(target_vector)
    if target_norm == 0:
        return [(k, 0.0) for k in valid_keys]
    normalized_target = target_vector / target_norm

    matrix_norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    matrix_norms[matrix_norms == 0] = 1.0 
    normalized_matrix = embeddings_matrix / matrix_norms

    all_similarities = np.dot(normalized_matrix, normalized_target)

    results = list(zip(valid_keys, map(float, all_similarities)))
    results.sort(key=lambda x: x[1], reverse=True)

    return results

def get_ollama_embeddings(input_data: Any, model_name: str = "mxbai-embed-large") -> List[List[float]]:
    raw_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    clean_url = raw_url.replace("[", "").replace("]", "").split("(")[0].strip()
    target_endpoint = f"{clean_url}/api/embed"
    
    payload = {
        "model": model_name,
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

def find_closest_matches_for_sentence(sentence: str, embeddings_dict: dict, model_name: str = "mxbai-embed-large") -> list:
    embeddings = get_ollama_embeddings(sentence, model_name)
    sentence_vector = np.array(embeddings[0])
    return find_closest_word(sentence_vector, embeddings_dict)

def generate_ollama_semantic_prompt(
    targets: List[TargetTask],
    system_instruction_template: str,
    model_name: str = "llama3.1:8b",
    temperature: float = 0.3,
    num_predict: int = 700,
    repeat_penalty: float = 1.05
) -> Dict[str, str]:
    raw_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    clean_url = raw_url.replace("[", "").replace("]", "").split("(")[0].strip()
    target_endpoint = f"{clean_url}/api/generate"
    
    targets_json = [t.data for t in targets]
    formatted_prompt = f"{system_instruction_template}\n\nTasks Metadata JSON:\n{json.dumps(targets_json, indent=2)}"
    
    payload = {
        "model": model_name,
        "prompt": formatted_prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "repeat_penalty": repeat_penalty
        }
    }
    
    req = urllib.request.Request(
        target_endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))
        return {"generated_output": result.get("response", "")}

def run_bench_test():
    urgency_vocabulary = ["low", "medium", "high"]
    urgency_dict = {}
    urgency_embeddings = get_ollama_embeddings(urgency_vocabulary, "mxbai-embed-large")
    for word, vector in zip(urgency_vocabulary, urgency_embeddings):
        urgency_dict[word] = np.array(vector)

    sensor_candidates = {
        "TIR": "thermal mapping heat signature surface temperature profile thermal anomaly detection",
        "VNIR": "multispectral vegetation analysis chlorophyll absorption level NDVI index",
        "SAR": "active radar scan microwave surface imaging cloud-penetrating capture",
        "NIR": "near-infrared reflection water body boundary mapping soil moisture assessment",
        "VISUAL": "high-resolution optical snapshot daylight photography RGB true-color imagery"
    }
    
    sensor_dict = {}
    phrases_list = list(sensor_candidates.values())
    tags_list = list(sensor_candidates.keys())
    
    sensor_embeddings = get_ollama_embeddings(phrases_list, "mxbai-embed-large")
    for tag, vector in zip(tags_list, sensor_embeddings):
        sensor_dict[tag] = np.array(vector)

    time_vocabulary = ["morning", "afternoon", "evening", "night", "none"]
    time_dict = {}
    time_embeddings = get_ollama_embeddings(time_vocabulary, "mxbai-embed-large")
    for word, vector in zip(time_vocabulary, time_embeddings):
        time_dict[word] = np.array(vector)

    urgency_vocabulary = ["low", "medium", "high"]
    urgency_dict = {}
    urgency_embeddings = get_ollama_embeddings(urgency_vocabulary, "mxbai-embed-large")
    for word, vector in zip(urgency_vocabulary, urgency_embeddings):
        urgency_dict[word] = np.array(vector)

    sample_sentence = "Translate this thermal infrared (TIR) mission to map heat emissions and surface temperature distributions across the region of Panamá, with a focus on capturing high-resolution imagery of the thermal signatures and heat dissipation anomalies. This is an absolute institutional priority, as per our contractual agreement, and requires an urgent operational execution within the next 48 hours, specifically by June 20th at 11:55 PM (-05). The mission must be executed without delay or preemption to ensure a critical payload capture."
    sample_sentence = "at the beginning of the day"

    print(f"Analyzing sentence: '{sample_sentence}'")
    print("-" * 60)
    
    print("[TEST 1] Urgency Classification:")
    urgency_matches = find_closest_matches_for_sentence(sample_sentence, urgency_dict)
    for i, (word, similarity) in enumerate(urgency_matches[:1]):
        print(f"  Top Match: '{word}' (Cosine Similarity: {similarity:.4f})")
    
    print("\n[TEST 2] Sensor Profile Matching (Tagged):")
    sensor_matches = find_closest_matches_for_sentence(sample_sentence, sensor_dict)
    for i, (tag, similarity) in enumerate(sensor_matches[:1]):
        print(f"  Top Match: {tag} (Cosine Similarity: {similarity:.4f})")
        print(f"  Profile Context: '{sensor_candidates[tag]}'")

    print("[TEST 3] Time Classification:")
    time_matches = find_closest_matches_for_sentence(sample_sentence, time_dict)
    for i, (word, similarity) in enumerate(time_matches[:1]):
        print(f"  Top Match: '{word}' (Cosine Similarity: {similarity:.4f})")

if __name__ == "__main__":
    run_bench_test()