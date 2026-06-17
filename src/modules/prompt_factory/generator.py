import os
import json
import requests
import time
import ssl 
import urllib3 
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import List, Dict, Any, Optional
from geopy.geocoders import Nominatim
from src.core.datatypes import TargetTask

__all__ = ["generate_ollama_semantic_prompt", "build_single_task_string"]


def _get_geocoded_info(lat: Optional[float], lon: Optional[float]) -> Dict[str, str]:
    print(f"\n[DEBUG GEOLOC] Iniciando geocodificación para coordenadas: lat={lat}, lon={lon}")
    
    if lat is None or lon is None:
        print("[DEBUG GEOLOC] Error: Una o ambas coordenadas son None. Se retorna N/A.")
        return {"country": "N/A", "city": "N/A", "landmark": "N/A"}
    
    time.sleep(1.2)
    
    try:
       
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": float(lat),
            "lon": float(lon),
            "format": "json",
            "addressdetails": 1
        }
        headers = {
            "User-Agent": f"satellite_constellation_research_{int(time.time())}"
        }
        
        print(f"[DEBUG GEOLOC] Realizando petición web directa con requests (verify=False) a {url}...")
        response = requests.get(url, params=params, headers=headers, verify=False, timeout=15)
        response.raise_for_status()
        raw_data = response.json()
        
        if raw_data and "address" in raw_data:
            print(f"[DEBUG GEOLOC] Conexión exitosa. Dirección completa devuelta: {raw_data.get('display_name')}")
            
            address = raw_data["address"]
            country = address.get("country", "N/A")
            
            city = address.get(
                "city", 
                address.get(
                    "town", 
                    address.get(
                        "village", 
                        address.get("county", "N/A")
                    )
                )
            )
            
            landmark = address.get(
                "tourism", 
                address.get(
                    "historic", 
                    address.get(
                        "amenity", 
                        address.get("building", "N/A")
                    )
                )
            )
            
            print(f"[DEBUG GEOLOC] Valores extraídos -> country: '{country}', city: '{city}', landmark: '{landmark}'")
            
            return {
                "country": country,
                "city": city,
                "landmark": landmark
            }
        else:
            print(f"[DEBUG GEOLOC] Nominatim retornó una estructura de respuesta inválida para: {lat}, {lon}")
            
    except Exception as e:
        print(f"[DEBUG GEOLOC ERROR] Excepción atrapada durante la geocodificación directa: {e}")
        
    return {"country": "N/A", "city": "N/A", "landmark": "N/A"}


def _format_remaining_time(deadline_utc: datetime, now_utc: datetime) -> str:
    delta = deadline_utc - now_utc
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return "past due"
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, _ = divmod(remainder, 60)
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return " ".join(parts)


def build_single_task_string(task: TargetTask, now_utc: datetime) -> str:
    local_tz = ZoneInfo("America/Bogota")
    clean_name = task.region_tag.replace("_", " ")
    primary_sensor = task.required_sensors[0] if task.required_sensors else "VIS"
    
    raw_deadline = getattr(task, "deadline", getattr(task, "deadline_s", 0))

    deadline_epoch = now_utc.timestamp() + raw_deadline
        
    task_deadline_utc = datetime.fromtimestamp(deadline_epoch, tz=timezone.utc)
    task_deadline_local = task_deadline_utc.astimezone(local_tz)
    remaining_str = _format_remaining_time(task_deadline_utc, now_utc)
    
    lat, lon = None, None
    
    coords = getattr(task, "coordinates", None)
    if coords and isinstance(coords, (list, tuple)) and len(coords) > 0:
        valid_coords = [c for c in coords if isinstance(c, (list, tuple)) and len(c) >= 2]
        if valid_coords:
            lat = sum(float(c[0]) for c in valid_coords) / len(valid_coords)
            lon = sum(float(c[1]) for c in valid_coords) / len(valid_coords)
            
    if lat is None or lon is None:
        lat = getattr(task, "latitude", getattr(task, "lat", None))
        lon = getattr(task, "longitude", getattr(task, "lon", None))
    
    if lat is None or lon is None:
        for attr in ["location", "position", "geometry"]:
            loc_obj = getattr(task, attr, None)
            if loc_obj:
                if isinstance(loc_obj, dict):
                    lat = loc_obj.get("latitude") or loc_obj.get("lat")
                    lon = loc_obj.get("longitude") or loc_obj.get("lon")
                else:
                    lat = getattr(loc_obj, "latitude", getattr(loc_obj, "lat", None))
                    lon = getattr(loc_obj, "longitude", getattr(loc_obj, "lon", None))
                break
                
    if lat is None or lon is None:
        lat_env = getattr(task, "lat_envelope", None)
        lon_env = getattr(task, "lon_envelope", None)
        if lat_env and lon_env and isinstance(lat_env, (list, tuple)) and isinstance(lon_env, (list, tuple)):
            if len(lat_env) >= 2 and len(lon_env) >= 2:
                lat = sum(lat_env) / len(lat_env)
                lon = sum(lon_env) / len(lon_env)
            
    print(f"\n[DEBUG TASK BUILD] Coordenadas finales extraídas de la tarea ({task.task_id}): lat={lat}, lon={lon}")
    
    geo_info = _get_geocoded_info(lat, lon)
    priority = getattr(task, "priority", getattr(task, "priority_level", 3))
    
    task_json_data = {
        "primary_sensor": primary_sensor,
        "deadline_utc": task_deadline_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
        "deadline_local": task_deadline_local.strftime('%Y-%m-%d %H:%M (%Z)'),
        "remaining_time": remaining_str,
        "location_details": {
            "country": geo_info['country'],
            "city": geo_info['city'],
            "landmark": geo_info['landmark']
        },
        "priority_level": priority
    }
    
    return json.dumps(task_json_data, indent=2, ensure_ascii=False)


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
    
    generated_prompts_map = {}
    
    for task in targets:
        now_utc = datetime.now(timezone.utc)
        task_string = build_single_task_string(task, now_utc)
        full_prompt = system_instruction_template.format(tasks_dataset=task_string)
        
        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "repeat_penalty": repeat_penalty
            }
        }

        print(f"\n[OLLAMA] Generating prompt for {task.task_id}...")
        response = requests.post(
            target_endpoint,
            json=payload,
            stream=True,
            timeout=180
        )
        response.raise_for_status()

        output_chunks = []
        for line in response.iter_lines():
            if not line:
                continue
            line_object = json.loads(line)
            if "response" not in line_object:
                continue
            
            chunk = line_object["response"]
            print(chunk, end="", flush=True)
            output_chunks.append(chunk)

            if line_object.get("done"):
                break

        print()
        raw_text = "".join(output_chunks)
        
        if "```prompt" in raw_text:
            parsed_text = raw_text.split("```prompt")[1].split("```")[0].strip()
        else:
            parsed_text = raw_text.strip()
            
        generated_prompts_map[task.task_id] = parsed_text
        
    return generated_prompts_map