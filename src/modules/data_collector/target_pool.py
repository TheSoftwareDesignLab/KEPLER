import random
from typing import List, Dict, Any, Optional
from src.core.datatypes import TargetTask

__all__ = ["generate_dynamic_tasks"]


def generate_dynamic_tasks(
    k: int,
    bounding_boxes: List[Dict[str, Any]],
    polygon_ratio: float = 0.5,
    min_area_deg: float = 0.05,
    max_area_deg: float = 0.20,
    min_duration: int = 5,
    max_duration: int = 30,
    min_release_delay: int = 0,
    max_release_delay: int = 7200,     
    min_lifetime: int = 1800,          
    max_lifetime: int = 7200,          
    available_sensors: List[str] = None,
    priority_weights: Optional[List[float]] = None,
    seed: Optional[int] = None
) -> List[TargetTask]:
    """
    Procedurally generates a deterministic list of TargetTasks with relative temporal 
    windows (release time and deadline) anchored to t0 = 0 seconds.
    """
    if k <= 0:
        return []
        
    if not bounding_boxes:
        raise ValueError("Task generation requires at least one active bounding box definition.")
        
    sensors = available_sensors if available_sensors is not None else ["VIS", "SAR", "TIR"]
    rng = random.Random(seed)
    tasks = []

    priorities_pool = [1, 2, 3, 4, 5]

    for i in range(k):
        task_id = f"TASK_GEN_{i+1:03d}"
        
        if priority_weights is not None:
            priority = int(rng.choices(priorities_pool, weights=priority_weights, k=1)[0])
        else:
            priority = rng.randint(1, 5)
        
        # RELATIVE TEMPORAL ENGINE (t0 = 0 seconds baseline)
        release_time = rng.randint(min_release_delay, max_release_delay)
        task_lifetime = rng.randint(min_lifetime, max_lifetime)
        deadline = release_time + task_lifetime
        
        box = bounding_boxes[i % len(bounding_boxes)]
        region_tag = box.get("name", "unknown_region")
        
        lat_envelope = box.get("lat_envelope", [0.0, 0.0])
        lon_envelope = box.get("lon_envelope", [0.0, 0.0])
        
        center_lat = rng.uniform(lat_envelope[0], lat_envelope[1])
        center_lon = rng.uniform(lon_envelope[0], lon_envelope[1])
        
        num_sensors = rng.randint(1, min(2, len(sensors)))
        required_sensors = rng.sample(sensors, num_sensors)
        
        is_polygon = rng.random() < polygon_ratio
        coordinates = []
        
        if not is_polygon:
            task_type = "point"
            coordinates.append((center_lat, center_lon))
        else:
            task_type = "polygon"
            delta_lat = rng.uniform(min_area_deg, max_area_deg)
            delta_lon = rng.uniform(min_area_deg, max_area_deg)
            
            p1 = (center_lat + delta_lat, center_lon - delta_lon)
            p2 = (center_lat + delta_lat, center_lon + delta_lon)
            p3 = (center_lat - delta_lat, center_lon + delta_lon)
            p4 = (center_lat - delta_lat, center_lon - delta_lon)
            
            coordinates.extend([p1, p2, p3, p4, p1])

        tasks.append(
            TargetTask(
                task_id=task_id,
                region_tag=region_tag,
                priority=priority,
                task_type=task_type,
                coordinates=coordinates,
                required_sensors=required_sensors,
                release_time=release_time,
                deadline=deadline
            )
        )
        
    return tasks