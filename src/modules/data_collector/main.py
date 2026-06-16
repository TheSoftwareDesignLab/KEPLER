import json
import pathlib
from typing import List, Dict, Any, Optional
from src.core.datatypes import CollectedContext
from .norad_pool import load_and_sample_satellites
from .gs_pool import load_and_sample_stations
from .payload_assign import assign_satellite_payloads
from .target_pool import generate_dynamic_tasks

__all__ = ["data_collector_main"]


def _export_scenario_report(context: CollectedContext, report_path: str = "data/scenario_report.json") -> None:
    """
    Serializes the runtime generated context parameters into a persistent JSON report.
    """
    path = pathlib.Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    report_data = {
        "metadata": {
            "total_satellites": len(context.satellites),
            "total_ground_stations": len(context.ground_stations),
            "total_target_tasks": len(context.targets)
        },
        "satellites": [
            {
                "norad_id": sat.norad_id,
                "name": sat.name,
                "tle_line1": sat.tle_line1,
                "tle_line2": sat.tle_line2,
                "assigned_band": sat.band,
                "assigned_sensors": sat.sensors
            }
            for sat in context.satellites
        ],
        "ground_stations": [
            {
                "id": gs.id,
                "name": gs.name,
                "latitude": gs.latitude,
                "longitude": gs.longitude,
                "elevation_m": gs.elevation,
                "bands_supported": gs.bands_supported
            }
            for gs in context.ground_stations
        ],
        "targets": [
            {
                "task_id": t.task_id,
                "region_tag": t.region_tag,
                "priority": t.priority,
                "task_type": t.task_type,
                "release_time_s": t.release_time,
                "deadline_s": t.deadline,
                "coordinates": t.coordinates
            }
            for t in context.targets
        ]
    }
    
    with path.open("w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)


def data_collector_main(
    sat_k: int,
    gs_k: int,
    available_sensors: List[str],
    tasks_k: int = 10,
    bounding_boxes: Optional[List[Dict[str, Any]]] = None,
    polygon_ratio: float = 0.5,
    min_area_deg: float = 0.05,
    max_area_deg: float = 0.20,
    min_duration: int = 5,
    max_duration: int = 30,
    min_release_delay: int = 0,
    max_release_delay: int = 3600,
    min_lifetime: int = 1800,
    max_lifetime: int = 7200,
    sat_file_path: Optional[str] = None,
    sat_group_name: Optional[str] = None,
    gs_file_path: str = "data/ground_station.csv",
    sensor_weights: Optional[List[float]] = None,
    band_weights_map: Optional[dict] = None,  
    min_sensors_per_sat: int = 1,
    max_sensors_per_sat: int = 1,
    priority_weights: Optional[List[float]] = None,
    seed: Optional[int] = None
) -> CollectedContext:
    """
    Unified local orchestrator for Phase 1. Sequences ground station sampling, 
    dynamic frequency constraint routing for satellite payload assignment, and 
    procedural multi-box target mission generation.
    """
    
    user_allowed_bands = list(band_weights_map.keys()) if band_weights_map else None

    ground_stations = load_and_sample_stations(
        file_path=gs_file_path, 
        k=gs_k, 
        allowed_bands=user_allowed_bands, 
        seed=seed
    )

    stations_supported_bands = set()
    for gs in ground_stations:
        for band in gs.bands_supported:
            stations_supported_bands.add(band)
            
    active_bands_pool = sorted(list(stations_supported_bands))

    if not active_bands_pool:
        raise ValueError("The sampled ground stations do not provide any valid communication bands.")

    active_band_weights = None
    if band_weights_map:
        active_band_weights = []
        for band in active_bands_pool:
            active_band_weights.append(float(band_weights_map.get(band, 1.0)))

    raw_satellites = load_and_sample_satellites(file_path=sat_file_path, group_name=sat_group_name, k=sat_k, seed=seed)

    configured_satellites = assign_satellite_payloads(
        satellites=raw_satellites,
        available_sensors=available_sensors,
        available_bands=active_bands_pool,  
        sensor_weights=sensor_weights,
        band_weights=active_band_weights,     
        min_sensors_per_sat=min_sensors_per_sat,
        max_sensors_per_sat=max_sensors_per_sat,
        seed=seed
    )

    active_boxes = bounding_boxes if bounding_boxes is not None else [
        {"name": "default_envelope", "lat_envelope": [2.0, 8.0], "lon_envelope": [-77.0, -72.0]}
    ]

    targets = generate_dynamic_tasks(
        k=tasks_k,
        bounding_boxes=active_boxes,
        polygon_ratio=polygon_ratio,
        min_area_deg=min_area_deg,
        max_area_deg=max_area_deg,
        min_duration=min_duration,
        max_duration=max_duration,
        min_release_delay=min_release_delay,
        max_release_delay=max_release_delay,
        min_lifetime=min_lifetime,
        max_lifetime=max_lifetime,
        available_sensors=available_sensors,
        priority_weights=priority_weights,
        seed=seed
    )

    context = CollectedContext(
        satellites=configured_satellites,
        ground_stations=ground_stations,
        targets=targets
    )
    
    _export_scenario_report(context)

    return context