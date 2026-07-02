import json
import pathlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from src.core.datatypes import CollectedContext
from .norad_pool import load_and_sample_satellites
from .gs_pool import load_and_sample_stations
from .payload_assign import assign_satellite_payloads
from .target_pool import generate_dynamic_tasks

__all__ = ["data_collector_main"]


def _parse_tle_epoch(tle_line1: str) -> datetime:
    """
    Parses the orbital epoch directly from Line 1 of the standard NORAD TLE format.
    
    Args:
        tle_line1: The first line of the TLE string.
        
    Returns:
        A datetime object set to the exact UTC epoch extracted from the TLE.
    """
    try:
        epoch_str = tle_line1[18:32].strip()
        year_two_digits = int(epoch_str[:2])
        year = 2000 + year_two_digits if year_two_digits < 57 else 1900 + year_two_digits
        day_of_year = float(epoch_str[2:])
        
        base_date = datetime(year, 1, 1, tzinfo=timezone.utc)
        epoch_datetime = base_date + timedelta(days=day_of_year - 1)
        return epoch_datetime
    except Exception:
        return datetime.now(timezone.utc)


def _export_scenario_report(context: CollectedContext, report_path: str = "data/scenario_report.json") -> None:
    path = pathlib.Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    report_data = {
        "metadata": {
            "total_satellites": len(context.satellites),
            "total_ground_stations": len(context.ground_stations),
            "total_target_tasks": len(context.targets),
            "anchored_tle_epoch_utc": context.tle_epoch_utc.strftime("%Y-%m-%d %H:%M:%S") if getattr(context, "tle_epoch_utc", None) else None
        },
        "satellites": [
            {
                "norad_id": sat.norad_id,
                "name": sat.name,
                "tle_line1": sat.tle_line1,
                "tle_line2": sat.tle_line2,
                "assigned_band": sat.band,
                "assigned_sensors": sat.sensors,
                "ssr_capacity_mb": sat.capacity_mb,
                "sensor_generation_rates": sat.sensor_generation_rates,
                "downlink_rate_mb_s": sat.downlink_rate_mb_s
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
                "coordinates": t.coordinates,
                "sensor_requirements": t.required_sensors
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
    storage_capacity_pool_mb: Optional[List[float]] = None,
    sensor_generation_rates: Optional[Dict[str, float]] = None,
    tasks_k: int = 10,
    bounding_boxes: Optional[List[Dict[str, Any]]] = None,
    polygon_ratio: float = 0.5,
    min_area_deg: float = 0.05,
    max_area_deg: float = 0.20,
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
    seed: Optional[int] = None,
    output_path: str = "data/scenario_report.json",
    **kwargs
) -> CollectedContext:
    """
    Orchestrates the metadata collection phase by parsing physical assets,
    assigning stochastic configurations, generating dynamic target regions,
    and dynamically extracting the tracking timeline from the underlying TLE epoch.
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

    active_band_weights = []
    band_downlink_rates = {}
    if band_weights_map:
        for band in active_bands_pool:
            band_info = band_weights_map.get(band, {})
            active_band_weights.append(float(band_info.get("weight", 1.0)))
            band_downlink_rates[band] = float(band_info.get("downlink_rate_mb_s", 10.0))
    else:
        active_band_weights = [1.0] * len(active_bands_pool)

    raw_satellites = load_and_sample_satellites(file_path=sat_file_path, group_name=sat_group_name, k=sat_k, seed=seed)

    configured_satellites = assign_satellite_payloads(
        satellites=raw_satellites,
        available_sensors=available_sensors,
        available_bands=active_bands_pool,  
        available_capacities=storage_capacity_pool_mb,
        sensor_weights=sensor_weights,
        band_weights=active_band_weights,     
        sensor_generation_rates=sensor_generation_rates,
        band_downlink_rates=band_downlink_rates,
        min_sensors_per_sat=min_sensors_per_sat,
        max_sensors_per_sat=max_sensors_per_sat,
        seed=seed
    )

    t0_dynamic = datetime.now(timezone.utc)
    if configured_satellites:
        t0_dynamic = _parse_tle_epoch(configured_satellites[0].tle_line1)

    active_boxes = bounding_boxes if bounding_boxes is not None else [
        {"name": "default_envelope", "lat_envelope": [2.0, 8.0], "lon_envelope": [-77.0, -72.0]}
    ]

    targets = generate_dynamic_tasks(
        k=tasks_k,
        bounding_boxes=active_boxes,
        polygon_ratio=polygon_ratio,
        min_area_deg=min_area_deg,
        max_area_deg=max_area_deg,
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
    context.tle_epoch_utc = t0_dynamic
    
    _export_scenario_report(context, report_path=output_path)

    return context