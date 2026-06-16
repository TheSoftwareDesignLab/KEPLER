import random
from typing import List, Optional

__all__ = ["assign_satellite_payloads"]


def assign_satellite_payloads(
    satellites: List[any],
    available_sensors: List[str],
    available_bands: List[str],
    sensor_weights: Optional[List[float]] = None,
    band_weights: Optional[List[float]] = None,
    min_sensors_per_sat: int = 1,
    max_sensors_per_sat: int = 1,
    seed: Optional[int] = None
) -> List[any]:
    """
    Stochastically assigns communication bands and sensor payloads to a collection 
    of satellites based on explicit user constraints and custom probability distributions.
    
    :param satellites: List of SatelliteConfig objects to configure.
    :param available_sensors: Explicit list of sensor types allowed.
    :param available_bands: Explicit list of communication bands allowed.
    :param sensor_weights: Optional probability weights associated with each sensor.
    :param band_weights: Optional probability weights associated with each communication band.
    :param min_sensors_per_sat: Minimum number of sensors assigned to a single asset.
    :param max_sensors_per_sat: Maximum number of sensors assigned to a single asset.
    :param seed: Random seed for experimental reproducibility.
    :return: List of fully payload-configured SatelliteConfig dataclasses.
    """
    if not satellites:
        return []
        
    if not available_sensors or not available_bands:
        raise ValueError("Available sensors and bands configuration pools cannot be empty.")
        
    if min_sensors_per_sat < 1 or max_sensors_per_sat > len(available_sensors) or min_sensors_per_sat > max_sensors_per_sat:
        raise ValueError("Invalid sensor allocation bounds provided for payload assignment.")

    if sensor_weights is not None and len(sensor_weights) != len(available_sensors):
        raise ValueError("Sensor weights layout must strictly match the size of available sensors pool.")
        
    if band_weights is not None and len(band_weights) != len(available_bands):
        raise ValueError("Band weights layout must strictly match the size of available bands pool.")

    rng = random.Random(seed)
    
    for sat in satellites:
        sat.band = rng.choices(available_bands, weights=band_weights, k=1)[0]
        
        num_sensors = rng.randint(min_sensors_per_sat, max_sensors_per_sat)
        
        chosen_sensors = []
        current_sensors = list(available_sensors)
        current_weights = list(sensor_weights) if sensor_weights is not None else [1.0] * len(available_sensors)
        
        while len(chosen_sensors) < num_sensors:
            pick = rng.choices(current_sensors, weights=current_weights, k=1)[0]
            chosen_sensors.append(pick)
            
            idx = current_sensors.index(pick)
            current_sensors.pop(idx)
            current_weights.pop(idx)
            
        sat.sensors = chosen_sensors
        
    return satellites