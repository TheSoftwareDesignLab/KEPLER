from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from datetime import datetime

@dataclass
class SatelliteConfig:
    norad_id: int
    name: str
    tle_line1: str
    tle_line2: str
    band: Optional[str] = None
    sensors: List[str] = field(default_factory=list)
    capacity_mb: Optional[float] = None
    sensor_generation_rates: Dict[str, float] = field(default_factory=dict)
    downlink_rate_mb_s: Optional[float] = None

@dataclass
class GroundStationConfig:
    id: str
    name: str
    latitude: float
    longitude: float
    elevation: float
    bands_supported: List[str]

@dataclass
class TargetTask:
    task_id: str
    region_tag: str
    priority: int
    task_type: str                 
    coordinates: List[Tuple[float, float]] 
    required_sensors: List[str]
    release_time: int               
    deadline: int      
    assigned_lvlh_pitch_deg: Optional[float] = None
    assigned_lvlh_roll_deg: Optional[float] = None             

@dataclass
class CollectedContext:
    satellites: List[SatelliteConfig] = field(default_factory=list)
    ground_stations: List[GroundStationConfig] = field(default_factory=list)
    targets: List[TargetTask] = field(default_factory=list)
    tle_epoch_utc: Optional[datetime] = None