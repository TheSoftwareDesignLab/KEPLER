import csv
import pathlib
import random
from typing import List, Optional
from src.core.datatypes import GroundStationConfig

__all__ = ["load_and_sample_stations"]


def _slugify(text: str) -> str:
    clean = "".join(c.lower() if c.isalnum() or c.isspace() else "" for c in text)
    return "_".join(clean.split())


def load_and_sample_stations(
    file_path: str = "data/ground_station.csv",
    k: Optional[int] = None,
    allowed_bands: Optional[List[str]] = None,
    seed: Optional[int] = None,
    custom_stations: Optional[List[GroundStationConfig]] = None
) -> List[GroundStationConfig]:
    """
    Loads ground stations locally from a pre-enriched CSV file container, strictly 
    filtering the asset pool to matching components present inside allowed_bands.
    """
    if custom_stations is not None:
        return custom_stations

    p = pathlib.Path(file_path)
    if not p.exists():
        return []

    target_bands_set = set(allowed_bands) if allowed_bands is not None else None

    stations = []
    try:
        with open(p, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row.get("Latitude") or not row.get("Longitude"):
                    continue
                    
                raw_bands = row.get("Bands")
                if raw_bands and str(raw_bands).strip().lower() != "nan":
                    bands = [b.strip() for b in raw_bands.split(",")]
                else:
                    bands = ["S", "X"]
                    
                if target_bands_set is not None:
                    matching_bands = sorted(list(set(bands).intersection(target_bands_set)))
                    if not matching_bands:
                        continue
                    bands = matching_bands

                name_str = row["Name"].strip()
                
                raw_elevation = row.get("Elevation")
                try:
                    elevation_val = float(raw_elevation) if raw_elevation else 0.0
                except ValueError:
                    elevation_val = 0.0

                stations.append(
                    GroundStationConfig(
                        id=_slugify(name_str),
                        name=name_str,
                        latitude=float(row["Latitude"]),
                        longitude=float(row["Longitude"]),
                        elevation=elevation_val,
                        bands_supported=bands
                    )
                )
    except (KeyError, ValueError, csv.Error) as e:
        print(f"  [ERROR] Failed to parse local ground station parameters: {e}")
        return []

    if not stations:
        return []

    rng = random.Random(seed)
    if k is None or k > len(stations):
        k = rng.randint(min(2, len(stations)), min(10, len(stations)))
        
    return rng.sample(stations, k)