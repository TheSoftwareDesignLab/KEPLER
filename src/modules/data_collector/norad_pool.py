import pathlib
import random
from typing import List, Optional
from src.core.datatypes import SatelliteConfig
from .celestrak_handler import fetch_celestrak_metadata, fetch_group_from_celestrak

__all__ = ["load_and_sample_satellites"]


def _read_local_file(file_path: str) -> List[int]:
    p = pathlib.Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"The specified NORAD pool file does not exist at: {file_path}")
    
    ids = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            clean_id = line.split("#")[0].strip()
            ids.append(int(clean_id))
        except ValueError:
            raise ValueError(f"Invalid NORAD ID format encountered in file {file_path} at line: '{line}'")
            
    if not ids:
        raise ValueError(f"The NORAD pool file at {file_path} is empty or contains no valid IDs.")
    return ids


def _is_strict_leo(tle_line2: str, min_mean_motion: float = 11.0) -> bool:
    """
    Parses the Mean Motion field (revolutions per day) from the standard 
    NORAD TLE format (Line 2, characters 52-63) to isolate LEO orbits.
    """
    try:
        # Standard TLE structural column slice for Mean Motion
        mean_motion_str = tle_line2[52:63].strip()
        mean_motion_val = float(mean_motion_str)
        return mean_motion_val >= min_mean_motion
    except (ValueError, IndexError):
        return False


def load_and_sample_satellites(
    file_path: Optional[str] = None,
    group_name: Optional[str] = None,
    k: Optional[int] = None,
    seed: Optional[int] = None,
    custom_satellites: Optional[List[SatelliteConfig]] = None
) -> List[SatelliteConfig]:
    """
    Loads real satellite orbital data strictly driven by user configuration inputs,
    filtering the final pool to retain only valid LEO operational assets.
    """
    if custom_satellites is not None:
        raw_pool = custom_satellites
    elif file_path is not None:
        local_ids = _read_local_file(file_path)
        raw_pool = []
        for norad_id in local_ids:
            metadata = fetch_celestrak_metadata(norad_id)
            if metadata:
                raw_pool.append(
                    SatelliteConfig(
                        norad_id=metadata["norad_id"],
                        name=metadata["name"],
                        tle_line1=metadata["tle_line1"],
                        tle_line2=metadata["tle_line2"]
                    )
                )
            else:
                raise RuntimeError(f"Failed to fetch metadata from CelesTrak for target NORAD ID: {norad_id}")
    elif group_name is not None:
        raw_pool = fetch_group_from_celestrak(group_name)
    else:
        raise ValueError("User must explicitly provide either 'custom_satellites', 'file_path', or 'group_name'.")

    # Strict LEO filtering layer via Mean Motion validation
    final_pool = [sat for sat in raw_pool if _is_strict_leo(sat.tle_line2)]

    if k is None:
        raise ValueError("The sample size parameter 'k' must be explicitly defined to ensure data factory scale control.")

    if not final_pool:
        raise ValueError("The resulting filtered LEO satellite pool is empty. No assets match the required orbital bounds.")

    if k > len(final_pool):
        raise ValueError(
            f"Requested sample size k={k} exceeds the available LEO filtered pool size of {len(final_pool)} satellites. "
            f"Note: Non-LEO assets (like GEO meteorological nodes) were safely scrubbed from the active pool."
        )

    rng = random.Random(seed)
    return rng.sample(final_pool, k)