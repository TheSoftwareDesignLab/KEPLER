import requests
from typing import Dict, Any, List, Optional
from src.core.datatypes import SatelliteConfig

__all__ = ["fetch_celestrak_metadata", "fetch_group_from_celestrak"]

CELESTRAK_GP_URL = "https://celestrak.org/NORAD/elements/gp.php"

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/plain, text/html, application/xhtml+xml, application/xml;q=0.9, image/webp, */*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5"
}


def fetch_celestrak_metadata(norad_id: int) -> Optional[Dict[str, Any]]:
    """
    Queries the CelesTrak General Perturbations (GP) API to retrieve
    updated orbital elements and metadata for a specific NORAD ID.
    Supports both 3-line and strict 2-line response streams.
    
    :param norad_id: The NORAD catalog number of the satellite.
    :return: A dictionary containing satellite metadata and TLE lines, 
             or None if the request fails.
    """
    url = f"{CELESTRAK_GP_URL}?CATNR={norad_id}&FORMAT=2LE"
    try:
        response = requests.get(url, headers=HTTP_HEADERS, timeout=10)
        if response.status_code != 200:
            return None
            
        lines = [line.strip() for line in response.text.splitlines() if line.strip()]
        
        if len(lines) == 2 and lines[0].startswith("1 ") and lines[1].startswith("2 "):
            return {
                "norad_id": norad_id,
                "name": f"NORAD_{norad_id}",
                "tle_line1": lines[0],
                "tle_line2": lines[1]
            }
            
        if len(lines) >= 3:
            return {
                "norad_id": norad_id,
                "name": lines[0],
                "tle_line1": lines[1],
                "tle_line2": lines[2]
            }
            
        return None
    except (requests.RequestException, ValueError):
        return None


def fetch_group_from_celestrak(group_name: str) -> List[SatelliteConfig]:
    """
    Queries the CelesTrak repository to download and strictly parse 
    an entire thematic text group of operational satellites.
    
    :param group_name: The CelesTrak group file name (e.g., 'weather', 'stations', 'earth-resources').
    :return: List of parsed SatelliteConfig dataclasses.
    """
    url = f"https://celestrak.org/NORAD/elements/{group_name}.txt"
    satellites = []
    
    try:
        response = requests.get(url, headers=HTTP_HEADERS, timeout=15)
        if response.status_code != 200:
            raise requests.HTTPError(f"CelesTrak Group Repository responded with status code {response.status_code}")
            
        raw_lines = response.text.splitlines()
        i = 0
        while i < len(raw_lines):
            line = raw_lines[i].strip()
            if not line:
                i += 1
                continue
                
            if not line.startswith("1 ") and not line.startswith("2 ") and (i + 2 < len(raw_lines)):
                name = line
                l1 = raw_lines[i+1].strip()
                l2 = raw_lines[i+2].strip()
                
                if l1.startswith("1 ") and l2.startswith("2 "):
                    try:
                        nid = int(l1[2:7])
                        satellites.append(
                            SatelliteConfig(
                                norad_id=nid,
                                name=name,
                                tle_line1=l1,
                                tle_line2=l2
                            )
                        )
                    except ValueError:
                        pass
                i += 3
            else:
                i += 1
                
    except requests.RequestException as e:
        raise ConnectionError(f"Failed to connect to CelesTrak Group API due to network constraints: {e}")
        
    if not satellites:
        raise ValueError(f"CelesTrak returned no valid satellite data for the group file: '{group_name}.txt'")
        
    return satellites