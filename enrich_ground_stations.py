import csv
import json
import pathlib
import ssl
import time
import urllib.request
import urllib.error

ORIGINAL_CSV = "data/ground_station_original.csv"
OUTPUT_CSV = "data/ground_station.csv"


def _fetch_chunk_elevations(chunk: list) -> dict:
    """
    Queries Open Topo Data using a single batched payload of up to 100 locations.
    """
    loc_string = "|".join(f"{lat},{lon}" for lat, lon in chunk)
    url = f"https://api.opentopodata.org/v1/srtm30m?locations={loc_string}"
    
    ssl_context = ssl._create_unverified_context()
    elevation_map = {}
    
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=15, context=ssl_context) as response:
            data = json.loads(response.read().decode())
            results = data.get("results", [])
            for res in results:
                loc = res.get("location", {})
                lat, lon = loc.get("lat"), loc.get("lng")
                elev = res.get("elevation")
                if lat is not None and lon is not None and elev is not None:
                    elevation_map[(float(lat), float(lon))] = float(elev)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
        print(f"    [ERROR] Batch network lookup failed: {e}")
        
    return elevation_map


def main():
    print("=================================================================")
    print("Running Isolated Ground Station Elevation Enrichment Preprocessor")
    print("=================================================================")
    
    input_path = pathlib.Path(ORIGINAL_CSV)
    output_path = pathlib.Path(OUTPUT_CSV)
    
    if not input_path.exists():
        print(f"[FATAL] Source file missing at: {input_path.resolve()}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Load coordinates from source file
    rows = []
    coordinates = []
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
            if row.get("Latitude") and row.get("Longitude"):
                coordinates.append((float(row["Latitude"]), float(row["Longitude"])))
                
    print(f"  - Successfully ingested {len(rows)} candidate stations from source file.")
    print("  - Launching batched remote elevation lookups (Chunk size: 100)...")
    
    # 2. Process locations in sequential batches to prevent API rate limiting
    chunk_size = 100
    global_elevation_registry = {}
    
    for i in range(0, len(coordinates), chunk_size):
        chunk = coordinates[i:i + chunk_size]
        print(f"    * Requesting batch {(i // chunk_size) + 1} ({len(chunk)} locations)...")
        
        batch_results = _fetch_chunk_elevations(chunk)
        global_elevation_registry.update(batch_results)
        
        # Enforce a non-aggressive temporal back-off delay to protect the endpoint
        if i + chunk_size < len(coordinates):
            time.sleep(2.0)
            
    # 3. Compile and rewrite the enriched database
    if "Elevation" not in fieldnames:
        fieldnames.append("Elevation")
        
    print(f"  - Writing finalized enriched dataset out to: {output_path}")
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in rows:
            if row.get("Latitude") and row.get("Longitude"):
                lat_val = float(row["Latitude"])
                lon_val = float(row["Longitude"])
                row["Elevation"] = global_elevation_registry.get((lat_val, lon_val), 0.0)
            else:
                row["Elevation"] = 0.0
                
            writer.writerow(row)
            
    print("=================================================================")
    print("Processing Completed Successfully. Dataset Offline and Ready.")
    print("=================================================================")


if __name__ == "__main__":
    main()