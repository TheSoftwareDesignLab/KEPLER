# Module: Data Collector (Phase 1)

## Overview
The `data_collector` module serves as the automated ingest and asset-enrichment pipeline for the **DatasetFactory** ecosystem. It is responsible for gathering real-world aerospace and geodetic data, applying stochastic payload configurations, and generating procedural mission targets.

The primary objective of this module is to construct a deterministic, memory-buffered `CollectedContext` object and export a `scenario_report.json` file. These assets establish the grounding telemetry required for orbital mechanics propagation and multi-agent schedule optimization.

---

## Architecture & Sub-Modules
The module is decoupled into localized, single-responsibility components:

src/modules/data_collector/
├── init.py
├── main.py              # Sub-orchestrator entrypoint (data_collector_main)
├── celestrak_handler.py # Network client for live NORAD TLE fetching
├── norad_pool.py        # TLE parser and strict LEO filtering engine
├── gs_pool.py           # Ground station CSV ingestion and band-filtering
├── payload_assign.py    # Stochastic payload assignment (Bands & Sensors)
└── target_pool.py       # Procedural point/polygon task generation

### 1. Asset Ingestion & Filtering
* **`norad_pool.py`**: Ingests space assets via specific NORAD ID text lists or CelesTrak group endpoints. It implements an explicit **Mean Motion filter** ($n \ge 11.0$ rev/day) directly on column slices of the TLE Line 2. This safely scrubs non-LEO assets (such as GEO or MEO meteorological nodes) to prevent downstream window calculation distortions.
* **`gs_pool.py`**: Reads local pre-enriched ground station registries (`ground_station.csv`). It performs an intersection lookup against user-allowed communication frequencies, discarding nodes that do not match the baseline profile.

### 2. Payload Assignment
* **`payload_assign.py`**: Combines user-configured sensor pools (`VIS`, `SAR`, `TIR`) and active communication bands. It uses weighted random selection (`rng.choices`) powered by seeds to deterministically bind communication channels and payload capacities onto space assets.

### 3. Procedural Task Generation
* **`target_pool.py`**: Synthesizes a configurable amount of localized imaging targets (`TargetTask`) constrained inside bounding box envelopes.
* **Point Tasks**: Modeled via unique geographic coordinate nodes.
* **Polygon Tasks**: Modeled as squared Regions of Interest (RoI) with closed perimeter topologies.
* **Priority Distribution**: Implements a non-equitable distribution via configurable priority weights. This ensures that routine tasks (Priority 1–2) dominate the buffer density, while critical tasks (Priority 5) remain scarce, setting up realistic stress tests for load-balancing solvers.

---

## Configuration Parameter Ingestion
The sub-orchestrator (`data_collector_main`) receives its parameters directly from the global `config.yaml` layout mapped through `src/main.py`.


## Data Output Structure
Upon a successful execution loop, the module serializes its memory buffers into data/scenario_report.json. This schema contains full structural definitions:

```json
  {
  "metadata": {
    "total_satellites": 3,
    "total_ground_stations": 2,
    "total_target_tasks": 15
  },
  "satellites": [
    {
      "norad_id": 33591,
      "name": "NOAA 19",
      "tle_line1": "1 33591U 09005A...",
      "tle_line2": "2 33591  98.7122...",
      "assigned_band": "X",
      "assigned_sensors": ["VIS"]
    }
  ],
  "ground_stations": [
    {
      "id": "bogota_station",
      "name": "Bogota Tracking Node",
      "latitude": 4.6097,
      "longitude": -74.0817,
      "elevation_m": 2640.0,
      "bands_supported": ["X"]
    }
  ],
  "targets": [
    {
      "task_id": "TASK_GEN_001",
      "region_tag": "colombia_andina",
      "priority": 1,
      "task_type": "polygon",
      "estimated_duration_s": 22,
      "release_time_s": 142,
      "deadline_s": 3840,
      "coordinates": [
        [4.712, -74.120],
        [4.712, -73.910],
        [4.490, -73.910],
        [4.490, -74.120],
        [4.712, -74.120]
      ]
    }
  ]
}

```

## Execution & Testing
The data collection pipeline runs natively as Phase 1 inside the global runner script. To execute or verify standalone module consistency, invoke the pipeline launcher:

python -m src.main