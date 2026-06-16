# Module: Physics Engine (Phase 2)

## Overview
The `physics_engine` module is the geometric and orbital propagation core of the **DatasetFactory** system. It ingests the enriched memory buffer (`CollectedContext`) generated during Phase 1 and evaluates the temporal windows of visibility (access passes) between LEO space assets, fixed ground infrastructure, and dynamic target regions.

The output of this module is a comprehensive, deterministic matrix of physical intersection windows (`physics_passes_report.json`), which serves as the hard timeline baseline for downstream scheduling, multi-agent coordination, and Satellite-as-a-Service (SaaS) auction simulation.

---

## Architectural Mechanics & Calculations
The physics engine abstracts the complex, continuous nature of orbital flight into a discretized time-series matrix using high-fidelity geodetic models.

src/modules/physics_engine/
├── init.py
├── main.py              # Sub-orchestrator entrypoint (physics_engine_main)
└── pass_calculator.py   # Analytical geometry and visibility window calculators

### 1. Orbital Propagation (SGP4 Baseline)
The space asset state vectors (Position $\mathbf{r}$ and Velocity $\mathbf{v}$ in the True Equator, Mean Equinox - TEME coordinate frame) are propagated dynamically through time utilizing the standard **SGP4 (Simplified General Perturbations 4)** model. This accounts for secular and periodic orbital variations driven by:
* Earth's oblateness ($J_2$, $J_3$, $J_4$ spherical harmonics).
* Atmospheric drag effects via the TLE $B^*$ (B-star) drag term.
* Lunar-solar gravitational third-body perturbations.

### 2. Infrastructure Access Windows (`compute_infrastructure_passes`)
Calculates the geometric lines-of-sight between LEO satellites and ground stations. 
* **Frequency Link Constraints:** The engine reads the `bands_config` argument. A pass is only registered if there is a frequency intersection (i.e., the satellite's assigned payload band is supported by the specific ground tracking node).
* **Elevation Bounds:** State vectors are transformed from the inertial frame to the local Topocentric Horizon system (AER: Azimuth, Elevation, Range) relative to the ground station's geodetic coordinates (WGS84). A pass starts (AOS) and ends (LOS) when the local elevation angle satisfies the band-specific constraint:
$$\theta_{\text{elevation}} \ge \theta_{\text{min\_elevation}}$$

### 3. Target Intersection Windows (`compute_target_passes`)
Evaluates the coverage windows over dynamic procedural tasks.
* **Point-type Targets:** Modeled as static geodetic coordinate targets. The visibility window is bounded by a baseline sensor-elevation threshold (typically $\ge 10.0^\circ$).
* **Polygon-type Targets:** For complex regions of interest (RoI), the engine evaluates intersection boundaries. The look-ahead window is adjusted by factoring the satellite's ground track linear velocity and a spatial buffer based on the polygon's bounding footprint envelope:
$$\Delta t_{\text{buffer}} \approx \frac{\text{RoI}_{\text{radius\_deg}}}{\omega_{\text{satellite\_angular\_velocity}}}$$
This optimizes the search space by sweeping only the relevant temporal domains where the sensor's cross-track swath can physically footprint the target perimeter.

---

## Configuration & Discretization Control
The propagation loop is fully bounded by absolute datetime anchors and a discretization time step specified in the global orchestrator (`src/main.py`):

* **Simulation Horizon ($t_0 \rightarrow t_f$):** Evaluated across a strict 24-hour baseline window (e.g., from `2026-06-15 12:00:00 UTC` to `2026-06-16 12:00:00 UTC`).
* **Discretization Step (`step_seconds = 20`):** Balances computational efficiency and window edge accuracy. A 20-second step provides high-granularity capture for LEO satellites traveling at $\sim 7.5 \text{ km/s}$, limiting edge discretization error to a safe bound for scheduling optimization.

---

## Physics Report Schema
Upon pipeline phase completion, all computed interactions are compiled into `data/physics_passes_report.json`.

```json
{
  "metadata": {
    "generated_at_utc": "2026-06-16 14:42:00",
    "simulation_start_utc": "2026-06-15 12:00:00",
    "simulation_end_utc": "2026-06-16 12:00:00",
    "total_satellites": 3,
    "total_ground_stations": 2,
    "total_targets": 15,
    "compiled_infrastructure_passes": 42,
    "compiled_target_passes": 118
  },
  "infrastructure_passes": [
    {
      "pass_id": "INFRA_PASS_001",
      "satellite_id": 33591,
      "ground_station_id": "bogota_station",
      "comm_band": "X",
      "aos_utc": "2026-06-15 14:22:10",
      "los_utc": "2026-06-15 14:31:40",
      "duration_seconds": 570,
      "max_elevation_deg": 68.4
    }
  ],
  "target_passes": [
    {
      "pass_id": "TARGET_PASS_001",
      "satellite_id": 33591,
      "task_id": "TASK_GEN_001",
      "aos_utc": "2026-06-15 17:05:20",
      "los_utc": "2026-06-15 17:08:50",
      "duration_seconds": 210,
      "intersection_type": "polygon_swath_overlap"
    }
  ]
}
```

## Dependencies & Math Validation
The module relies heavily on the following core analytical layers:

sgp4: Python implementation of the standard Hoots/Cranford orbital propagation routines.
pyproj / numpy: Vector transformations from Geocentric Inertial (TEME/ECEF) to Topocentric (AER) reference systems using WGS84 ellipsoidal geometry.


## Configuration Parameter Ingestion
The sub-orchestrator (`physics_engine_main`) receives its parameters directly from the global `config.yaml` layout mapped through `src/main.py`.

