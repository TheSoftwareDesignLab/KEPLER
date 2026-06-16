import sys
from datetime import datetime, timezone
from src.core.datatypes import SatelliteConfig, GroundStationConfig, TargetTask
from src.modules.physics_engine.pass_calculator import compute_infrastructure_passes, compute_target_passes


def main():
    print("=========================================================================")
    print("Testing Vectorized Orbital Pass Calculator & Target Window Intersection Engine")
    print("=========================================================================")

    mock_sat = SatelliteConfig(
        norad_id=33591,
        name="NOAA 19",
        tle_line1="1 33591U 09005A   26166.45267156  .00000109  00000-0  96637-4 0  9998",
        tle_line2="2 33591  98.7122  63.5359 0014232 214.3418 145.7196 14.12423987893910"
    )
    
    mock_sat.band = "X"

    mock_gs = GroundStationConfig(
        id="bogota_station",
        name="Bogota Tracking Node",
        latitude=4.6097,
        longitude=-74.0817,
        elevation=2640.0,
        bands_supported=["S", "X"]
    )

    mock_task = TargetTask(
        task_id="TASK_VALID_001",
        region_tag="colombia_andina",
        priority=4,
        task_type="polygon",
        coordinates=[
            (4.7000, -74.1000),
            (4.7000, -73.9000),
            (4.5000, -73.9000),
            (4.5000, -74.1000),
            (4.7000, -74.1000)
        ],
        required_sensors=["VIS"],
        release_time=0,
        deadline=86400
    )

    mock_bands_config = {
        "X": {
            "weight": 0.6,
            "min_elevation_deg": 10.0
        },
        "S": {
            "weight": 0.2,
            "min_elevation_deg": 5.0
        },
        "Ka": {
            "weight": 0.2,
            "min_elevation_deg": 15.0
        }
    }

    t0 = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    tf = datetime(2026, 6, 16, 12, 0, 0, tzinfo=timezone.utc)

    print("\n[Scenario 1] Computing Vectorized Station Infrastructure Passes...")
    try:
        gs_passes = compute_infrastructure_passes(
            satellite=mock_sat,
            ground_station=mock_gs,
            start_dt_utc=t0,
            end_dt_utc=tf,
            bands_config=mock_bands_config,
            step_seconds=20
        )
        
        print(f"  SUCCESS: Discovered {len(gs_passes)} valid passes over station footprint.")
        for p in gs_passes[:2]:
            print(f"    * Sat: {p['satellite_id']} | GS: {p['ground_station_id']} | AOS: {p['aos_utc']} | LOS: {p['los_utc']} | Duration: {p['duration_s']}s | Max El: {p['max_el_deg']:.2f}°")
            
            assert p["duration_s"] > 0, "Pass duration calculation drifted below threshold."
            assert p["max_el_deg"] >= 10.0, "Elevation constraint violated."
            
    except Exception as e:
        print(f"  [CRASH] Scenario 1 simulation failed: {e}")
        sys.exit(1)

    print("\n[Scenario 2] Computing Task Centroid Overlapping Target Passes...")
    try:
        task_passes = compute_target_passes(
            satellite=mock_sat,
            task=mock_task,
            simulation_start_utc=t0,
            min_el_deg=10.0,
            step_seconds=20
        )
        
        print(f"  SUCCESS: Discovered {len(task_passes)} feasible windows over procedural target.")
        for p in task_passes[:2]:
            print(f"    * Sat: {p['satellite_id']} | Task: {p['task_id']} | AOS: {p['aos_utc']} | LOS: {p['los_utc']} | Duration: {p['duration_s']}s")
            
            assert p["is_feasible"] is True, "Feasibility flag error."
            assert p["duration_s"] >= mock_task.estimated_duration, "Pass duration clipped below required payload exposure threshold."
            
    except Exception as e:
        print(f"  [CRASH] Scenario 2 simulation failed: {e}")
        sys.exit(1)

    print("\n=========================================================================")
    print("All Physics Engine Pass Calculator tests PASSED successfully!")
    print("=========================================================================")


if __name__ == "__main__":
    main()