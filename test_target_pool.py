import sys
from src.modules.data_collector.target_pool import generate_dynamic_tasks

def main():
    print("=================================================================================")
    print("Testing Procedural Target Generation with Deep Diagnostic Logging...")
    print("=================================================================================")

    mock_boxes = [
        {"name": "colombia_andina", "lat_envelope": [2.0, 8.0], "lon_envelope": [-77.0, -72.0]},
        {"name": "panama_canal", "lat_envelope": [7.5, 9.5], "lon_envelope": [-80.5, -78.5]}
    ]
    
    sensors_pool = ["VIS", "SAR", "TIR"]

    print("\n[Scenario 1] Executing scale, topological closure, and timeline audit...")
    try:
        tasks = generate_dynamic_tasks(
            k=6,
            bounding_boxes=mock_boxes,
            polygon_ratio=0.5,  
            min_area_deg=0.05,
            max_area_deg=0.15,
            min_duration=10,
            max_duration=25,
            min_release_delay=0,
            max_release_delay=600,
            seed=42
        )
        
        assert len(tasks) == 6, f"Expected 6 tasks, got {len(tasks)}"
        
        for t in tasks:
            print("-" * 81)
            print(f"     ID: {t.task_id:<12} | Region: {t.region_tag:<17} | Type: {t.task_type.upper():<7}")
            print(f"     Sensors : {str(t.required_sensors):<20} | Min Duration Required: {t.estimated_duration}s")
            print(f"     Timeline: Release (t_arr): {t.release_time:<4}s | Deadline (t_dl): {t.deadline:<5}s | Lifetime: {t.deadline - t.release_time}s")
            
            # Formateo y auditoría de coordenadas geodésicas (Lat, Lon)
            print("     Coordinates:")
            if t.task_type == "point":
                lat, lon = t.coordinates[0]
                print(f"       * Point Vector -> Lat: {lat:7.4f}° , Lon: {lon:7.4f}°")
            else:
                assert len(t.coordinates) == 5, f"Polygon should have 5 vertices, got {len(t.coordinates)}"
                assert t.coordinates[0] == t.coordinates[-1], "Polygon topology is not strictly closed."
                for idx, (lat, lon) in enumerate(t.coordinates):
                    tag = f"Vertex {idx+1}" if idx < 4 else "Closure "
                    print(f"       * {tag} -> Lat: {lat:7.4f}° , Lon: {lon:7.4f}°")
            
            # Strict validation checkpoints
            assert t.deadline > t.release_time, "Timeline fault: Deadline occurs before release."
            assert (t.deadline - t.release_time) >= t.estimated_duration, "Task lifetime is too short for capture duration."
            
        print("=" * 81)
        print("  SUCCESS: Dynamic relative timelines and geospatial topologies are consistent.")
        print("=" * 81)
        
    except Exception as e:
        print(f"\nScenario 1 FAILED: {e}")
        sys.exit(1)

    print("\n[Scenario 2] Validating absolute deterministic reproducibility (Seed Control)...")
    try:
        run_a = generate_dynamic_tasks(k=5, bounding_boxes=mock_boxes, seed=999)
        run_b = generate_dynamic_tasks(k=5, bounding_boxes=mock_boxes, seed=999)
        
        assert [t.coordinates for t in run_a] == [t.coordinates for t in run_b], "Seeds drifted."
        print("  SUCCESS: Mathematical reproducibility verified. Matrix generation is deterministic.")
        
    except Exception as e:
        print(f"Scenario 2 FAILED: {e}")
        sys.exit(1)

    print("\nAll Target Pool Procedural tests PASSED successfully!")

if __name__ == "__main__":
    main()