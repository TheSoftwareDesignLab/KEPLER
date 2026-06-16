import sys
import pathlib
from datetime import datetime, timezone
yaml_available = True
try:
    import yaml
except ImportError:
    yaml_available = False
from src.modules.data_collector.main import data_collector_main
from src.modules.physics_engine.main import physics_engine_main


def load_config(config_path: str) -> dict:
    if not yaml_available:
        raise ImportError("The 'pyyaml' package is required to parse YAML configurations. Run 'pip install pyyaml'.")
    p = pathlib.Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with p.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not config:
        raise ValueError("The configuration file is empty.")
    return config


def main():
    print("==================================================")
    print("Launching DatasetFactory Global Pipeline...")
    
    CONFIG_FILE = "config.yaml"
    
    try:
        cfg = load_config(CONFIG_FILE)
        sim_cfg = cfg.get("simulation", {})
        pay_cfg = cfg.get("payload", {})
        task_cfg = cfg.get("task_generation", {})
        path_cfg = cfg.get("paths", {})
        
        bands_config = pay_cfg.get("bands_config", {})
        band_map = {b: info.get("weight", 1.0) for b, info in bands_config.items()}
        
        print(f"\nExecuting Phase 1: Data Collection & Asset Enrichment using '{CONFIG_FILE}'...")
        
        collector_kwargs = {
            "sat_k": sim_cfg.get("sat_k"),
            "gs_k": sim_cfg.get("gs_k"),
            "tasks_k": sim_cfg.get("tasks_k", 10),
            "bounding_boxes": task_cfg.get("bounding_boxes"),
            "polygon_ratio": task_cfg.get("polygon_ratio", 0.5),
            "min_area_deg": task_cfg.get("min_area_deg", 0.05),
            "max_area_deg": task_cfg.get("max_area_deg", 0.20),
            "min_duration": task_cfg.get("min_duration", 5),
            "max_duration": task_cfg.get("max_duration", 30),
            "min_release_delay": task_cfg.get("min_release_delay", 0),
            "max_release_delay": task_cfg.get("max_release_delay", 3600),
            "min_lifetime": task_cfg.get("min_lifetime", 1800),
            "max_lifetime": task_cfg.get("max_lifetime", 7200),
            "gs_file_path": path_cfg.get("gs_file_path", "data/ground_station.csv"),
            "available_sensors": pay_cfg.get("sensors_pool"),
            "sensor_weights": pay_cfg.get("sensor_weights"),
            "band_weights_map": band_map,  
            "min_sensors_per_sat": pay_cfg.get("min_sensors_per_sat", 1),
            "max_sensors_per_sat": pay_cfg.get("max_sensors_per_sat", 1),
            "priority_weights": task_cfg.get("priority_weights"),  # Mapeo directo del vector de prioridades desde YAML
            "seed": sim_cfg.get("seed")
        }
        
        if sim_cfg.get("sat_group_name"):
            collector_kwargs["sat_group_name"] = sim_cfg.get("sat_group_name")
        else:
            collector_kwargs["sat_file_path"] = path_cfg.get("sat_file_path")
            
        context = data_collector_main(**collector_kwargs)
        
        print("\nPhase 1 Execution SUCCESSFUL. Context Summary:")
        print("-" * 50)
        print(f"  - Active Satellites Collected: {len(context.satellites)}")
        for sat in context.satellites:
            print(f"    * {sat.name:<20} (NORAD {sat.norad_id:<5}) | Band: {sat.band:<3} | Sensors: {sat.sensors}")
            
        print(f"\n  - Ground Stations Sampled: {len(context.ground_stations)}")
        for gs in context.ground_stations:
            print(f"    * {gs.name:<25} | ID: {gs.id:<10} | Supported Bands: {gs.bands_supported}")
            
        print(f"\n  - Procedural Target Tasks Generated: {len(context.targets)}")
        point_tasks = [t for t in context.targets if t.task_type == "point"]
        poly_tasks = [t for t in context.targets if t.task_type == "polygon"]
        print(f"    * Point-type targets   : {len(point_tasks)}")
        print(f"    * Polygon-type targets : {len(poly_tasks)}")
        
        if context.targets:
            print("    * Sample Meta (First 3 Tasks):")
            for t in context.targets[:3]:
                print(f"      > [{t.task_id}] Region: {t.region_tag:<16} | {t.task_type.upper():<7} | Windows: [{t.release_time}s -> {t.deadline}s] | Priority: {t.priority}")
            if len(context.targets) > 3:
                print(f"      > ... and {len(context.targets) - 3} more structured target tasks in memory buffer.")
        
        print("\n" + "=" * 50)
        print("Executing Phase 2: Physics Matrix Propagation & Target Intersection...")
        print("=" * 50)
        
        # Absolute timeline boundaries (24-hour baseline horizon)
        t0 = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        tf = datetime(2026, 6, 16, 12, 0, 0, tzinfo=timezone.utc)
        
        physics_engine_main(
            context=context,
            bands_config=bands_config,
            simulation_start_utc=t0,
            simulation_end_utc=tf,
            output_path="data/physics_passes_report.json",
            step_seconds=20
        )
        
        print("\nGlobal Pipeline Execution completed SUCCESSFUL.")
            
    except Exception as e:
        print(f"\nPipeline CRASHED during setup or execution: {e}")
        sys.exit(1)

    print("==================================================")


if __name__ == "__main__":
    main()