import sys
import pathlib
import json
from datetime import datetime, timezone
yaml_available = True
try:
    import yaml
except ImportError:
    yaml_available = False
from src.modules.data_collector.main import data_collector_main
from src.modules.physics_engine.main import physics_engine_main
from src.modules.prompt_factory.main import prompt_factory_main


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


def load_semantic_categories(categories_path: str) -> dict:
    p = pathlib.Path(categories_path)
    if not p.exists():
        raise FileNotFoundError(f"Semantic categories JSON file not found at: {categories_path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("==================================================")
    print("Launching DatasetFactory Bulk Generation Pipeline...")
    print("==================================================")
    
    CONFIG_FILE = "config.yaml"
    CATEGORIES_FILE = "semantic_categories.json"
    
    try:
        cfg = load_config(CONFIG_FILE)
        sem_categories = load_semantic_categories(CATEGORIES_FILE)
        
        sim_cfg = cfg.get("simulation", {})
        pay_cfg = cfg.get("payload", {})
        task_cfg = cfg.get("task_generation", {})
        path_cfg = cfg.get("paths", {})
        
        dataset_name = sim_cfg.get("dataset_name", "dataset_output")
        num_scenarios = sim_cfg.get("num_scenarios", 1)
        base_seed = sim_cfg.get("seed", 42)
        
        bands_config = pay_cfg.get("bands_config", {})
        band_map = {b: info.get("weight", 1.0) for b, info in bands_config.items()}
        
        print(f"[DATASET] Parent Dataset Directory: 'data/{dataset_name}'")
        print(f"[DATASET] Total iterations to generate: {num_scenarios}\n")

        for idx in range(1, num_scenarios + 1):
            scenario_folder_name = f"scenario_{idx}"
            scenario_dir = pathlib.Path("data") / dataset_name / scenario_folder_name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            
            current_seed = base_seed + idx if base_seed is not None else None
            
            print("\n" + "#" * 60)
            print(f" GENERATING ITERATION {idx}/{num_scenarios}: '{scenario_folder_name}'")
            print(f" Target Folder: {scenario_dir.resolve()}")
            print(f" Active Seed: {current_seed}")
            print("#" * 60)
            
            print(f"\nExecuting Phase 1: Data Collection & Asset Enrichment...")
            
            scenario_report_path = scenario_dir / "scenario_report.json"

            collector_kwargs = {
                "sat_k": sim_cfg.get("sat_k"),
                "gs_k": sim_cfg.get("gs_k"),
                "tasks_k": sim_cfg.get("tasks_k", 10),
                "bounding_boxes": task_cfg.get("bounding_boxes"),
                "polygon_ratio": task_cfg.get("polygon_ratio", 0.5),
                "min_area_deg": task_cfg.get("min_area_deg", 0.05),
                "max_area_deg": task_cfg.get("max_area_deg", 0.20),
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
                "priority_weights": task_cfg.get("priority_weights"),  
                "seed": current_seed,
                "output_path": str(scenario_report_path)
            }
            
            if sim_cfg.get("sat_group_name"):
                collector_kwargs["sat_group_name"] = sim_cfg.get("sat_group_name")
            else:
                collector_kwargs["sat_file_path"] = path_cfg.get("sat_file_path")
                
            context = data_collector_main(**collector_kwargs)
            
            print(f"Phase 1 SUCCESSFUL. Generated {len(context.targets)} Targets for current iteration.")
            
            print("\nExecuting Semantic Prompt Generation Phase (Ollama Inferences & Semantic Embedding Validation)...")
            
            prompt_cfg = task_cfg.get("prompt_generation", {})
            ollama_model = sim_cfg.get("ollama_model", "llama3.1:8b")
            ollama_temp = sim_cfg.get("ollama_temperature", 0.3)
            
            prompt_factory_main(
                targets=context.targets,
                prompt_config=prompt_cfg,
                output_dir=str(scenario_dir),  
                model_name=ollama_model,
                temperature=ollama_temp,
                sensor_categories=sem_categories.get("sensor_categories"),
                priority_categories=sem_categories.get("priority_categories"),
                days_categories=sem_categories.get("days_categories"),
                hours_categories=sem_categories.get("hours_categories")
            )
            
            print("\nExecuting Phase 2: Physics Matrix Propagation & Target Intersection...")
            
            t0 = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
            tf = datetime(2026, 6, 16, 12, 0, 0, tzinfo=timezone.utc)
            
            physics_report_path = scenario_dir / "physics_passes_report.json"
            
            physics_engine_main(
                context=context,
                bands_config=bands_config,
                simulation_start_utc=t0,
                simulation_end_utc=tf,
                output_path=str(physics_report_path), 
                step_seconds=20
            )
            
            print(f"Iteration '{scenario_folder_name}' processed and isolated.")
            
        print("\n==================================================")
        print(f"Global Bulk Generation SUCCESSFUL. Total Scenarios Built: {num_scenarios}")
        print("==================================================")
            
    except Exception as e:
        print(f"\nPipeline CRASHED during setup or execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()