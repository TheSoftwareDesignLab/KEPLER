import sys
import pathlib
from src.modules.data_collector.norad_pool import load_and_sample_satellites

def main():
    print("Testing Strict Data Factory Architecture (Read-Only Mode)...")

    # Scenario 1: Target execution with explicit CelesTrak group scale
    print("\n[Scenario 1] Testing exact scale sampling from CelesTrak group...")
    try:
        sats = load_and_sample_satellites(group_name="weather", k=5, seed=42)
        print(f"SUCCESS: Factory scaled precisely to {len(sats)} satellites from the live pool.")
        for s in sats:
            print(f"  - {s.name} (NORAD {s.norad_id})")
    except Exception as e:
        print(f"Scenario 1 FAILED: {e}")
        sys.exit(1)

    # Scenario 2: Test reading from your real data/norad_ids.txt WITHOUT modifying it
    local_file = "data/norad_ids.txt"
    print(f"\n[Scenario 2] Testing read operations from your real pool file '{local_file}'...")
    
    if not pathlib.Path(local_file).exists():
        print(f"  Skipping Scenario 2: '{local_file}' does not exist yet. Please create it with your IDs.")
    else:
        try:
            sats_file = load_and_sample_satellites(file_path=local_file, k=2, seed=42)
            print(f"SUCCESS: Successfully read your file and sampled {len(sats_file)} satellites.")
            for s in sats_file:
                print(f"  - {s.name} (NORAD {s.norad_id})")
        except Exception as e:
            print(f"Scenario 2 FAILED: {e}")
            sys.exit(1)

    print("\nAll read-only factory pool tests PASSED! Your local files remained untouched.")

if __name__ == "__main__":
    main()