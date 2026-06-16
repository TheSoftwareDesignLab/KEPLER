import sys
from src.core.datatypes import GroundStationConfig
from src.modules.data_collector.gs_pool import load_and_sample_stations


def main():
    print("Testing Ground Station Pool...")

    csv_path = "data/ground_station.csv"

    # Scenario 1: Test loading and sampling from the CSV file with global bands
    print("\n[Scenario 1] Loading from CSV with allowed bands filter...")
    try:
        sampled_from_file = load_and_sample_stations(
            file_path=csv_path, 
            k=3, 
            allowed_bands=["S", "X"], 
            seed=42
        )
        print(f"Successfully sampled {len(sampled_from_file)} stations from CSV.")
        for gs in sampled_from_file:
            print(f"  - ID: {gs.id:<30} | Name: {gs.name:<35} | Bands: {gs.bands_supported}")
            for b in gs.bands_supported:
                assert b in ["S", "X"], f"Unexpected band {b} leaked past filtering system."
    except Exception as e:
        print(f"Scenario 1 FAILED: {e}")
        sys.exit(1)

    # Scenario 2: Test strict subset filtering down to a single specific band
    print("\n[Scenario 2] Testing strict single-band exclusion filtering...")
    try:
        single_band_pool = load_and_sample_stations(
            file_path=csv_path, 
            k=3, 
            allowed_bands=["Ka"], 
            seed=99
        )
        print(f"Successfully sampled {len(single_band_pool)} stations supporting 'Ka' band.")
        for gs in single_band_pool:
            print(f"  - ID: {gs.id:<30} | Name: {gs.name:<35} | Bands: {gs.bands_supported}")
            assert "Ka" in gs.bands_supported, "Station without matching subset leaked into pool."
    except Exception as e:
        print(f"Scenario 2 FAILED: {e}")
        sys.exit(1)

    # Scenario 3: Test passing a strict custom list of stations using descriptive geometry fields
    print("\n[Scenario 3] Bypassing file with a custom list...")
    custom_list = [
        GroundStationConfig(id="custom_bog", name="Bogota Hub", latitude=4.711, longitude=-74.072, elevation=2550.0, bands_supported=["S", "X"]),
        GroundStationConfig(id="custom_sva", name="Svalbard Core", latitude=78.230, longitude=15.400, elevation=400.0, bands_supported=["S", "X", "Ka"]),
        GroundStationConfig(id="custom_nno", name="New Norcia ESA", latitude=-31.048, longitude=116.191, elevation=252.0, bands_supported=["X"])
    ]

    try:
        sampled_from_custom = load_and_sample_stations(custom_stations=custom_list, k=2, seed=123)
        print(f"Successfully sampled {len(sampled_from_custom)} stations from custom list.")
        for gs in sampled_from_custom:
            print(f"  - ID: {gs.id:<30} | Name: {gs.name:<35} | Bands: {gs.bands_supported}")
    except Exception as e:
        print(f"Scenario 3 FAILED: {e}")
        sys.exit(1)

    print("\nAll Ground Station Pool tests PASSED!")


if __name__ == "__main__":
    main()