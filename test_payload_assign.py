import sys
from src.core.datatypes import SatelliteConfig
from src.modules.data_collector.payload_assign import assign_satellite_payloads

def main():
    print("Testing Strict Payload Assignment with Probability Distributions...")

    # Mock clean satellites list (10 items to see the distribution effect clearly)
    mock_satellites = [SatelliteConfig(norad_id=i, name=f"SAT_{i}", tle_line1=".", tle_line2=".") for i in range(10)]

    sensors_pool = ["VIS", "PAN", "NIR", "SWIR", "TIR", "GNSS", "SAR"]
    bands_pool = ["S", "X", "Ka"]

    # Custom distribution: 
    # Optical = 70%, SAR = 20%, Thermal = 10%
    custom_sensor_weights = [0.7, 0.2, 0.1]
    
    # X band = 80%, S band = 10%, Ka band = 10%
    custom_band_weights = [0.1, 0.8, 0.1]

    print("\n[Test 1] Executing custom probability distribution assignment...")
    try:
        configured_sats = assign_satellite_payloads(
            satellites=mock_satellites,
            available_sensors=sensors_pool,
            available_bands=bands_pool,
            sensor_weights=custom_sensor_weights,
            band_weights=custom_band_weights,
            min_sensors_per_sat=1,
            max_sensors_per_sat=2,
            seed=42
        )
        
        for sat in configured_sats:
            print(f"  - Satellite: {sat.name:<8} | Band: {sat.band:<3} | Sensors: {sat.sensors}")
            
    except Exception as e:
        print(f"Test 1 FAILED: {e}")
        sys.exit(1)

    print("\n[Test 2] Validating size mismatch enforcement (Error Control)...")
    try:
        print("  Attempting to send an invalid weights array size...")
        invalid_weights = [0.5, 0.5] # Missing one element for the 3 sensors available
        assign_satellite_payloads(mock_satellites, sensors_pool, bands_pool, sensor_weights=invalid_weights)
        print("Test 2 FAILED: Module allowed size anomalies without throwing an exception.")
        sys.exit(1)
    except ValueError as e:
        print(f"SUCCESS: System correctly rejected distribution anomalies: '{e}'")

    print("\nAll Payload Distribution tests PASSED!")

if __name__ == "__main__":
    main()