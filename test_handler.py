import sys
from src.modules.data_collector.celestrak_handler import fetch_celestrak_metadata

def main():
    print("Testing CelesTrak handler...")
    
    # ISS (ZARYA) NORAD ID
    test_id = 25544
    
    # Query the API
    result = fetch_celestrak_metadata(test_id)
    
    # Verify the output
    if result is None:
        print(f"Test FAILED: Could not retrieve data for NORAD {test_id}")
        sys.exit(1)
        
    print("\nTest PASSED! Data retrieved successfully:")
    print("-" * 40)
    print(f"Satellite Name : {result['name']}")
    print(f"NORAD ID       : {result['norad_id']}")
    print(f"TLE Line 1     : {result['tle_line1']}")
    print(f"TLE Line 2     : {result['tle_line2']}")
    print("-" * 40)

if __name__ == "__main__":
    main()