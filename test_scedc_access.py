#!/usr/bin/env python3
"""
Test script for SCEDC data access
This script tests various approaches to access SCEDC data to ensure we can retrieve real data.
"""

import requests
from datetime import datetime, timedelta
from obspy import read, UTCDateTime
import io

def test_scedc_access():
    """Test various SCEDC data access methods"""
    
    base_url = "https://service.scedc.caltech.edu"
    
    print("=== Testing SCEDC Data Access ===\n")
    
    # Test 1: Check if the service is accessible
    print("1. Testing service availability...")
    try:
        r = requests.get(f"{base_url}/fdsnws/dataselect/1/query", timeout=10)
        print(f"   Service status: {r.status_code}")
        if r.status_code == 200:
            print("   ✓ Service is accessible")
        else:
            print("   ✗ Service returned error")
    except Exception as e:
        print(f"   ✗ Service error: {e}")
        return
    
    # Test 2: Try to get recent event data
    print("\n2. Testing event catalog access...")
    try:
        # Use a recent date range
        event_params = {
            'starttime': '2024-12-01',
            'endtime': '2024-12-31',
            'minmagnitude': 2.0,  # Lower magnitude threshold
            'format': 'json'
        }
        
        r = requests.get(f"{base_url}/fdsnws/event/1/query", params=event_params, timeout=30)
        print(f"   Event service status: {r.status_code}")
        
        if r.status_code == 200:
            data = r.json()
            if 'events' in data and len(data['events']) > 0:
                print(f"   ✓ Found {len(data['events'])} events")
                # Show first event
                event = data['events'][0]
                mag = event.get('magnitudes', [{}])[0].get('mag', 'N/A')
                time = event.get('origins', [{}])[0].get('time', 'N/A')
                print(f"   First event: Magnitude {mag}, Time {time}")
            else:
                print("   ⚠ No events found in this time range")
        else:
            print(f"   ✗ Event service error: {r.text}")
    except Exception as e:
        print(f"   ✗ Event service error: {e}")
    
    # Test 3: Try to get waveform data with different approaches
    print("\n3. Testing waveform data access...")
    
    # Try different time periods and stations
    test_cases = [
        # Recent data
        ("2024-12-01T00:00:00", "2024-12-01T00:10:00", "CI", "PAS", "BHZ"),
        ("2024-12-01T00:00:00", "2024-12-01T00:10:00", "CI", "SVD", "BHZ"),
        # Historical data (more likely to exist)
        ("2023-01-01T00:00:00", "2023-01-01T00:10:00", "CI", "PAS", "BHZ"),
        ("2022-01-01T00:00:00", "2022-01-01T00:10:00", "CI", "PAS", "BHZ"),
        # Different channels
        ("2023-01-01T00:00:00", "2023-01-01T00:10:00", "CI", "PAS", "BHN"),
        ("2023-01-01T00:00:00", "2023-01-01T00:10:00", "CI", "PAS", "BHE"),
    ]
    
    waveform_found = False
    for i, (start, end, net, sta, cha) in enumerate(test_cases, 1):
        print(f"   Test {i}: {net}.{sta}.{cha} ({start} to {end})")
        
        try:
            params = {
                'starttime': start,
                'endtime': end,
                'net': net,
                'sta': sta,
                'cha': cha,
                'format': 'mseed'
            }
            
            r = requests.get(f"{base_url}/fdsnws/dataselect/1/query", params=params, timeout=30)
            
            if r.status_code == 200 and len(r.content) > 0:
                # Try to read the data
                try:
                    st = read(io.BytesIO(r.content))
                    if len(st) > 0:
                        print(f"   ✓ Success! Found {len(st)} traces")
                        print(f"   Trace info: {st[0]}")
                        waveform_found = True
                        break
                    else:
                        print("   ⚠ No traces in response")
                except Exception as e:
                    print(f"   ⚠ Could not parse data: {e}")
            else:
                print(f"   ⚠ No data (status: {r.status_code})")
                
        except Exception as e:
            print(f"   ✗ Error: {e}")
    
    if not waveform_found:
        print("   ⚠ No waveform data found in any test case")
    
    # Test 4: Check station metadata
    print("\n4. Testing station metadata access...")
    try:
        station_params = {
            'net': 'CI',
            'sta': 'PAS',
            'format': 'xml'
        }
        
        r = requests.get(f"{base_url}/fdsnws/station/1/query", params=station_params, timeout=30)
        print(f"   Station service status: {r.status_code}")
        
        if r.status_code == 200 and len(r.text) > 0:
            print("   ✓ Station metadata retrieved successfully")
            print(f"   Metadata length: {len(r.text)} characters")
        else:
            print(f"   ✗ Station service error: {r.text}")
    except Exception as e:
        print(f"   ✗ Station service error: {e}")
    
    print("\n=== Test Summary ===")
    print("If you see ✓ marks above, the SCEDC services are working correctly.")
    print("If you see ⚠ marks, the services work but no data was found for the test parameters.")
    print("If you see ✗ marks, there were errors accessing the services.")

if __name__ == "__main__":
    test_scedc_access() 