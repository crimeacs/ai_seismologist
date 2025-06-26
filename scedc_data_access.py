"""
SCEDC Data Access Examples
Southern California Earthquake Data Center

This script demonstrates how to access various types of seismic data from SCEDC:
1. Waveform data via Web STP and FDSN services
2. Event catalog data
3. Station metadata
"""

import io
import requests
import logging
from datetime import datetime, timedelta
from obspy import read, UTCDateTime
import json

# Set up logging
logger = logging.getLogger(__name__)

class SCEDCDataAccess:
    """Class for accessing SCEDC data services"""
    
    def __init__(self):
        self.base_url = "https://service.scedc.caltech.edu"
        self.fdsn_url = f"{self.base_url}/fdsnws/dataselect/1/query"
        self.event_url = f"{self.base_url}/fdsnws/event/1/query"
        self.station_url = f"{self.base_url}/fdsnws/station/1/query"
        logger.info("🔧 SCEDCDataAccess initialized")
    
    def get_waveform_data(self, starttime, endtime, network="CI", station="PAS", 
                         channel="BHZ", format="mseed"):
        """
        Get waveform data using FDSN Web Service
        
        Parameters:
        - starttime: Start time (UTCDateTime or string)
        - endtime: End time (UTCDateTime or string)
        - network: Network code (default: CI for Caltech)
        - station: Station code (default: PAS for Pasadena)
        - channel: Channel code (default: BHZ for broadband Z)
        - format: Data format (default: mseed)
        """
        logger.info(f"📊 Requesting waveform data: {network}.{station}.{channel}")
        logger.info(f"   Time range: {starttime} to {endtime}")
        
        # Convert times to string if needed
        if isinstance(starttime, UTCDateTime):
            starttime = starttime.isoformat()
        if isinstance(endtime, UTCDateTime):
            endtime = endtime.isoformat()
        
        params = {
            'starttime': starttime,
            'endtime': endtime,
            'net': network,
            'sta': station,
            'cha': channel,
            'format': format
        }
        
        try:
            logger.info(f"   URL: {self.fdsn_url}")
            logger.info(f"   Parameters: {params}")
            
            r = requests.get(self.fdsn_url, params=params, timeout=30)
            logger.info(f"   Response status: {r.status_code}")
            logger.info(f"   Response size: {len(r.content)} bytes")
            r.raise_for_status()
            
            if r.content and len(r.content) > 0:
                logger.info("   ✓ Data received, parsing with ObsPy...")
                st = read(io.BytesIO(r.content))
                logger.info(f"   ✓ Successfully parsed {len(st)} traces")
                for tr in st:
                    logger.info(f"     - {tr.stats.network}.{tr.stats.station}.{tr.stats.channel}")
                    logger.info(f"       Start: {tr.stats.starttime}")
                    logger.info(f"       End: {tr.stats.endtime}")
                    logger.info(f"       Sample Rate: {tr.stats.sampling_rate} Hz")
                    logger.info(f"       Samples: {tr.stats.npts}")
                return st
            else:
                logger.warning("   ⚠ No data returned (empty response)")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"   ✗ Error downloading data: {e}")
            return None
    
    def get_event_data(self, starttime, endtime, minmagnitude=None, 
                      maxmagnitude=None, format="text"):
        """
        Get earthquake catalog data
        
        Parameters:
        - starttime: Start time (YYYY-MM-DD format for event service)
        - endtime: End time (YYYY-MM-DD format for event service)
        - minmagnitude: Minimum magnitude
        - maxmagnitude: Maximum magnitude
        - format: Response format (text, xml) - JSON not supported by SCEDC
        """
        logger.info(f"📋 Requesting event data")
        logger.info(f"   Time range: {starttime} to {endtime}")
        logger.info(f"   Magnitude range: {minmagnitude} to {maxmagnitude}")
        
        # Convert times to proper format for event service
        if isinstance(starttime, str) and 'T' in starttime:
            starttime = starttime.split('T')[0]  # Extract date part only
        if isinstance(endtime, str) and 'T' in endtime:
            endtime = endtime.split('T')[0]  # Extract date part only
        
        params = {
            'starttime': starttime,
            'endtime': endtime,
            'format': format
        }
        
        if minmagnitude:
            params['minmagnitude'] = minmagnitude
        if maxmagnitude:
            params['maxmagnitude'] = maxmagnitude
        
        try:
            logger.info(f"   URL: {self.event_url}")
            logger.info(f"   Parameters: {params}")
            
            r = requests.get(self.event_url, params=params, timeout=30)
            logger.info(f"   Response status: {r.status_code}")
            logger.info(f"   Response size: {len(r.text)} characters")
            r.raise_for_status()
            
            if r.text:
                lines = r.text.strip().split('\n')
                logger.info(f"   ✓ Received {len(lines)} lines of event data")
                if len(lines) > 1:
                    logger.info(f"   ✓ Found {len(lines) - 1} events (excluding header)")
                return r.text
            else:
                logger.warning("   ⚠ No event data returned")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"   ✗ Error downloading event data: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"   Response content: {e.response.text}")
            return None
    
    def get_station_metadata(self, network="CI", station="PAS", format="xml"):
        """
        Get station metadata
        
        Parameters:
        - network: Network code
        - station: Station code
        - format: Response format (xml, text)
        """
        logger.info(f"📍 Requesting station metadata: {network}.{station}")
        
        params = {
            'net': network,
            'sta': station,
            'format': format
        }
        
        try:
            logger.info(f"   URL: {self.station_url}")
            logger.info(f"   Parameters: {params}")
            
            r = requests.get(self.station_url, params=params, timeout=30)
            logger.info(f"   Response status: {r.status_code}")
            logger.info(f"   Response size: {len(r.text)} characters")
            r.raise_for_status()
            
            if r.text:
                logger.info("   ✓ Station metadata received successfully")
                return r.text
            else:
                logger.warning("   ⚠ No station metadata returned")
                return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"   ✗ Error downloading station metadata: {e}")
            return None

def example_usage():
    """Example usage of SCEDC data access"""
    logger.info("🚀 Starting SCEDC data access examples...")
    
    # Initialize the data access object
    scedc = SCEDCDataAccess()
    
    # Example 1: Get waveform data for a more recent time period
    logger.info("\n📊 Example 1: Waveform Data")
    logger.info("-" * 30)
    # Try multiple stations and recent dates
    stations_to_try = ["PAS", "SVD", "LAD", "RPV"]
    start_time = "2024-12-01T00:00:00"
    end_time = "2024-12-01T00:10:00"
    
    st = None
    for station_code in stations_to_try:
        logger.info(f"Trying station: {station_code}")
        st = scedc.get_waveform_data(start_time, end_time, 
                                    network="CI", station=station_code, channel="BHZ")
        if st and len(st) > 0:
            logger.info(f"Found data for station {station_code}")
            break
    
    if st and len(st) > 0:
        logger.info("✅ Waveform data example completed")
    else:
        logger.warning("⚠ No waveform data found for any station")
    
    # Example 2: Get earthquake catalog data with correct date format
    logger.info("\n📋 Example 2: Event Catalog")
    logger.info("-" * 30)
    # Use recent dates and correct format (YYYY-MM-DD)
    events = scedc.get_event_data("2024-12-01", "2024-12-31", 
                                 minmagnitude=3.0, format="text")
    
    if events:
        logger.info("✅ Event catalog example completed")
    else:
        logger.warning("⚠ Failed to retrieve event data")
    
    # Example 3: Get station metadata
    logger.info("\n📍 Example 3: Station Metadata")
    logger.info("-" * 30)
    metadata = scedc.get_station_metadata("CI", "PAS")
    if metadata:
        logger.info("✅ Station metadata example completed")
    else:
        logger.warning("⚠ Failed to retrieve station metadata")

if __name__ == "__main__":
    # Set up logging for the main script
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    example_usage() 