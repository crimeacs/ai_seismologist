# SCEDC Data Access Examples

This repository contains examples for accessing seismic data from the Southern California Earthquake Data Center (SCEDC) using Python and ObsPy.

## Overview

The Southern California Earthquake Data Center (SCEDC) archives seismic waveform data primarily from the Southern California Seismic Network (SCSN). The data includes:

- **Digital Seismograms**: MiniSEED format waveforms (1977-present for event data, 1999-present for continuous data)
- **Event-based data**: Triggered waveforms for specific earthquakes
- **Continuous data**: Ongoing seismic monitoring data
- **Station metadata**: Instrument response information

## Data Access Methods

### 1. Web STP Service
- Web service version of STP (Seismogram Transfer Program)
- Allows retrieval of continuous or triggered waveform data
- Supports various formats including miniSEED
- Accessible via web forms or programmatic calls

### 2. FDSN Waveform Web Service
- Follows FDSN Web Service specifications (fdsnws-dataselect)
- For continuous waveform data in miniSEED format
- Programmatic access via HTTP requests

### 3. Event Catalog Service
- Earthquake catalog data with event parameters
- Magnitude, location, time information
- Available in JSON, XML, or text formats

### 4. Station Metadata Service
- Instrument response information
- Station location and configuration data
- Required for converting digital counts to physical units

## Files in this Repository

### `app.py`
Basic example showing how to download and plot seismic waveform data from SCEDC.

### `scedc_data_access.py`
Comprehensive class-based approach for accessing SCEDC data services including:
- Waveform data retrieval
- Event catalog queries
- Station metadata access

## Usage Examples

### Basic Waveform Data Access

```python
from scedc_data_access import SCEDCDataAccess

# Initialize the data access object
scedc = SCEDCDataAccess()

# Get waveform data for a specific time period
st = scedc.get_waveform_data(
    starttime="2024-01-01T00:00:00",
    endtime="2024-01-01T00:10:00",
    network="CI",      # California Institute of Technology
    station="PAS",     # Pasadena station
    channel="BHZ"      # Broadband High-gain Z component
)

if st:
    st.plot()
```

### Event Catalog Access

```python
# Get earthquake events with magnitude >= 3.0
events = scedc.get_event_data(
    starttime="2024-01-01",
    endtime="2024-01-31",
    minmagnitude=3.0,
    format="json"
)

if events and 'events' in events:
    for event in events['events']:
        mag = event.get('magnitudes', [{}])[0].get('mag', 'N/A')
        time = event.get('origins', [{}])[0].get('time', 'N/A')
        print(f"Magnitude: {mag}, Time: {time}")
```

### Station Metadata Access

```python
# Get station metadata for instrument response
metadata = scedc.get_station_metadata(
    network="CI",
    station="PAS",
    format="xml"
)
```

## URL Construction Examples

### FDSN Waveform Service
```
https://service.scedc.caltech.edu/fdsnws/dataselect/1/query?
starttime=2024-01-01T00:00:00&
endtime=2024-01-01T00:10:00&
net=CI&
sta=PAS&
cha=BHZ&
format=mseed
```

### Event Catalog Service
```
https://service.scedc.caltech.edu/fdsnws/event/1/query?
starttime=2024-01-01&
endtime=2024-01-31&
minmagnitude=3.0&
format=json
```

### Station Metadata Service
```
https://service.scedc.caltech.edu/fdsnws/station/1/query?
net=CI&
sta=PAS&
format=xml
```

## Common Network and Station Codes

### Networks
- **CI**: California Institute of Technology (SCSN)
- **SC**: Southern California Seismic Network (historical)

### Stations
- **PAS**: Pasadena
- **SVD**: Soledad Canyon
- **LAD**: La Habra
- **RPV**: Rancho Palos Verdes

### Channels
- **BHZ**: Broadband High-gain Z component
- **BHN**: Broadband High-gain N component
- **BHE**: Broadband High-gain E component
- **HHZ**: High-frequency High-gain Z component

## Data Availability

- **Event-based data**: 1977-present
- **Continuous data**: 
  - Low sample rate (≤40 sps): 1999-present
  - High sample rate (≥80 sps): 2008-present
  - All continuous data: 2010-present

## Requirements

```bash
pip install obspy requests
```

## Additional Resources

- [SCEDC Website](https://scedc.caltech.edu/)
- [Waveform Archival Policies](https://scedc.caltech.edu/data/waveform-archival-policies.html)
- [Station Metadata and Maps](https://scedc.caltech.edu/data/stations/)
- [AWS Public Dataset](https://registry.opendata.aws/scedc/)

## Citation

When using SCEDC data, please cite:
```
Hutton, K., Woessner, J. and Hauksson, E. (2010). Earthquake Monitoring in Southern California for Seventy-Seven Years (1932-2008), Bull. Seism. Soc. Am., 100, 423-446.
```

## License

This code is provided as-is for educational and research purposes. Please refer to SCEDC's data usage policies for restrictions on data redistribution. 