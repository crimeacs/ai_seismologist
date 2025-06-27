#!/usr/bin/env python3
"""
SCEDC Data Access Gradio App
Interactive web interface for accessing Southern California Earthquake Data Center data
"""

import gradio as gr
import requests
import io
import logging
from datetime import datetime, timedelta
from obspy import read, UTCDateTime
from obspy.taup import TauPyModel
from obspy.geodetics import locations2degrees
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import xml.etree.ElementTree as ET

# Set up logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PhaseNet model (via SeisBench)
# ---------------------------------------------------------------------------
# We load a pretrained PhaseNet picker via the SeisBench model zoo.  On first
# use, the weight file (~30 MB) is downloaded and cached by SeisBench.  If the
# package is not available the app will fall back to TauP-only estimates.

try:
<<<<<<< HEAD
    from seisbench.models import PhaseNet as SBPhaseNet  # type: ignore
    HAS_SEISBENCH = True
except Exception:
    HAS_SEISBENCH = False

# Log PhaseNet availability at import time
if HAS_PHASENET:
    logger.info("🤖 PhaseNet available – ML phase picking enabled.")
else:
    logger.warning("⚠️  PhaseNet NOT available – falling back to TauP estimates.")
=======
    import seisbench.models as sbm  # type: ignore

    # "geofon" weights are small (~30 MB) and tuned for global broadband data.
    # Feel free to switch to a different pretrained set (see list_pretrained()).
    PN_MODEL = sbm.PhaseNet.from_pretrained("geofon")
    HAS_PHASENET = True
    logger.info("🤖 PhaseNet (SeisBench) loaded – ML phase picking enabled.")
except Exception as e:  # pragma: no cover – missing optional dep
    HAS_PHASENET = False
    PN_MODEL = None  # type: ignore
    logger.warning(f"⚠️  PhaseNet via SeisBench NOT available – reason: {e}")
>>>>>>> c795fa4 (Integrate SeisBench PhaseNet Model for Enhanced Phase Picking)

if HAS_SEISBENCH:
    logger.info("🤖 SeisBench available – advanced models enabled.")
else:
    logger.warning("⚠️  SeisBench NOT available – skipping waveform classification.")

class SCEDCInterface:
    """Interface for SCEDC data services"""
    
    def __init__(self):
        self.base_url = "https://service.scedc.caltech.edu"
        self.fdsn_url = f"{self.base_url}/fdsnws/dataselect/1/query"
        self.event_url = f"{self.base_url}/fdsnws/event/1/query"
        self.station_url = f"{self.base_url}/fdsnws/station/1/query"
        logger.info("🔧 SCEDCInterface initialized")
    
    def get_waveform_data(self, starttime, endtime, network, station, channel):
        """Get waveform data"""
        logger.info(f"📊 Requesting waveform data: {network}.{station}.{channel}")
        logger.info(f"   Time range: {starttime} to {endtime}")
        
        try:
            params = {
                'starttime': starttime,
                'endtime': endtime,
                'net': network,
                'sta': station,
                'cha': channel,
                'format': 'mseed'
            }
            
            logger.info(f"   URL: {self.fdsn_url}")
            logger.info(f"   Parameters: {params}")
            
            r = requests.get(self.fdsn_url, params=params, timeout=30)
            logger.info(f"   Response status: {r.status_code}")
            logger.info(f"   Response size: {len(r.content)} bytes")
            
            # Log the actual URL that was requested
            logger.info(f"   Full URL: {r.url}")
            
            if r.status_code == 204:
                logger.warning(f"   ⚠ 204 No Content - No data available for {network}.{station}.{channel}")
                logger.warning(f"   This usually means the station was not recording or no data exists for this time window")
                return None
            elif r.status_code == 400:
                logger.error(f"   ✗ 400 Bad Request - Invalid parameters")
                logger.error(f"   Response text: {r.text}")
                return None
            elif r.status_code == 404:
                logger.error(f"   ✗ 404 Not Found - Station or channel not found")
                return None
            
            r.raise_for_status()
            
            if r.content and len(r.content) > 0:
                logger.info("   ✓ Data received, parsing with ObsPy...")
                st = read(io.BytesIO(r.content))
                logger.info(f"   ✓ Successfully parsed {len(st)} trace(s)")
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
                
        except Exception as e:
            logger.error(f"   ✗ Error downloading waveform data: {e}")
            return None
    
    def get_event_data(self, starttime, endtime, minmagnitude, maxmagnitude):
        """Get earthquake catalog data"""
        logger.info(f"📋 Requesting event data")
        logger.info(f"   Time range: {starttime} to {endtime}")
        logger.info(f"   Magnitude range: {minmagnitude} to {maxmagnitude}")
        
        try:
            # Convert dates to proper format
            if 'T' in starttime:
                starttime = starttime.split('T')[0]
            if 'T' in endtime:
                endtime = endtime.split('T')[0]
            
            params = {
                'starttime': starttime,
                'endtime': endtime,
                'format': 'text'
            }
            
            if minmagnitude:
                params['minmagnitude'] = minmagnitude
            if maxmagnitude:
                params['maxmagnitude'] = maxmagnitude
            
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
                
        except Exception as e:
            logger.error(f"   ✗ Error downloading event data: {e}")
            return None
    
    def get_station_metadata(self, network, station):
        """Get station metadata"""
        logger.info(f"📍 Requesting station metadata: {network}.{station}")
        
        try:
            params = {
                'net': network,
                'sta': station,
                'format': 'xml'
            }
            
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
            
        except Exception as e:
            logger.error(f"   ✗ Error downloading station metadata: {e}")
            return None

def plot_waveform(st, phase_arrivals=None, event_time=None, station=None):
    """Create a matplotlib plot of the waveform data, overlaying phase arrivals if provided"""
    logger.info("🎨 Creating waveform plot...")
    
    if st is None or len(st) == 0:
        logger.warning("   ⚠ No data to plot")
        return None
    
    try:
        # Apply seismic filtering to enhance signal quality
        logger.info("   🔧 Applying seismic filters...")
        st_filtered = st.copy()
        
        for i, tr in enumerate(st_filtered):
            logger.info(f"   📊 Filtering trace {i+1}: {tr.stats.network}.{tr.stats.station}.{tr.stats.channel}")
            
            # Remove mean and linear trend
            tr.detrend('linear')
            tr.detrend('constant')
            logger.info(f"     ✓ Removed trend and mean")
            
            # Apply bandpass filter (0.1-10 Hz) to enhance P and S waves
            # This removes both low-frequency noise and high-frequency noise
            tr.filter('bandpass', freqmin=0.1, freqmax=10.0, corners=4, zerophase=True)
            logger.info(f"     ✓ Applied bandpass filter (0.1-10 Hz)")
            
            # Apply high-pass filter (0.5 Hz) to remove long-period noise
            tr.filter('highpass', freq=0.5, corners=4, zerophase=True)
            logger.info(f"     ✓ Applied high-pass filter (0.5 Hz)")
            
            # Optional: Apply a notch filter to remove power line interference (60 Hz in US)
            # tr.filter('bandstop', freqmin=58, freqmax=62, corners=4, zerophase=True)
            # logger.info(f"     ✓ Applied notch filter (58-62 Hz)")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot original data
        for i, tr in enumerate(st):
            times = np.arange(len(tr.data)) / tr.stats.sampling_rate
            ax1.plot(times, tr.data, label=f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel} (Raw)", 
                    linewidth=0.8, alpha=0.7, color='gray')
        
        ax1.set_ylabel('Amplitude (counts)', fontsize=12)
        ax1.set_title('Raw Seismic Data', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot filtered data
        for i, tr in enumerate(st_filtered):
            times = np.arange(len(tr.data)) / tr.stats.sampling_rate
            ax2.plot(times, tr.data, label=f"{tr.stats.network}.{tr.stats.station}.{tr.stats.channel} (Filtered)", 
                    linewidth=0.8, color='blue')
            logger.info(f"   ✓ Plotted trace {i+1}: {tr.stats.network}.{tr.stats.station}.{tr.stats.channel}")
        
        # Add estimated phase arrivals if not provided
        if not phase_arrivals and event_time and station:
            logger.info("   📍 Adding estimated phase arrivals based on typical travel times")
            
            # Estimate distance from event to station (rough approximation)
            # For Southern California, typical distances are 50-200 km
            estimated_distance_km = 100  # Default estimate
            
            # Typical P-wave velocity: ~6 km/s, S-wave velocity: ~3.5 km/s
            p_velocity = 6.0  # km/s
            s_velocity = 3.5  # km/s
            
            # Calculate travel times
            p_travel_time = estimated_distance_km / p_velocity  # seconds
            s_travel_time = estimated_distance_km / s_velocity  # seconds
            
            # Add some uncertainty and typical arrival patterns
            phase_arrivals = [
                {'phase': 'P', 'time': p_travel_time - 5, 'abs_time': f'~{p_travel_time-5:.1f}s'},
                {'phase': 'S', 'time': s_travel_time - 5, 'abs_time': f'~{s_travel_time-5:.1f}s'},
            ]
            
            logger.info(f"   📍 Estimated P arrival at ~{p_travel_time-5:.1f}s")
            logger.info(f"   📍 Estimated S arrival at ~{s_travel_time-5:.1f}s")
        
        # Overlay phase arrivals on filtered plot
        if phase_arrivals:
            for phase in phase_arrivals:
                t = phase['time']
                label = phase['phase']
                color = 'red' if label.upper() == 'P' else ('blue' if label.upper() == 'S' else 'green')
                ax2.axvline(t, color=color, linestyle='--', linewidth=2, alpha=0.8, label=f"{label} arrival")
                
                # Add text label
                y_pos = ax2.get_ylim()[1] * 0.9
                ax2.text(t, y_pos, f"{label}", color=color, fontsize=14, fontweight='bold', 
                       rotation=90, ha='right', va='top', bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor='white', alpha=0.8, edgecolor=color))
        
        # Add event origin time marker on both plots
        if event_time:
            for ax in [ax1, ax2]:
                ax.axvline(0, color='black', linestyle='-', linewidth=3, alpha=0.7, label='Event Origin')
                ax.text(0, ax.get_ylim()[1] * 0.95, 'EVENT\nORIGIN', color='black', fontsize=12, 
                       fontweight='bold', ha='center', va='top', bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor='yellow', alpha=0.8))
        
        # Add informative labels and title
        ax2.set_xlabel('Time (seconds from event origin)', fontsize=12)
        ax2.set_ylabel('Amplitude (counts)', fontsize=12)
        
        # Create informative title
        title = f'Seismic Waveform: {st[0].stats.network}.{st[0].stats.station}.{st[0].stats.channel}'
        if event_time:
            title += f'\nEvent: {event_time}'
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Add legend to filtered plot
        ax2.legend(loc='upper right', fontsize=10)
        
        # Add grid
        ax2.grid(True, alpha=0.3)
        
        # Add text box with waveform information
        info_text = f"Sample Rate: {st[0].stats.sampling_rate} Hz\n"
        info_text += f"Duration: {st[0].stats.endtime - st[0].stats.starttime:.1f} seconds\n"
        info_text += f"Samples: {st[0].stats.npts:,}\n"
        info_text += f"Filters Applied:\n"
        info_text += f"  • Bandpass: 0.1-10 Hz\n"
        info_text += f"  • High-pass: 0.5 Hz\n"
        if phase_arrivals:
            info_text += "\nPhase Arrivals (estimated):\n"
            for arr in phase_arrivals:
                info_text += f"  {arr['phase']}: {arr['abs_time']}\n"
        
        # Position text box in upper left of filtered plot
        ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
               facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        logger.info("   ✓ Waveform plot created successfully with filtering and phase arrivals")
        return fig
        
    except Exception as e:
        logger.error(f"   ✗ Error creating plot: {e}")
        return None

def get_event_station_phase_info(start, end, min_m, max_m):
    """Search events, return event list and for each event, stations with phase picks"""
    logger.info("🔍 Searching events and fetching station/phase info...")
    scedc = SCEDCInterface()
    events_text = scedc.get_event_data(start, end, min_m, max_m)
    if not events_text:
        return [], {}, "No events found."
    lines = events_text.strip().split('\n')
    if len(lines) < 2:
        return [], {}, "No events found."
    header = [col.strip().replace('#','') for col in lines[0].split('|')]
    logger.info(f"Event header columns: {header}")
    data_lines = lines[1:]
    event_list = []
    event_station_map = {}
    event_time_map = {}
    for i, line in enumerate(data_lines):
        values = [v.strip() for v in line.split('|')]
        if len(values) < len(header):
            continue
        event = dict(zip(header, values))
        if i == 0:
            logger.info(f"Sample event dict: {event}")
        event_id = event.get('Time', f'Event {i+1}')
        event_list.append({
            'Time': event.get('Time', 'N/A'),
            'Magnitude': float(event.get('Magnitude', 0)),
            'Location': event.get('EventLocationName', 'N/A'),
            'Raw': event
        })
        event_time_map[event_id] = event.get('Time', '')
        # Fetch stations with phase picks for this event
        stations = get_stations_with_phases(event.get('Time', ''))
        event_station_map[event_id] = stations
    return event_list, event_station_map, ""

def get_stations_with_phases(event_time):
    """Get a list of common SCEDC stations that typically have data"""
    logger.info(f"🔎 Getting common SCEDC stations for event_time={event_time}")
    
    # Common SCEDC stations that typically have good data coverage
    # These are major stations in the Southern California network
    common_stations = [
        "PAS", "PFO", "SVD", "GSC", "LAC", "RPV", "WMC", "BKS", "BRK", "CMB",
        "CWC", "DGR", "DLA", "DSS", "FAR", "FMP", "GLA", "GMR", "HEC",
        "HOL", "JRC", "KNW", "LBC", "LBN", "LBS", "LGB", "LON", "LSA",
        "MAL", "MHC", "MNT", "MOR", "MUR", "NEE", "NSS", "OAR", "OBS",
        "PKD", "PLM", "POC", "POM", "SAL", "SAS", "SBC", "SBR",
        "TIN", "TOA", "WLA", "WMF", "WON", "YOR"
    ]
    
    # Remove duplicates and sort
    unique_stations = sorted(list(set(common_stations)))
    
    logger.info(f"   ✓ Returning {len(unique_stations)} unique SCEDC stations")
    return unique_stations

def create_event_map(events):
    # events: list of dicts with keys 'Time', 'Magnitude', 'Location', 'Raw'
    if not events:
        return go.Figure()
    lats, lons, mags, times, hover = [], [], [], [], []
    for ev in events:
        raw = ev['Raw']
        try:
            lat = float(raw.get('Latitude', 0))
            lon = float(raw.get('Longtitude', 0))
        except Exception as e:
            logging.warning(f"Bad lat/lon: {e} for event {raw}")
            lat, lon = 0, 0
        mag = float(raw.get('Magnitude', 0))
        time = raw.get('Time', '')
        loc = raw.get('EventLocationName', '')
        lats.append(lat)
        lons.append(lon)
        mags.append(mag)
        times.append(time)
        hover.append(f"<b>{time}</b><br>Mag: {mag}<br>{loc}")
    logging.info(f"Event map lat/lon: {list(zip(lats, lons))}")
    # Filter out events with lat/lon == 0
    valid = [(lat, lon, mag, t, h) for lat, lon, mag, t, h in zip(lats, lons, mags, times, hover) if lat != 0 and lon != 0]
    if not valid:
        logging.warning("No valid event locations found. Showing fallback marker in Los Angeles.")
        fig = go.Figure(go.Scattermapbox(
            lat=[34.05], lon=[-118.25], mode='markers',
            marker=dict(size=14, color='red'),
            text=["No valid events found. Check event data."], hoverinfo='text'
        ))
        fig.update_layout(
            mapbox_style="open-street-map",
            mapbox_zoom=5, mapbox_center={"lat": 34.5, "lon": -118},
            margin=dict(l=0, r=0, t=0, b=0),
            height=400
        )
        return fig
    lats, lons, mags, times, hover = zip(*valid)
    fig = go.Figure(go.Scattermapbox(
        lat=lats, lon=lons, mode='markers',
        marker=dict(size=[6+2*m for m in mags], color=mags, colorscale='Viridis', showscale=True, colorbar=dict(title='Mag')),
        text=hover, hoverinfo='text', customdata=times
    ))
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=5, mapbox_center={"lat": 34.5, "lon": -118},
        margin=dict(l=0, r=0, t=0, b=0),
        height=400
    )
    return fig

def create_event_timeline(events):
    if not events:
        return go.Figure()
    times = [ev['Raw'].get('Time', '') for ev in events]
    mags = [float(ev['Raw'].get('Magnitude', 0)) for ev in events]
    # Convert times to datetime for sorting
    times_dt = pd.to_datetime(times)
    fig = go.Figure(go.Scatter(
        x=times_dt, y=mags, mode='markers',
        marker=dict(size=[6+2*m for m in mags], color=mags, colorscale='Viridis', showscale=True),
        text=[f"<b>{t}</b><br>Mag: {m}" for t, m in zip(times, mags)],
        hoverinfo='text', customdata=times
    ))
    fig.update_layout(
        xaxis_title='Time', yaxis_title='Magnitude',
        title='Event Timeline', height=250,
        margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig

def create_station_map(stations, event_lat, event_lon):
    # stations: list of station codes (strings)
    # For demo, plot stations in a circle around the event
    if not stations or event_lat is None or event_lon is None:
        return go.Figure()
    n = len(stations)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    radius = 0.5  # degrees
    lats = [event_lat + radius * np.cos(a) for a in angles]
    lons = [event_lon + radius * np.sin(a) for a in angles]
    fig = go.Figure(go.Scattermapbox(
        lat=lats, lon=lons, mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=stations, textposition='top right',
        hoverinfo='text', customdata=stations
    ))
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=7, mapbox_center={"lat": event_lat, "lon": event_lon},
        margin=dict(l=0, r=0, t=0, b=0),
        height=350
    )
    return fig

def create_app():
    logger.info("🚀 Creating next-gen SCEDC Gradio app with map, timeline, and filters...")
    with gr.Blocks(theme=gr.themes.Monochrome(), css="""
        .gr-box {background: #f8fafc; border-radius: 12px; box-shadow: 0 2px 8px #0001;}
        .section-header {font-size: 1.3em; font-weight: bold; margin: 1em 0 0.5em 0; color: #2b4162;}
        .gr-button {font-size: 1.1em;}
        .gr-textbox, .gr-number {margin-bottom: 0.5em;}
    """) as app:
        gr.Markdown("""
        <div style='text-align:center; margin-bottom:1em;'>
        <span style='font-size:2em;'>🌍 <b>SCEDC Interactive Seismic Explorer</b></span><br>
        <span style='color:#2b4162;'>Search, visualize, and explore Southern California earthquakes</span>
        </div>
        """)
        with gr.Row():
            with gr.Column():
                gr.Markdown("<div class='section-header'>🔎 Event Search</div>")
                start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", value="2024-12-01")
                end_date = gr.Textbox(label="End Date (YYYY-MM-DD)", value="2024-12-31")
                min_mag = gr.Number(label="Minimum Magnitude", value=3.0, minimum=0.0, maximum=10.0)
                max_mag = gr.Number(label="Maximum Magnitude", value=None, minimum=0.0, maximum=10.0)
                search_btn = gr.Button("🔍 Search Events", variant="primary")
                search_status = gr.Textbox(label="Status", interactive=False)
            with gr.Column():
                gr.Markdown("<div class='section-header'>⚙️ Waveform Filter Controls</div>")
                bandpass_min = gr.Number(label="Bandpass Min (Hz)", value=0.1)
                bandpass_max = gr.Number(label="Bandpass Max (Hz)", value=10.0)
                highpass = gr.Number(label="Highpass (Hz)", value=0.5)
        with gr.Row():
            with gr.Column():
                gr.Markdown("<div class='section-header'>📋 Event Selection</div>")
                event_dropdown = gr.Dropdown(label="Select Event", choices=[], value=None, interactive=True, allow_custom_value=False)
            with gr.Column():
                gr.Markdown("<div class='section-header'>📍 Station Selection</div>")
                station_dropdown = gr.Dropdown(label="Select Station", choices=[], value=None, interactive=True, allow_custom_value=False)
                fetch_waveform_btn = gr.Button("📊 Fetch Waveform", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("<div class='section-header'>🗺️ Event Map</div>")
                event_map = gr.Plot(label=None)
            with gr.Column():
                gr.Markdown("<div class='section-header'>📈 Event Timeline</div>")
                event_timeline = gr.Plot(label=None)
        with gr.Row():
            station_map = gr.Plot(label="Station Map")
        with gr.Row():
            plot_output = gr.Plot(label="Waveform Plot (with phases)")
            phasenet_plot = gr.Plot(label="PhaseNet Picks")
            data_info = gr.Textbox(label="Data Information", lines=8, interactive=False)
        # State variables
        event_list_state = gr.State([])
        selected_event_state = gr.State(None)
        event_station_map_state = gr.State({})
        
        # Search callback
        def do_search(start, end, minm, maxm):
            events, event_station_map, msg = get_event_station_phase_info(start, end, minm, maxm)
            if not events:
                logger.info("No events found, returning empty choices.")
                return (
                    msg,
                    gr.update(choices=[], value=None),
                    go.Figure(),
                    go.Figure(),
                    {},
                    None,
                    go.Figure(),
                    gr.update(choices=[], value=None),
                    [],
                )
            event_choices = []
            for i, event in enumerate(events):
                time_str = event['Time']
                mag = event['Magnitude']
                location = event['Location']
                summary = f"{time_str} | M{mag} | {location[:50]}..."
                event_choices.append(summary)
            logger.info(f"Event choices for dropdown: {event_choices}")
            map_fig = create_event_map(events)
            timeline_fig = create_event_timeline(events)
            return (
                f"✓ {len(events)} events loaded",
                gr.update(choices=event_choices, value=None),
                map_fig,
                timeline_fig,
                event_station_map,
                None,
                go.Figure(),
                gr.update(choices=[], value=None),
                events,
            )
        
        search_btn.click(
            do_search,
            inputs=[start_date, end_date, min_mag, max_mag],
            outputs=[search_status, event_dropdown, event_map, event_timeline, event_station_map_state, selected_event_state, station_map, station_dropdown, event_list_state]
        )
        
        # Event selection callback
        def on_event_select(event_choice, events, event_station_map):
            if not event_choice or not events:
                return None, [], go.Figure()
            
            # Find the selected event by matching the choice string
            selected_event = None
            for event in events:
                time_str = event['Time']
                mag = event['Magnitude']
                location = event['Location']
                summary = f"{time_str} | M{mag} | {location[:50]}..."
                if summary == event_choice:
                    selected_event = event
                    break
            
            if not selected_event:
                return None, [], go.Figure()
            
            # Get event lat/lon
            raw = selected_event['Raw']
            try:
                lat = float(raw.get('Latitude', 0))
                lon = float(raw.get('Longtitude', 0))
            except:
                lat, lon = 34.05, -118.25  # Default to LA if coordinates are invalid
            
            # Get stations for this event
            event_id = selected_event['Time']
            stations = event_station_map.get(event_id, [])
            
            # Create station dropdown choices
            station_choices = stations[:20]  # Limit to first 20 stations
            
            # Create station map
            station_fig = create_station_map(stations, lat, lon)
            
            return (
                selected_event,
                gr.update(choices=station_choices, value=None),
                station_fig,
            )
        
        event_dropdown.change(
            on_event_select,
            inputs=[event_dropdown, event_list_state, event_station_map_state],
            outputs=[selected_event_state, station_dropdown, station_map]
        )
        
        # Waveform fetch callback
        def fetch_waveform(selected_event, station, bandpass_min, bandpass_max, highpass):
            if not selected_event or not station:
                return None, None, "Please select an event and station first."
            
            logger.info(f"📊 Fetching waveform for event: {selected_event['Time']}, station: {station}")
            
            # Parse event time
            event_time = selected_event['Time']
            try:
                # Convert to UTCDateTime
                dt = UTCDateTime(event_time)
                start_time = dt - 60  # 1 minute before
                end_time = dt + 300   # 5 minutes after
            except Exception as e:
                logger.error(f"Error parsing event time: {e}")
                return None, None, f"Error parsing event time: {e}"
            
            # Fetch waveform data
            scedc = SCEDCInterface()
            # Retrieve three-component broadband data for PhaseNet (Z,N,E)
            st = scedc.get_waveform_data(
                start_time.isoformat(),
                end_time.isoformat(),
                "CI",               # Network
                station,
                "BH?"               # Channel pattern -> BHZ, BHN, BHE
            )
            
            if st is None:
                return None, None, f"No waveform data available for {station}"
            
            # TauP arrivals as reference
            taup_arrs = compute_phase_arrivals(selected_event["Raw"], "CI", station)

            # Try SeisBench classification first
            phase_arrs: list[dict] = []
<<<<<<< HEAD
            if HAS_SEISBENCH:
                phase_arrs = compute_seisbench_arrivals(st)

            # Fall back to obspy-phasenet
            if not phase_arrs and HAS_PHASENET:
                phase_arrs = compute_phasenet_arrivals(st, selected_event['Raw'], selected_event['Raw'].get('Network'), selected_event['Raw'].get('Station'))

            # If still none, use TauP estimates for plotting
            if not phase_arrs:
                phase_arrs = taup_arrs

            # Create plot with custom filter parameters
            fig = plot_waveform(st, phase_arrivals=phase_arrs, event_time=event_time, station=station)

            # Compute delta between TauP and classified picks
            delta_lines = []
            phase_dict = {p["phase"].upper(): p for p in phase_arrs}
            for t in taup_arrs:
                phase = t["phase"].upper()
                if phase in phase_dict:
                    delta = phase_dict[phase]["time"] - t["time"]
                    delta_lines.append(f"{phase} Δ: {delta:.2f}s")

            if fig is None:
                return None, "Error creating waveform plot"

=======
            picks_for_plot = []
            if HAS_PHASENET:
                phase_arrs = compute_phasenet_arrivals(
                    st,
                    selected_event['Raw'],
                    selected_event['Raw'].get('Network'),
                    selected_event['Raw'].get('Station'),
                )
                # Run a second time (cheap) to collect all raw picks for plotting
                try:
                    classify_out = PN_MODEL.classify(st, batch_size=64)
                    picks_for_plot = getattr(classify_out, "picks", [])
                except Exception as e:
                    logger.error(f"PhaseNet classify error for plotting: {e}")

            # Create plot with custom filter parameters
            fig = plot_waveform(st, phase_arrivals=phase_arrs, event_time=event_time, station=station)
            
            # Build separate PhaseNet picks figure
            picks_fig = plot_phasenet_picks(picks_for_plot, st, selected_event['Raw']) if picks_for_plot else None
            
            if fig is None:
                return None, None, "Error creating waveform plot"
            
>>>>>>> c795fa4 (Integrate SeisBench PhaseNet Model for Enhanced Phase Picking)
            # Create info text
            info = f"Event: {event_time}\n"
            info += f"Station: {station}\n"
            info += f"Network: CI\n"
            info += f"Channel: BHZ\n"
            info += f"Time window: {start_time} to {end_time}\n"
            info += f"Filters: Bandpass {bandpass_min}-{bandpass_max} Hz, Highpass {highpass} Hz\n"
            info += f"Traces: {len(st)}"
<<<<<<< HEAD
            if delta_lines:
                info += "\n" + "; ".join(delta_lines)

            return fig, info
=======
            
            return fig, picks_fig, info
>>>>>>> c795fa4 (Integrate SeisBench PhaseNet Model for Enhanced Phase Picking)
        
        fetch_waveform_btn.click(
            fetch_waveform,
            inputs=[selected_event_state, station_dropdown, bandpass_min, bandpass_max, highpass],
            outputs=[plot_output, phasenet_plot, data_info]
        )
    
    logger.info("✅ Next-gen SCEDC Gradio app with event selection ready.")
    return app

# ---------------------------------------------------------------------------
# Phase arrival computation using TauP and station metadata
# ---------------------------------------------------------------------------

def compute_phase_arrivals(event_raw: dict, network: str, station: str):
    """Return P & S arrival dicts using TauP (iasp91) given event & station."""
    try:
        ev_lat = float(event_raw.get("Latitude", 0))
        ev_lon = float(event_raw.get("Longtitude", 0))
        depth = float(event_raw.get("Depth/km", 10))  # km

        # Fetch station metadata for coordinates
        scedc = SCEDCInterface()
        xml_txt = scedc.get_station_metadata(network, station)
        if not xml_txt:
            logger.warning("Station metadata unavailable, skipping TauP arrivals.")
            return []
        root = ET.fromstring(xml_txt)
        sta_elem = root.find('.//{*}Station')
        if sta_elem is None:
            logger.warning("Station element not found in StationXML.")
            return []
        sta_lat = float(sta_elem.find('.//{*}Latitude').text)
        sta_lon = float(sta_elem.find('.//{*}Longitude').text)

        distance_deg = locations2degrees(ev_lat, ev_lon, sta_lat, sta_lon)
        model = TauPyModel(model="iasp91")
        arrivals = model.get_travel_times(source_depth_in_km=depth, distance_in_degree=distance_deg, phase_list=["P", "S"])

        out = []
        for arr in arrivals:
            out.append({"phase": arr.name, "time": arr.time, "abs_time": f"{arr.time:.1f}s"})
        logger.info(f"Computed TauP arrivals: {out}")
        return out
    except Exception as e:
        logger.error(f"TauP arrival computation error: {e}")
        return []

# ---------------------------------------------------------------------------
# PhaseNet picker
# ---------------------------------------------------------------------------

def compute_phasenet_arrivals(st, event_raw: dict | None, network: str | None, station: str | None):
    """Run PhaseNet picker (SeisBench) and filter picks close to TauP estimates (±4 s).

    Returns a list of dicts::
        [{"phase": "P", "time": 12.3, "abs_time": "12.3s"}, ...]
    """

    if not HAS_PHASENET or PN_MODEL is None:
        logger.info("PhaseNet not available; skipping ML picks.")
        return []

    # If event + station metadata present, compute theoretical arrivals for pruning
    taup_dict: dict[str, float]
    if event_raw and network and station:
        taup_arrs = compute_phase_arrivals(event_raw, network, station)
        taup_dict = {a["phase"].upper(): a["time"] for a in taup_arrs}
    else:
        taup_dict = {}

    # Determine event origin time (UTCDateTime); needed to derive relative seconds
    if event_raw and event_raw.get("Time"):
        try:
            origin_time = UTCDateTime(event_raw["Time"])
        except Exception:
            origin_time = None
    else:
        origin_time = None

    try:
        logger.info("🔮 Running PhaseNet (SeisBench) picker …")
        # SeisBench models accept obspy.Stream directly
        classify_out = PN_MODEL.classify(st, batch_size=64)
        picks_list = getattr(classify_out, "picks", [])
        logger.info(f"🔮 PhaseNet produced {len(picks_list)} picks")

        tolerance = 4.0  # seconds window around TauP prediction
        selected: dict[str, dict] = {}

        for p in picks_list:
            phase = (p.phase or "").upper()
            if phase not in ("P", "S"):
                continue

            # Compute time relative to event origin (if known) else trace start offset
            if origin_time is not None and p.peak_time is not None:
                rel_event = p.peak_time - origin_time
            else:
                # Fall back to difference to trace start minus 60 s window offset
                rel_event = (p.peak_time - st[0].stats.starttime) - 60.0  # type: ignore

            if phase in taup_dict and abs(rel_event - taup_dict[phase]) > tolerance:
                continue  # Discard outliers beyond tolerance

            if phase not in selected or rel_event < selected[phase]["time"]:
                selected[phase] = {
                    "phase": phase,
                    "time": float(rel_event),
                    "abs_time": f"{rel_event:.1f}s",
                }

        arrivals = list(selected.values())
        logger.info(
            f"🔮 PhaseNet filtered to {len(arrivals)} pick(s) within ±{tolerance}s of TauP"
        )
        return arrivals
    except Exception as e:
        logger.error(f"PhaseNet picking error: {e}")
        return []

# ---------------------------------------------------------------------------
<<<<<<< HEAD
# SeisBench PhaseNet picker (if available)
# ---------------------------------------------------------------------------

def compute_seisbench_arrivals(st):
    """Run SeisBench PhaseNet classification on stream."""
    if not HAS_SEISBENCH:
        logger.info("SeisBench not installed; skipping classification.")
        return []

    try:
        logger.info("🔮 Running SeisBench PhaseNet classifier...")
        model = SBPhaseNet.from_pretrained("phasenet")
        df = model.classify(st)
        picks = []
        for _, row in df.iterrows():
            phase = str(row.get("phase", "")).upper()
            if phase in ("P", "S"):
                picks.append({"phase": phase, "time": float(row.get("time", 0.0)), "abs_time": f"{float(row.get('time', 0.0)):.1f}s"})
        logger.info(f"🔮 SeisBench returned {len(picks)} picks")
        return picks
    except Exception as e:
        logger.error(f"SeisBench classification error: {e}")
        return []
=======
# Helper to visualise raw PhaseNet picks
# ---------------------------------------------------------------------------

def plot_phasenet_picks(picks: list, st, event_raw: dict | None):
    """Return a Matplotlib figure of PhaseNet pick times per phase."""

    try:
        import matplotlib.pyplot as plt

        # Determine reference origin time (UTCDateTime)
        if event_raw and event_raw.get("Time"):
            try:
                origin_time = UTCDateTime(event_raw["Time"])
            except Exception:
                origin_time = None
        else:
            origin_time = None

        xs_p, xs_s = [], []
        for p in picks:
            phase = (p.phase or "").upper()
            if phase not in ("P", "S"):
                continue

            if origin_time is not None and p.peak_time is not None:
                t_rel = p.peak_time - origin_time
            else:
                t_rel = (p.peak_time - st[0].stats.starttime) - 60.0  # type: ignore

            if phase == "P":
                xs_p.append(float(t_rel))
            else:
                xs_s.append(float(t_rel))

        fig, ax = plt.subplots(figsize=(12, 2))
        if xs_p:
            ax.eventplot(xs_p, lineoffsets=1, colors="red", linewidths=2, label="P picks")
        if xs_s:
            ax.eventplot(xs_s, lineoffsets=0, colors="blue", linewidths=2, label="S picks")

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["S", "P"])
        ax.set_xlabel("Time (s from event origin)")
        ax.set_title("PhaseNet predicted pick times")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Error plotting PhaseNet picks: {e}")
        return None
>>>>>>> c795fa4 (Integrate SeisBench PhaseNet Model for Enhanced Phase Picking)

if __name__ == "__main__":
    logger.info("🌍 Starting SCEDC Interactive Gradio App...")
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True) 