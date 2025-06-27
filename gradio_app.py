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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

def plot_waveform_plotly(
    st,
    phase_arrivals=None,
    event_time=None,
    bandpass=(0.1, 10.0),
    highpass=0.5,
    ann_stream=None,
):
    """Return a Plotly figure with raw & filtered traces plus arrival markers.

    Parameters
    ----------
    st : obspy.Stream
        Raw waveform traces.
    phase_arrivals : list[dict] | None
        Arrivals to plot as vertical lines (keys: phase, time).
    event_time : str | None
        Event origin time string (UTC). Used for zero reference.
    bandpass : tuple[float, float]
        (min, max) Hz bandpass filter bounds.
    highpass : float
        High-pass corner frequency (Hz).
    ann_stream : obspy.Stream | None
        Stream of probability traces returned by PN_MODEL.annotate.
    """
    logger.info("🎨 Creating Plotly waveform plot…")
    
    if st is None or len(st) == 0:
        logger.warning("   ⚠ No data to plot")
        return None
    
    try:
        # Apply filters (same as before) ------------------------------------
        logger.info("   🔧 Applying seismic filters…")
        st_filtered = st.copy()
        
        for i, tr in enumerate(st_filtered):
            logger.info(f"   📊 Filtering trace {i+1}: {tr.stats.network}.{tr.stats.station}.{tr.stats.channel}")
            
            # Remove mean and linear trend
            tr.detrend('linear')
            tr.detrend('constant')
            logger.info(f"     ✓ Removed trend and mean")
            
            # Apply bandpass filter to enhance P and S waves
            tr.filter('bandpass', freqmin=bandpass[0], freqmax=bandpass[1], corners=4, zerophase=True)
            logger.info(f"     ✓ Applied bandpass filter ({bandpass[0]}-{bandpass[1]} Hz)")
            
            # Apply high-pass filter to remove long-period noise
            if highpass and highpass > 0:
                tr.filter('highpass', freq=highpass, corners=4, zerophase=True)
            logger.info(f"     ✓ Applied high-pass filter ({highpass} Hz)")
            
            # Optional: Apply a notch filter to remove power line interference (60 Hz in US)
            # tr.filter('bandstop', freqmin=58, freqmax=62, corners=4, zerophase=True)
            # logger.info(f"     ✓ Applied notch filter (58-62 Hz)")
        
        # Determine time shift to align with origin (if known)
        if event_time:
            try:
                origin_dt = UTCDateTime(event_time)
                shift_sec = st[0].stats.starttime - origin_dt
            except Exception:
                shift_sec = 0.0
        else:
            shift_sec = 0.0

        # Build Plotly subplots --------------------------------------------
        # How many subplot rows? 2 (raw, filtered) + optional probabilities
        n_rows = 3 if ann_stream is not None else 2
        titles = ["Raw Seismic Data", "Filtered Data"]
        if n_rows == 3:
            titles.append("PhaseNet Probabilities")

        # Row height ratios: raw 2, filtered 2, probabilities 1
        if n_rows == 3:
            row_heights = [2, 2, 1]
        else:
            row_heights = None  # default equal heights

        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=tuple(titles),
            row_heights=row_heights,
        )

        # Raw traces
        for tr in st:
            times = shift_sec + (np.arange(len(tr.data)) / tr.stats.sampling_rate)
            fig.add_trace(go.Scatter(x=times, y=tr.data,
                                     name=f"{tr.stats.channel} (Raw)",
                                     line=dict(width=0.8, color="gray"), opacity=0.7),
                          row=1, col=1)

        # Filtered traces
        for tr in st_filtered:
            times = shift_sec + (np.arange(len(tr.data)) / tr.stats.sampling_rate)
            fig.add_trace(go.Scatter(x=times, y=tr.data,
                                     name=f"{tr.stats.channel} (Filtered)",
                                     line=dict(width=0.8, color="blue")),
                          row=2, col=1)
        
        # Phase arrivals as vertical lines in second subplot
        if phase_arrivals:
            for arr in phase_arrivals:
                t = arr["time"]
                col = "red" if arr["phase"].upper() == "P" else "blue"
                fig.add_vline(x=t, line_width=2, line_dash="dash", line_color=col, row=2, col=1)

        # Event origin vertical line
        fig.add_vline(x=0, line_width=3, line_color="black", row="all", col=1)

        fig.update_yaxes(title_text="Amplitude (counts)", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude (counts)", row=2, col=1)

        last_row = n_rows
        # Probabilities plot -------------------------------------------------
        if ann_stream is not None:
            p_tr = None
            s_tr = None
            for tr in ann_stream:
                ch = tr.stats.channel.upper()
                if "P" in ch and p_tr is None:
                    p_tr = tr
                elif "S" in ch and s_tr is None:
                    s_tr = tr

            for tr, col, lab in ((p_tr, "red", "P prob"), (s_tr, "blue", "S prob")):
                if tr is None:
                    continue
                times_prob = shift_sec + (np.arange(tr.stats.npts) / tr.stats.sampling_rate)
                fig.add_trace(
                    go.Scatter(x=times_prob, y=tr.data, name=lab, line=dict(color=col)),
                    row=3, col=1,
                )
            fig.update_yaxes(title_text="Probability", row=3, col=1, range=[0, 1])
            fig.update_xaxes(title_text="Time (s from origin)", row=3, col=1)
        else:
            fig.update_xaxes(title_text="Time (s from origin)", row=2, col=1)

        title_main = "Seismic Waveform"
        if event_time:
            title_main += f"<br>Event: {event_time}"
        base_height = 600
        if n_rows == 3:
            base_height += 200
        fig.update_layout(title=title_main, height=base_height, legend=dict(orientation="h"))
        
        logger.info("   ✓ Plotly waveform figure ready")
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
    ci_lat, ci_lon = 34.5, -118.0  # Rough centroid of CI network (SoCal)

    for i, line in enumerate(data_lines):
        values = [v.strip() for v in line.split('|')]
        if len(values) < len(header):
            continue
        event = dict(zip(header, values))
        if i == 0:
            logger.info(f"Sample event dict: {event}")
        event_id = event.get('Time', f'Event {i+1}')
        try:
            ev_lat = float(event.get('Latitude', 0))
            ev_lon = float(event.get('Longtitude', 0))
            dist_deg = locations2degrees(ev_lat, ev_lon, ci_lat, ci_lon)
            dist_km = dist_deg * 111.195  # rough km
        except Exception:
            dist_km = None

        event_list.append({
            'Time': event.get('Time', 'N/A'),
            'Magnitude': float(event.get('Magnitude', 0)),
            'Location': event.get('EventLocationName', 'N/A'),
            'Distance_km': dist_km,
            'Raw': event
        })
        event_time_map[event_id] = event.get('Time', '')
        # Fetch stations with phase picks for this event
        stations = get_stations_with_phases(event.get('Time', ''))
        event_station_map[event_id] = stations

    # Sort by distance (nearest first)
    event_list_sorted = sorted(
        event_list,
        key=lambda e: e.get('Distance_km', float('inf'))
    )

    return event_list_sorted, event_station_map, ""

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
    """Create and configure the Gradio application"""
    logger.info("🚀 Creating Gradio app interface...")
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto !important;
    }
    .plot-container {
        min-height: 500px;
    }
    .info-panel {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        padding: 15px;
    }
    """
    
    with gr.Blocks(
        title="🌍 SCEDC Seismic Data Explorer", 
        theme=gr.themes.Soft(),
        css=custom_css
    ) as app:
        
        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #4CAF50, #2196F3); border-radius: 15px; margin-bottom: 20px;">
            <h1 style="color: white; margin: 0; font-size: 2.5em;">🌍 SCEDC Seismic Data Explorer</h1>
            <p style="color: white; margin: 10px 0 0 0; font-size: 1.2em;">Interactive earthquake analysis with PhaseNet ML picking</p>
        </div>
        """)
        
        # ----------------------  NEW LAYOUT  ----------------------
        with gr.Row():
            # ==== SIDEBAR CONTROLS ====
            with gr.Column(scale=1, min_width=320):
                with gr.Accordion("🗓️ Search Parameters", open=False):
                    start_date = gr.Textbox(
                        label="Start Date (YYYY-MM-DD)", value="2024-12-20"
                    )
                    end_date = gr.Textbox(
                        label="End Date (YYYY-MM-DD)", value="2024-12-27"
                    )
                    with gr.Row():
                        min_magnitude = gr.Number(label="Min Mag", value=3.0, minimum=0, maximum=10)
                        max_magnitude = gr.Number(label="Max Mag", value=8.0, minimum=0, maximum=10)
                    search_btn = gr.Button("🔍 Search / Refresh", variant="primary")

                event_dropdown = gr.Dropdown(label="📋 Select Event", choices=[], interactive=True)
                station_dropdown = gr.Dropdown(label="🏗️ Station", choices=[], interactive=True)

                with gr.Accordion("🔧 Filters", open=False):
                    bandpass_min = gr.Number(label="Bandpass Min (Hz)", value=0.1, minimum=0.01, maximum=50)
                    bandpass_max = gr.Number(label="Bandpass Max (Hz)", value=10.0, minimum=0.1, maximum=50)
                    highpass = gr.Number(label="High-pass (Hz)", value=0.5, minimum=0.01, maximum=10)

                fetch_waveform_btn = gr.Button("📊 Analyze Waveform", variant="primary", size="lg")

                event_info = gr.HTML(elem_classes=["info-panel"])
                data_info = gr.Textbox(label="📋 Data Info", lines=6, interactive=False)

            # ==== MAIN DISPLAY AREA ====
            with gr.Column(scale=3):
                combined_map = gr.Plot(label="Events & Stations Map", elem_classes=["plot-container"])
                waveform_plot = gr.Plot(label="Seismic Waveforms", elem_classes=["plot-container"])

                # Debug accordion (hidden by default)
                phasenet_debug = gr.Textbox(label="Raw Pick Details", lines=10, interactive=False, visible=False)

        # State management
        events_state = gr.State([])
        selected_event_state = gr.State(None)
        
        # Event handlers
        def search_events_enhanced(start, end, min_mag, max_mag):
            """Enhanced event search with better UI feedback"""
            try:
                events, event_station_map, msg = get_event_station_phase_info(start, end, min_mag, max_mag)
                
                # Create enhanced event map
                map_fig = create_combined_map(events, None, None)
                
                # Format dropdown choices
                choices = []
                for i, event in enumerate(events):
                    mag = event.get('Magnitude', 'Unknown')
                    dist = event.get('Distance_km')
                    dist_str = f"{dist:.0f} km" if dist is not None else "? km"
                    time_str = event.get('Time', 'Unknown time')[:16]
                    choices.append(f"{dist_str} | M{mag} | {time_str}")
                
                return (
                    gr.update(choices=choices, value=None),  # event_dropdown
                    events,  # events_state
                    map_fig,  # combined_map
                    f"<h4>🎯 Found {len(events)} events</h4><p>{msg}</p>",  # event_info
                    gr.update(choices=[], value=None)  # station_dropdown (reset)
                )
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return gr.update(), [], None, f"<h4>❌ Search failed</h4><p>{str(e)}</p>", gr.update()
        
        def on_event_selection(event_choice, events):
            """Handle event selection with enhanced feedback"""
            if not event_choice or not events:
                return None, None, gr.update(choices=[]), "<h4>No event selected</h4>"
            
            try:
                event_idx = next(i for i, event in enumerate(events) if event['Time'][:16] in event_choice)
                selected_event = events[event_idx]
                
                # Get candidate stations and keep only those with data
                candidate_stations = get_stations_with_phases(selected_event['Time'])
                stations = filter_stations_with_waveforms(selected_event['Time'], candidate_stations)
                
                # Combined event & station map
                combined_map_fig = create_combined_map(events, selected_event, stations)
                
                # Enhanced event info
                event_html = f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 10px;">
                    <h4>🎯 Selected Event</h4>
                    <p><strong>📅 Time:</strong> {selected_event.get('Time', 'Unknown')}</p>
                    <p><strong>📏 Magnitude:</strong> {selected_event.get('Magnitude', 'Unknown')}</p>
                    <p><strong>📍 Location:</strong> {selected_event.get('Location', 'Unknown')}</p>
                    <p><strong>🏗️ Stations available:</strong> {len(stations)}</p>
                </div>
                """
                
                # Preselect first available station if present
                station_choices = [f"{s} (CI network)" for s in stations]
                default_station_val = station_choices[0] if station_choices else None
                return selected_event, combined_map_fig, gr.update(choices=station_choices, value=default_station_val), event_html
                
            except Exception as e:
                logger.error(f"Event selection failed: {e}")
                return None, None, gr.update(choices=[]), f"<h4>❌ Selection failed</h4><p>{str(e)}</p>"
        
        def analyze_waveform_enhanced(selected_event, station_choice, bp_min, bp_max, hp):
            """Enhanced waveform analysis with better error handling"""
            if not selected_event:
                return None, "", "⚠️ Please select an event first."

            # If user did not pick a station, automatically select the first available one
            if not station_choice:
                candidate_stations = get_stations_with_phases(selected_event["Time"])
                stations_avail = filter_stations_with_waveforms(selected_event["Time"], candidate_stations)
                if not stations_avail:
                    return None, "", "⚠️ No stations with data found for this event."
                station_choice = f"{stations_avail[0]} (CI network)"

            try:
                station = station_choice.split()[0]  # Extract station code
                logger.info(f"🔬 Analyzing: {selected_event['Time']} at {station}")
                
                # Call existing fetch_waveform logic but with enhanced returns
                fig, debug_txt, info = fetch_waveform(
                    selected_event, station, bp_min, bp_max, hp
                )
                
                return fig, debug_txt, info
                
            except Exception as e:
                logger.error(f"Waveform analysis failed: {e}")
                error_msg = f"❌ Analysis failed: {str(e)}"
                return None, "", error_msg
        
        # Wire up the enhanced event handlers
        search_btn.click(
            search_events_enhanced,
            inputs=[start_date, end_date, min_magnitude, max_magnitude],
            outputs=[event_dropdown, events_state, combined_map, event_info, station_dropdown]
        )
        
        event_dropdown.change(
            on_event_selection,
            inputs=[event_dropdown, events_state],
            outputs=[selected_event_state, combined_map, station_dropdown, event_info]
        )
        
        fetch_waveform_btn.click(
            analyze_waveform_enhanced,
            inputs=[selected_event_state, station_dropdown, bandpass_min, bandpass_max, highpass],
            outputs=[waveform_plot, phasenet_debug, data_info]
        )

        # ---------------------------------------------------------------
        # App auto-load: fetch default event/station & initial waveform
        # ---------------------------------------------------------------

        def _app_default_load():
            """Populate UI with first available event, station & waveform on startup."""

            # Use the same defaults as the visible search boxes
            start_default = start_date.value
            end_default = end_date.value
            min_mag = float(min_magnitude.value)
            max_mag = float(max_magnitude.value)

            # --- Event search ---
            events, _, _ = get_event_station_phase_info(start_default, end_default, min_mag, max_mag)
            if not events:
                return [gr.update(), [], None, None, "<h4>No default events</h4>", gr.update(), None, None, None, None, "", ""]

            first_event = events[0]
            choices = []
            for ev in events:
                dist= ev.get('Distance_km'); dist_str=f"{dist:.0f} km" if dist is not None else "? km"; choices.append(f"{dist_str} | M{ev['Magnitude']} | {ev['Time'][:16]}")

            # --- Station selection ---
            candidate_stations = get_stations_with_phases(first_event['Time'])
            stations = filter_stations_with_waveforms(first_event['Time'], candidate_stations)
            station_choices = [f"{s} (CI network)" for s in stations]
            first_station_label = station_choices[0] if station_choices else None

            combined_map_fig = create_combined_map(events, first_event, stations)

            event_html = f"<h4>Auto-selected Event M{first_event['Magnitude']} @ {first_event['Time']}</h4>"

            # --- Initial waveform analysis ---
            if first_station_label:
                wave_fig, debug_txt, info_txt = fetch_waveform(
                    first_event, stations[0], bandpass_min.value, bandpass_max.value, highpass.value
                )
            else:
                wave_fig = None
                debug_txt = info_txt = ""

            return [
                gr.update(choices=choices, value=choices[0]),  # event_dropdown
                events,  # events_state
                combined_map_fig,
                first_event,  # selected_event_state
                wave_fig,
                debug_txt,
                info_txt,
            ]

        # Run default load when the app starts
        app.load(
            _app_default_load,
            inputs=None,
            outputs=[
                event_dropdown,
                events_state,
                combined_map,
                selected_event_state,
                waveform_plot,
                phasenet_debug,
                data_info,
            ],
        )
    
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
# Helper to visualise raw PhaseNet picks
# ---------------------------------------------------------------------------

def plot_phasenet_picks(picks: list, st, event_raw: dict | None):
    """Return a Matplotlib figure of PhaseNet pick times per phase."""

    try:
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

# ---------------------------------------------------------------------------
# Probability plot helper
# ---------------------------------------------------------------------------

def plot_phasenet_probabilities(ann_stream, event_raw):
    """Plot P & S probability traces returned by PN_MODEL.annotate."""
    try:
        # Origin reference for x-axis
        if event_raw and event_raw.get("Time"):
            try:
                origin_time = UTCDateTime(event_raw["Time"])
            except Exception:
                origin_time = None
        else:
            origin_time = None

        p_tr = None
        s_tr = None
        for tr in ann_stream:
            ch = tr.stats.channel.upper()
            if "P" in ch and p_tr is None:
                p_tr = tr
            elif "S" in ch and s_tr is None:
                s_tr = tr

        fig, ax = plt.subplots(figsize=(12, 3))
        for tr, col, lab in ((p_tr, "red", "P prob"), (s_tr, "blue", "S prob")):
            if tr is None:
                continue
            times = np.arange(tr.stats.npts) / tr.stats.sampling_rate
            if origin_time is not None:
                # shift so that 0 = event origin
                delta = (tr.stats.starttime - origin_time)
                times = times + delta
            ax.plot(times, tr.data, color=col, alpha=0.8, label=lab)

        ax.set_xlabel("Time (s from origin)")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig
    except Exception as e:
        logger.error(f"Prob plot error: {e}")
        return None

# ---------------------------------------------------------------------------
# Combined map helper (events + stations together)
# ---------------------------------------------------------------------------

def create_combined_map(events, selected_event, stations, selected_station=None):
    """Return a Plotly mapbox figure showing events and station locations."""

    # Base: plot all events (grey) -------------------------------------------------
    lats_e, lons_e, mags_e, hover_e = [], [], [], []
    for ev in events or []:
        raw = ev["Raw"]
        try:
            lat = float(raw.get("Latitude", 0))
            lon = float(raw.get("Longtitude", 0))
        except Exception:
            continue
        mag = float(raw.get("Magnitude", 0))
        lats_e.append(lat)
        lons_e.append(lon)
        mags_e.append(mag)
        hover_e.append(f"M{mag:.2f}<br>{raw.get('Time','')}")

    fig = go.Figure()
    if lats_e:
        fig.add_trace(go.Scattermapbox(
            lat=lats_e,
            lon=lons_e,
            mode="markers",
            marker=dict(size=[6+2*m for m in mags_e], color="lightgray"),
            hoverinfo="text",
            text=hover_e,
            name="Events"
        ))

    # Highlight selected event -----------------------------------------------------
    if selected_event:
        try:
            lat = float(selected_event["Raw"].get("Latitude", 0))
            lon = float(selected_event["Raw"].get("Longtitude", 0))
            fig.add_trace(go.Scattermapbox(
                lat=[lat],
                lon=[lon],
                mode="markers",
                marker=dict(size=14, color="red"),
                name="Selected Event",
                hoverinfo="skip"
            ))
        except Exception:
            pass

    # Stations --------------------------------------------------------------------
    if stations:
        sta_lats = []
        sta_lons = []
        for sta in stations:
            lat_s, lon_s = get_station_coords("CI", sta)  # CI network hard-coded for now
            if lat_s is not None and lon_s is not None:
                sta_lats.append(lat_s)
                sta_lons.append(lon_s)

        # Fallback to circle around event when coords unavailable
        if not sta_lats or not sta_lons:
            if selected_event:
                ev_lat = float(selected_event["Raw"].get("Latitude", 34))
                ev_lon = float(selected_event["Raw"].get("Longtitude", -118))
            else:
                ev_lat, ev_lon = 34.0, -118.0
            n = len(stations)
            angles = np.linspace(0, 2*np.pi, n, endpoint=False)
            radius = 0.5
            sta_lats = [ev_lat + radius * np.cos(a) for a in angles]
            sta_lons = [ev_lon + radius * np.sin(a) for a in angles]

        fig.add_trace(go.Scattermapbox(
            lat=sta_lats,
            lon=sta_lons,
            mode="markers+text",
            text=stations,
            textposition="top right",
            marker=dict(size=9, color="blue"),
            hoverinfo="text",
            name="Stations"
        ))

    if selected_station and selected_event:
        # Highlight selected station (uses same dummy coords)
        idx = stations.index(selected_station) if (stations and selected_station in stations) else None
        if idx is not None:
            fig.add_trace(go.Scattermapbox(
                lat=[fig.data[-1].lat[idx]],
                lon=[fig.data[-1].lon[idx]],
                mode="markers",
                marker=dict(size=14, color="yellow"),
                name="Selected Station",
                hoverinfo="skip"
            ))

    # Determine center
    if selected_event:
        center_lat = float(selected_event["Raw"].get("Latitude", 34.5))
        center_lon = float(selected_event["Raw"].get("Longtitude", -118))
    elif lats_e:
        center_lat = float(np.mean(lats_e))
        center_lon = float(np.mean(lons_e))
    else:
        center_lat, center_lon = 34.5, -118

    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=4,
        mapbox_center={"lat": center_lat, "lon": center_lon},
        margin=dict(l=0, r=0, t=0, b=0),
        height=350,
    )
    return fig

# ---------------------------------------------------------------------------
# Fetch waveform helper (missing implementation)
# ---------------------------------------------------------------------------

def fetch_waveform(selected_event: dict, station: str, bp_min: float, bp_max: float, hp: float):
    """Download waveform, run TauP + PhaseNet, and produce figures & debug text."""

    logger.info(f"📊 Fetching waveform for event: {selected_event['Time']}, station: {station}")

    scedc = SCEDCInterface()

    # Build time window: origin ± 3 min
    origin = UTCDateTime(selected_event["Time"])
    start = origin - 60  # 1 min before
    end = origin + 300   # 5 min after

    st = scedc.get_waveform_data(start.isoformat(), end.isoformat(), "CI", station, "BH?")
    if st is None or len(st) == 0:
        msg = f"⚠️ No waveform data for CI.{station} in this window."
        return None, None, None, "", msg

    # Theoretical arrivals (TauP)
    taup_arrivals = compute_phase_arrivals(selected_event["Raw"], "CI", station)

    # PhaseNet picks / probabilities
    pn_picks = []
    debug_lines = []
    if HAS_PHASENET and PN_MODEL is not None:
        try:
            ann_stream = PN_MODEL.annotate(st, batch_size=64)
            classify_out = PN_MODEL.classify(st, batch_size=64)
            pn_picks = getattr(classify_out, "picks", [])

            # Debug lines ----------------------------------------------------
            for p in pn_picks:
                prob_val = getattr(p, 'probability', getattr(p, 'prob', 0.0))
                debug_lines.append(f"{p.phase}  {p.peak_time}  prob={prob_val:.2f}")
        except Exception as e:
            logger.error(f"PhaseNet annotate/classify error: {e}")

    # Filter PhaseNet picks near TauP
    pn_filtered = compute_phasenet_arrivals(st, selected_event["Raw"], "CI", station)

    # Merge arrivals: PN first then TauP for remainders
    arrivals_plot = pn_filtered or taup_arrivals

    wave_fig = plot_waveform_plotly(
        st,
        phase_arrivals=arrivals_plot,
        event_time=selected_event["Time"],
        bandpass=(bp_min, bp_max),
        highpass=hp,
        ann_stream=ann_stream,
    )

    info_lines = [
        f"Traces: {len(st)}",
        f"Sampling rate: {st[0].stats.sampling_rate} Hz",
        f"Duration: {st[0].stats.npts / st[0].stats.sampling_rate:.1f} s",
        f"PhaseNet picks: {len(pn_picks)} (filtered: {len(pn_filtered)})",
    ]

    return wave_fig, "\n".join(debug_lines), "\n".join(info_lines)

# ---------------------------------------------------------------------------
# Availability helper
# ---------------------------------------------------------------------------

def filter_stations_with_waveforms(event_time: str, stations: list[str], max_checks: int = 20):
    """Return subset of stations that actually have waveform data for the event window.

    It checks up to *max_checks* stations sequentially to keep latency reasonable.
    """

    if not stations:
        return []

    scedc = SCEDCInterface()

    try:
        origin = UTCDateTime(event_time)
    except Exception:
        origin = None

    if origin is None:
        return stations  # fallback – cannot determine window

    start = origin - 30  # 30 s before
    end = origin + 150   # 2.5 min after

    available: list[str] = []
    for sta in stations[:max_checks]:
        try:
            st = scedc.get_waveform_data(start.isoformat(), end.isoformat(), "CI", sta, "BHZ")
            if st and len(st) > 0:
                available.append(sta)
        except Exception:
            continue

    # If none found in checked subset, return original list to avoid empty UI
    return available or stations

# ---------------------------------------------------------------------------
# Station coordinate helper with cache
# ---------------------------------------------------------------------------

_STA_COORD_CACHE: dict[str, tuple[float, float]] = {}

def get_station_coords(network: str, station: str) -> tuple[float | None, float | None]:
    """Return (lat, lon) for station using cached StationXML where possible."""

    key = f"{network}.{station}"
    if key in _STA_COORD_CACHE:
        return _STA_COORD_CACHE[key]

    scedc = SCEDCInterface()
    xml_txt = scedc.get_station_metadata(network, station)
    if not xml_txt:
        return None, None

    try:
        root = ET.fromstring(xml_txt)
        sta_elem = root.find('.//{*}Station')
        if sta_elem is None:
            return None, None
        lat = float(sta_elem.find('.//{*}Latitude').text)
        lon = float(sta_elem.find('.//{*}Longitude').text)
        _STA_COORD_CACHE[key] = (lat, lon)
        return lat, lon
    except Exception:
        return None, None

if __name__ == "__main__":
    logger.info("🌍 Starting SCEDC Interactive Gradio App...")
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True) 