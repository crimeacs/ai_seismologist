# SCEDC Seismic Data Explorer

Interactive web application for browsing and analysing Southern California Earthquake Data Center (SCEDC) events.

Key features
------------
* 🌐  Gradio front-end for an instant in-browser experience.
* 🗺️  Event & station visualisation on Mapbox and timeline plots.
* 🔄  60-second three-component waveform download per station.
* ⚡️  Real-time phase picking:
  * 𝐓𝐚𝐮𝐏 – theoretical P & S arrivals (IASP91).
  * **PhaseNet** (SeisBench) – machine-learning picks.
  * **Claude "AI Seismologist"** – Gen-AI picks via Anthropic API.
* 📊  Plot overlays with source-specific styling (TauP dashed, PhaseNet solid, Claude dotted).
* 🪄  Debug panel with raw inputs/outputs for PhaseNet & Claude.

Quick start
-----------
```bash
# Create and activate a virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies → requirements.txt
pip install -r requirements.txt

# Add your Anthropic key (or put it in a .env file – see below)
export ANTHROPIC_API_KEY="sk-ant-…"

# Launch the app
python gradio_app.py
```
The interface will be available at http://localhost:7860 by default.

Environment variables (.env)
---------------------------
The app automatically loads a `.env` file in the project root at start-up.
Minimum content:
```dotenv
# .env
ANTHROPIC_API_KEY=sk-ant-…
```
You can still override the key at runtime via the sidebar "🤖 Claude Settings".

Repository layout
-----------------
```
│ README.md            ← this file
│ requirements.txt     ← Python dependencies
│ gradio_app.py        ← Main application
│ claude_picker.py     ← Claude integration helper
│ run_gradio_app.py    ← thin wrapper (optional)
└── tests/             ← pytest unit tests (TBD)
```

Development & contributing
-------------------------
1. Follow *Quick start* above using a virtualenv.
2. Run `python -m pytest` to execute tests (add more in `tests/`).
3. Ensure `ruff` / `black` are happy:
   ```bash
   ruff check .
   black --check .
   ```
4. Create a feature branch, commit & open a pull request.

House-keeping
-------------
* All waveform processing is limited to a 60 s window (−10 s → +50 s around the event origin).
* The Claude system prompt lives in `claude_picker.py::SYSTEM_PROMPT`; edit via the UI if required.
* Debug logs (INFO level) are verbose by design; pipe output when running in CI. 