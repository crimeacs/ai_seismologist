# SCEDC Interactive Seismic Explorer

A Gradio web application for searching Southern California earthquake events, visualising them on an interactive map/timeline, and fetching station waveforms with basic filters.

![screenshot](docs/screenshot.png)

---

## Features

* Search SCEDC event catalogue by date and magnitude.
* Interactive Plotly map & timeline visualisation of events.
* Dropdown-based event & station selection (Gradio-friendly).
* Waveform download via FDSN web service (ObsPy backend).
* Basic preprocessing & filter controls (band-pass, high-pass).
* Phase arrival markers (simple estimates, placeholder for future picks).

---

## Quickstart

```bash
# Clone & enter repository
$ git clone https://github.com/your-org/scedc-explorer.git
$ cd scedc-explorer

# Create virtual env (optional)
$ python -m venv .venv && source .venv/bin/activate

# Install deps
$ pip install -r requirements.txt

# Run the app
$ python run_gradio_app.py
```

Open http://localhost:7860 in your browser.

---

## Project layout

```
.
├── gradio_app.py        # core Blocks app
├── run_gradio_app.py    # launcher script w/ dep checks
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Contributing

1. Create a feature branch: `git checkout -b feature/awesome`  
2. Commit your changes.  
3. Push and open a PR.

---

## License

MIT 