"""Interact with the Claude API to pick seismic phases.

This module converts ObsPy streams to ASCII, sends them to the Claude
API with a professional seismologist system prompt, and computes error
between Claude's picks and PhaseNet picks.
"""
from __future__ import annotations

from typing import Dict, List
import json

from obspy import Stream

try:  # pragma: no cover - optional dependency
    import anthropic
except Exception:  # pragma: no cover - missing dependency at runtime
    anthropic = None


SYSTEM_PROMPT = """
You are “AI Seismologist”, a deterministic specialist that identifies
P- and S-wave arrival times on a three-component waveform.

##### ❶  INPUT FORMAT
- The user provides plain ASCII text for each component:
  * a header line `# <net>.<sta>.<chan> sr=<sampling_rate>`
  * a second line with space separated float samples.
  * Only the first 60 s of each trace are shown.

##### ❷  TASK
Return the most probable P- and S-wave arrival times (seconds after trace
start) **or** `"no_pick"` when a phase is absent.

##### ❸  REQUIRED OUTPUT
A JSON object with the exact keys:
{
  "P_time": 12.34,
  "S_time": 22.71,
  "P_prob": 0.87,
  "S_prob": 0.92,
  "qc_flag": "good"
}

##### ❹  METHOD CONSTRAINTS
1. Detrend, demean and band-pass 1–20 Hz (four-pole Butterworth,
   zero-phase).
2. Normalize each component by its median absolute deviation.
3. Use a PhaseNet-style U-Net to estimate phase probabilities and smooth
   them with a 0.5 s Gaussian window.
4. Pick the first local maximum ≥ 0.5 for P; the first maximum ≥ `P_time`
   for S. If no maximum exceeds the threshold, return `"no_pick"` and
   probability `0.0`.
5. Compute SNR around each pick (0.5 s window) and set `qc_flag` to
   `"good"` if SNR ≥ 5 and prob ≥ 0.6, `"uncertain"` if 3 ≤ SNR < 5 or
   0.5 ≤ prob < 0.6, otherwise `"bad"`.

##### ❺  RESPONSE STYLE
Output **only** the JSON object. Do not include explanations or
additional text.
"""


def stream_to_ascii(
    st: Stream, max_points: int = 1000, max_seconds: float = 60.0
) -> str:
    """Return a simple ASCII representation of the traces.

    At most the first ``max_seconds`` of each trace are included.  The data
    may be downsampled so that no more than ``max_points`` values appear per
    trace to keep the prompt small.
    """
    lines: List[str] = []
    for tr in st:
        data = tr.data
        max_samples = int(tr.stats.sampling_rate * max_seconds)
        data = data[:max_samples]
        if len(data) > max_points:
            step = max(1, len(data) // max_points)
            data = data[::step]
        header = f"# {tr.stats.network}.{tr.stats.station}.{tr.stats.channel} "
        header += f"sr={tr.stats.sampling_rate}"
        lines.append(header)
        lines.append(" ".join(f"{x:.2f}" for x in data))
    return "\n".join(lines)


def call_claude(waveform_ascii: str, api_key: str) -> Dict[str, float]:
    """Send the waveform to Claude and parse the JSON response."""
    if anthropic is None:
        raise RuntimeError("anthropic package not available")

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": waveform_ascii}],
    )
    text = msg.content[0].text if msg.content else ""
    return json.loads(text)


def compute_error(
    claude_picks: Dict[str, float], phasenet_picks: List[Dict]
) -> Dict[str, float]:
    """Return absolute difference in seconds between Claude and PhaseNet picks."""
    errors: Dict[str, float] = {}
    for p in phasenet_picks:
        ph = p.get("phase", "").upper()
        if ph in ("P", "S") and ph in claude_picks:
            errors[ph] = abs(claude_picks[ph] - float(p.get("time", 0)))
    return errors


__all__ = [
    "stream_to_ascii",
    "call_claude",
    "compute_error",
]
