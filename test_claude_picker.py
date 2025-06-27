import numpy as np
from obspy import Stream, Trace
from unittest.mock import MagicMock, patch

from claude_picker import stream_to_ascii, compute_error, call_claude


def _make_stream():
    data = np.arange(10, dtype=float)
    tr = Trace(data=data)
    tr.stats.network = "CI"
    tr.stats.station = "ABC"
    tr.stats.channel = "BHZ"
    tr.stats.sampling_rate = 1.0
    return Stream([tr])


def test_stream_to_ascii():
    st = _make_stream()
    txt = stream_to_ascii(st)
    assert "CI.ABC.BHZ" in txt
    assert len(txt.split()) > 10


def test_compute_error():
    phasenet = [
        {"phase": "P", "time": 10.0},
        {"phase": "S", "time": 20.0},
    ]
    claude = {"P": 11.0, "S": 18.0}
    err = compute_error(claude, phasenet)
    assert err["P"] == 1.0
    assert err["S"] == 2.0


def test_call_claude():
    st = _make_stream()
    ascii_str = stream_to_ascii(st)
    dummy_client = MagicMock()
    dummy_msg = MagicMock()
    dummy_msg.content = [MagicMock(text='{"P": 12.0, "S": 22.0}')]
    dummy_client.messages.create.return_value = dummy_msg

    with patch("claude_picker.anthropic") as mock_module:
        mock_module.Anthropic.return_value = dummy_client
        out = call_claude(ascii_str, api_key="sk-xxx")
    assert out == {"P": 12.0, "S": 22.0}
