"""Unit tests for SCEDCInterface using mocked network calls."""

from unittest.mock import patch, MagicMock

import pytest


from gradio_app import SCEDCInterface


class DummyResponse:
    def __init__(self, status_code=200, content=b"", text="", url="http://fake"):
        self.status_code = status_code
        self.content = content
        self.text = text
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("HTTP error")


@patch("gradio_app.requests.get")
def test_get_waveform_no_content(mock_get):
    """Verify get_waveform_data handles 204 No Content gracefully."""

    mock_get.return_value = DummyResponse(status_code=204)

    iface = SCEDCInterface()
    st = iface.get_waveform_data("2020-01-01T00:00:00", "2020-01-01T00:10:00", "CI", "PAS", "BHZ")

    assert st is None


@patch("gradio_app.requests.get")
def test_get_station_metadata_ok(mock_get):
    """Provide minimal StationXML and ensure text is returned."""

    station_xml = (
        """<?xml version='1.0'?>\n"
        "<FDSNStationXML xmlns=\"http://www.fdsn.org/xml/station/1\">\n"
        "  <Network code=\"CI\">\n"
        "    <Station code=\"PAS\">\n"
        "      <Latitude>34.0</Latitude>\n"
        "      <Longitude>-118.0</Longitude>\n"
        "    </Station>\n"
        "  </Network>\n"
        "</FDSNStationXML>"""
    )

    mock_get.return_value = DummyResponse(status_code=200, text=station_xml)

    iface = SCEDCInterface()
    txt = iface.get_station_metadata("CI", "PAS")

    assert "Station" in txt 