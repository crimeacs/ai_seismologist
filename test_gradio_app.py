#!/usr/bin/env python3
"""
Test script for the Gradio app
"""

import pytest

from gradio_app import create_app

def test_create_app():
    """Ensure the Gradio Blocks object is created without errors."""

    app = create_app()
    assert app is not None 