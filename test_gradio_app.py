#!/usr/bin/env python3
"""
Test script for the Gradio app
"""

import sys
import time
import requests
from gradio_app import create_app

def test_gradio_app():
    """Test that the Gradio app can be created and launched"""
    
    print("🧪 Testing Gradio App...")
    
    try:
        # Test 1: Create the app
        print("1. Creating Gradio app...")
        app = create_app()
        print("   ✓ App created successfully")
        
        # Test 2: Launch the app
        print("2. Launching app...")
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=True,
            inbrowser=False
        )
        
        # Wait a moment for the server to start
        time.sleep(5)
        
        # Test 3: Check if the server is responding
        print("3. Testing server response...")
        try:
            response = requests.get("http://127.0.0.1:7860", timeout=10)
            if response.status_code == 200:
                print("   ✓ Server is responding")
                print("   ✓ Gradio app is working!")
                return True
            else:
                print(f"   ✗ Server returned status code: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"   ✗ Could not connect to server: {e}")
            return False
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_gradio_app()
    if success:
        print("\n🎉 All tests passed! The Gradio app is working correctly.")
        print("You can now run: python gradio_app.py")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
        sys.exit(1) 