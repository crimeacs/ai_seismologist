#!/usr/bin/env python3
"""
Launcher script for SCEDC Gradio App
Checks dependencies and launches the web interface
"""

import sys
import subprocess
import importlib
import logging

# Set up logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required packages are installed"""
    logger.info("🔍 Checking dependencies...")
    
    required_packages = [
        'gradio',
        'obspy', 
        'requests',
        'matplotlib',
        'numpy',
        'pandas'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            logger.info(f"   ✓ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"   ✗ {package} is missing")
    
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Installing missing packages...")
        
        try:
            logger.info(f"Running: pip install {' '.join(missing_packages)}")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                *missing_packages
            ])
            logger.info("✓ All packages installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to install packages: {e}")
            logger.error("Please install manually:")
            logger.error(f"pip install {' '.join(missing_packages)}")
            return False
    
    logger.info("✅ All dependencies are available")
    return True

def main():
    """Main launcher function"""
    logger.info("🌍 SCEDC Data Access - Gradio App Launcher")
    logger.info("=" * 60)
    
    # Check dependencies
    logger.info("Step 1: Checking dependencies...")
    if not check_dependencies():
        logger.error("❌ Dependency check failed. Please install missing dependencies and try again.")
        sys.exit(1)
    
    logger.info("Step 2: Importing Gradio app...")
    try:
        from gradio_app import create_app
        logger.info("   ✓ Successfully imported gradio_app")
    except ImportError as e:
        logger.error(f"   ✗ Failed to import gradio_app: {e}")
        sys.exit(1)
    
    logger.info("Step 3: Creating app...")
    try:
        app = create_app()
        logger.info("   ✓ App created successfully")
    except Exception as e:
        logger.error(f"   ✗ Failed to create app: {e}")
        sys.exit(1)
    
    logger.info("Step 4: Launching server...")
    logger.info("🚀 Launching Gradio app...")
    logger.info("   The app will open in your web browser.")
    logger.info("   If it doesn't open automatically, go to: http://localhost:7860")
    logger.info("   Press Ctrl+C to stop the server.")
    logger.info("-" * 60)
    
    try:
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        logger.info("\n👋 App stopped by user.")
    except Exception as e:
        logger.error(f"\n❌ Error launching app: {e}")
        logger.error("Please check that all dependencies are installed correctly.")
        sys.exit(1)

if __name__ == "__main__":
    main() 