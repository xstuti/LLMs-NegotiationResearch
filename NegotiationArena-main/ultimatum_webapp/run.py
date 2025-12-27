#!/usr/bin/env python3
"""
Ultimatum Game Visualizer Launcher Script
Cross-platform launcher for the Streamlit webapp
"""

import subprocess
import sys
from pathlib import Path


def check_streamlit():
    """Check if streamlit is installed"""
    try:
        import streamlit

        return True
    except ImportError:
        return False


def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    requirements_file = Path(__file__).parent / "requirements.txt"

    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
        )
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def main():
    """Main launcher function"""
    print("🎮 Starting Ultimatum Game Visualizer...")
    print()

    # Check if streamlit is installed
    if not check_streamlit():
        print("❌ Streamlit is not installed!")
        if not install_dependencies():
            sys.exit(1)

    # Get the directory of this script
    script_dir = Path(__file__).parent
    app_file = script_dir / "app.py"

    if not app_file.exists():
        print(f"❌ Could not find app.py at {app_file}")
        sys.exit(1)

    # Run the Streamlit app
    print("🚀 Launching webapp...")
    print("📍 Access the app at: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the server")
    print()

    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_file)])
    except KeyboardInterrupt:
        print()
        print("👋 Shutting down...")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
