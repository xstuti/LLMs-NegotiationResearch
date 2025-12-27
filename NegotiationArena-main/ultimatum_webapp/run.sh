#!/bin/bash

# Ultimatum Game Visualizer Launcher Script

echo "🎮 Starting Ultimatum Game Visualizer..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed!"
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Run the Streamlit app
echo "🚀 Launching webapp..."
echo "📍 Access the app at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd "$SCRIPT_DIR"
streamlit run app.py
