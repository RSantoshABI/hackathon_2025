"""
Quick launcher script for the Price Optimization Streamlit App
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit app"""
    
    app_path = Path(__file__).parent / "app.py"
    
    if not app_path.exists():
        print(f"❌ Error: app.py not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Launching Price Optimization App...")
    print(f"📁 App location: {app_path}")
    print("\n" + "="*60)
    print("The app will open in your default browser.")
    print("To stop the app, press Ctrl+C in this terminal.")
    print("="*60 + "\n")
    
    # Launch streamlit
    subprocess.run([
        "streamlit", "run", str(app_path),
        "--theme.base", "dark",
        "--theme.primaryColor", "#FFD700",
        "--theme.backgroundColor", "#1a1a1a",
        "--theme.secondaryBackgroundColor", "#2d2d2d",
        "--theme.textColor", "#e0e0e0"
    ])


if __name__ == "__main__":
    main()
