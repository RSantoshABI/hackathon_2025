"""
Run script for the Price Optimization Streamlit App
"""

import subprocess
import sys
import os

def run_app():
    """Run the Streamlit application"""
    
    print("🚀 Starting Price Optimization Web App...")
    print("📍 App will be available at: http://localhost:8501")
    print("💡 Press Ctrl+C to stop the application")
    print("\n" + "="*50 + "\n")
    
    # Change to the optimization directory
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "optimizer_app.py",
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error running application: {str(e)}")
        print("\n💡 Make sure Streamlit is installed:")
        print("   pip install streamlit plotly openpyxl")

if __name__ == "__main__":
    run_app()