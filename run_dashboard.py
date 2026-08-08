"""Runner script for the Neural Child Dashboard.

This script ensures that all required packages are installed and launches the dashboard.
"""

import importlib.util
import os
import subprocess
import sys

# Required packages for the dashboard
REQUIRED_PACKAGES = [
    "dash",
    "dash-bootstrap-components",
    "plotly",
    "pandas",
    "numpy",
    "pydantic",
]

def check_and_install_packages():
    """Check if required packages are installed and install them if needed."""
    packages_to_install = []

    for package in REQUIRED_PACKAGES:
        spec = importlib.util.find_spec(package.replace("-", "_"))
        if spec is None:
            packages_to_install.append(package)

    if packages_to_install:
        print(f"Installing required packages: {', '.join(packages_to_install)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *packages_to_install])
        print("All required packages installed successfully!")
    else:
        print("All required packages are already installed.")

def main():
    """Main function to run the dashboard."""
    # Check and install required packages
    check_and_install_packages()

    # Get the absolute path to the dashboard script
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neural_child_dashboard.py")

    # Run the dashboard
    if os.path.exists(dashboard_path):
        print("Starting Neural Child Dashboard...")
        subprocess.call([sys.executable, dashboard_path])
    else:
        dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neural-child-dashboard.py")
        if os.path.exists(dashboard_path):
            print("Starting Neural Child Dashboard...")
            subprocess.call([sys.executable, dashboard_path])
        else:
            print(f"Error: Dashboard file not found at '{dashboard_path}'")
            print("Please ensure the dashboard script is in the same directory as this runner script.")
            return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
