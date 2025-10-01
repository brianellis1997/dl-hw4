#!/bin/bash

# Install PySuperTuxKart dependencies and the package
echo "Installing PySuperTuxKart dependencies..."

# Install system dependencies
apt-get update -qq
apt-get install -y -qq xvfb x11-utils > /dev/null 2>&1

# Install Python packages
pip install -q PySuperTuxKartData
pip install -q imageio imageio-ffmpeg

# Install PySuperTuxKart from the course index
pip install PySuperTuxKart --index-url=https://www.cs.utexas.edu/~bzhou/dl_class/pystk

echo "PySuperTuxKart installation complete!"