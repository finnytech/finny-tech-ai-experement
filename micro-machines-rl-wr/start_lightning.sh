#!/usr/bin/env bash
set -e

echo "============================================================"
echo "⚡ FINNY TECH DEEP LABS - LIGHTNING.AI SPEEDRUN RUNNER"
echo "============================================================"

# 1. System packages
sudo apt-get update -qq && sudo apt-get install -y -qq cmake libgl1-mesa-glx libglib2.0-0 zlib1g-dev

# 2. Python dependencies (JAX GPU/CPU automatic support)
pip install -q flax optax opencv-python stable-retro

# 3. Cloudflared for instant Live-Stream
if [ ! -f "cloudflared" ]; then
    wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    chmod +x cloudflared
fi

# 4. Start genuine Sega Mega Drive Training & Live Stream
python train.py
