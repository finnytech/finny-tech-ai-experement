#!/bin/bash
# === Finny VM Auto-Setup & Scraper Launcher ===

echo "=============================================="
echo "🚀 Starte System-Update und Installation..."
echo "=============================================="
sudo apt-get update
sudo apt-get install -y git python3 python3-pip python3-venv

echo "=============================================="
echo "📦 Richte Python Virtual Environment (venv) ein..."
echo "=============================================="
python3 -m venv venv
source venv/bin/activate

echo "=============================================="
echo "⚡ Installiere Scraper-Bibliotheken (pip)..."
echo "=============================================="
pip install --upgrade pip
pip install aiohttp beautifulsoup4

echo "=============================================="
echo "📁 Erstelle Daten-Ordner..."
echo "=============================================="
mkdir -p data

echo "=============================================="
echo "🔥 Starte Scraper V2 (8 Cores, Concurrency: 60)..."
echo "=============================================="
python webscraper/scraper_v2.py --limit 250000 --concurrency 60 --delay 0.5 --output data/knowledge_data.jsonl
