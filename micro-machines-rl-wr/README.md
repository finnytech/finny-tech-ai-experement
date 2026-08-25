# 🏎️ Finny Tech Deep Labs - Micro Machines 2 WR Agent (JAX/Flax on TPU v5e-1)

Reinforcement Learning Agent mit **Causal Transformer**, **RAM-State Telemetrie**, **Live-Stream (Cloudflare Tunnel)** und **World Record Tracker** für *Micro Machines 2: Turbo Tournament* (Sega Mega Drive / Genesis).

---

## ⚡ Schnellstart in Google Colab (All-in-One Zelle)

Kopiere diesen gesamten Block in eine einzige Zelle in Google Colab:

```python
# 1. Repository herunterladen & nach /content kopieren
!rm -rf /content/finny-tech-ai-experement
!git clone https://github.com/finnytech/finny-tech-ai-experement.git /content/finny-tech-ai-experement
!cp -rf /content/finny-tech-ai-experement/micro-machines-rl-wr/* /content/
%cd /content

# 2. Systempakete & RL-Abhängigkeiten installieren
!apt-get update -qq && apt-get install -y -qq cmake libgl1-mesa-glx libglib2.0-0 zlib1g-dev
!pip install -q flax optax opencv-python stable-retro
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /content/cloudflared
!chmod +x /content/cloudflared

# 3. Training & Live-Stream starten
!python3 /content/train.py
```

---

## 📂 Struktur

- `model.py`: From-Scratch Causal Transformer Policy & Value Network (`bfloat16` & `float32` universal support).
- `ppo_jax.py`: JAX/Optax PPO Training Step & GAE Berechnung.
- `boot_manager.py`: Automatischer Menü-Bypass (Super League Division 1 + Spider).
- `env_wrapper.py`: Human Frame-Skipping (4 Frames / 15 Hz) & RAM State Caching.
- `wr_tracker.py`: World Record Logger, Deltas, Siege & Punkte.
- `streamer.py`: MJPEG Web-Server mit Live-HUD & Cloudflare Public Stream Link.
- `micro_machines_wr_colab.ipynb`: Fertiges Colab-Notebook.
