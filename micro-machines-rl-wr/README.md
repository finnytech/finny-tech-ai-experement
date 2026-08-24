# 🏎️ Finny Tech Deep Labs - Micro Machines 2 WR Agent (JAX/Flax on TPU v5e-1)

Reinforcement Learning Agent mit **Causal Transformer**, **RAM-State Telemetrie**, **Live-Stream (Cloudflare Tunnel)** und **World Record Tracker** für *Micro Machines 2: Turbo Tournament* (Sega Mega Drive / Genesis).

---

## ⚡ Schnellstart in Google Colab (TPU v5e-1)

Führe diese Zellen in Google Colab aus:

```python
# 1. Repository klonen & ins Verzeichnis wechseln
!git clone https://github.com/finnytech/finny-tech-ai-experement.git
%cd finny-tech-ai-experement/micro-machines-rl-wr

# 2. Abhängigkeiten installieren
!pip install -q jax[tpu] flax optax opencv-python gym-retro stable-retro
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared

# 3. Training & Live-Stream starten
!python3 train.py
```

---

## 📂 Struktur

- `model.py`: From-Scratch Causal Transformer Policy & Value Network (`bfloat16`, Fused Attention).
- `ppo_jax.py`: JAX/Optax PPO Training Step & GAE Berechnung.
- `boot_manager.py`: Automatischer Menü-Bypass (Super League Division 1 + Spider).
- `env_wrapper.py`: Human Frame-Skipping (4 Frames / 15 Hz) & RAM State Caching.
- `wr_tracker.py`: World Record Logger, Deltas, Siege & Punkte.
- `streamer.py`: MJPEG Web-Server mit Live-HUD & Cloudflare Public Stream Link.
- `micro_machines_wr_colab.ipynb`: Fertiges Colab-Notebook.
