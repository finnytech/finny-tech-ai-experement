import time
import cv2
import numpy as np
import threading
import subprocess
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from wr_tracker import WorldRecordTracker

class StreamBroadcaster:
    def __init__(self, port: int = 8080):
        self.port = port
        self.latest_jpeg = None
        self.lock = threading.Lock()
        self.running = False
        self.server = None
        self.server_thread = None
        self.public_url = None

    def update_frame(
        self,
        raw_bgr_frame: np.ndarray,
        tracker_summary: dict,
        current_lap_ms: float,
        speed: float,
        action_name: str,
        episode: int,
        reward: float
    ):
        # Create HUD overlay onto game frame
        frame = raw_bgr_frame.copy()
        h, w, _ = frame.shape

        # Scale up if needed for crisp viewing
        if h < 480:
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
            h, w, _ = frame.shape

        # Overlay Banner at top
        cv2.rectangle(frame, (0, 0), (w, 90), (15, 15, 15), -1)
        cv2.line(frame, (0, 90), (w, 90), (0, 255, 200), 2)

        # Telemetry Texts
        target_wr = tracker_summary.get("target_wr_str", "--:--:--")
        best_time = tracker_summary.get("current_best_str", "--:--:--")
        cur_lap_str = WorldRecordTracker.ms_to_time_str(current_lap_ms)
        wins = tracker_summary.get("total_wins", 0)
        points = tracker_summary.get("total_points", 0)
        wr_beaten = tracker_summary.get("wr_beaten", False)

        # Status text
        status_color = (0, 255, 0) if wr_beaten else (0, 200, 255)
        status_text = " WORLD RECORD BEATEN! " if wr_beaten else f"HUNTING WR ({target_wr})"

        cv2.putText(frame, f"FINNY TECH DEEP LABS - MM2 WR AGENT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
        cv2.putText(frame, f"STATUS: {status_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
        cv2.putText(frame, f"LAP: {cur_lap_str}  |  BEST: {best_time}  |  WR TARGET: {target_wr}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Bottom HUD
        cv2.rectangle(frame, (0, h - 45), (w, h), (15, 15, 15), -1)
        cv2.line(frame, (0, h - 45), (w, h - 45), (0, 255, 200), 1)
        cv2.putText(frame, f"Ep: {episode} | Pts: {points} | Wins: {wins} | Speed: {int(speed)} | Act: {action_name} | Rew: {reward:.1f}", 
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        with self.lock:
            self.latest_jpeg = jpeg.tobytes()

    def get_frame(self):
        with self.lock:
            return self.latest_jpeg

    def start_http_server(self):
        broadcaster = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.end_headers()
                    try:
                        while broadcaster.running:
                            frame = broadcaster.get_frame()
                            if frame is not None:
                                self.wfile.write(b"--frame\r\n")
                                self.send_header("Content-Type", "image/jpeg")
                                self.send_header("Content-Length", str(len(frame)))
                                self.end_headers()
                                self.wfile.write(frame)
                                self.wfile.write(b"\r\n")
                            time.sleep(0.033) # ~30 FPS
                    except Exception:
                        pass
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Finny Tech Deep Labs - Micro Machines 2 WR Tracker</title>
    <style>
        body {{ background-color: #0b0f19; color: #fff; font-family: monospace; text-align: center; margin: 0; padding: 20px; }}
        h1 {{ color: #00ffc8; margin-bottom: 5px; }}
        .stream-box {{ display: inline-block; border: 3px solid #00ffc8; border-radius: 8px; overflow: hidden; box-shadow: 0 0 20px rgba(0,255,200,0.3); }}
        img {{ display: block; max-width: 100%; height: auto; }}
        .badge {{ display: inline-block; padding: 6px 12px; margin: 10px; border-radius: 4px; background: #1f293d; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>🏎️ Finny Tech Deep Labs - Micro Machines 2 World Record Live Stream</h1>
    <div>
        <span class="badge" style="color: #00ffc8;">TPU v5e-1 JAX/Flax Transformer</span>
        <span class="badge" style="color: #ffaa00;">Realtime HUD & WR Telemetry</span>
    </div>
    <div class="stream-box">
        <img src="/stream.mjpg" width="640" height="480" />
    </div>
</body>
</html>"""
                    self.wfile.write(html.encode("utf-8"))

            def log_message(self, format, *args):
                return  # Suppress request logging

        self.server = HTTPServer(("0.0.0.0", self.port), Handler)
        self.running = True
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        print(f"[Streamer] Live Web Server started on port {self.port}")

    def start_cloudflare_tunnel(self):
        """Starts cloudflared in the background to get a public HTTPS link in Colab."""
        try:
            # Download cloudflared if not present (Linux/Colab)
            if not os.path.exists("./cloudflared"):
                subprocess.run(["wget", "-q", "-nc", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", "cloudflared"], check=False)
                subprocess.run(["chmod", "+x", "./cloudflared"], check=False)
            
            p = subprocess.Popen(
                ["./cloudflared", "tunnel", "--url", f"http://localhost:{self.port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # Find tunnel url in stderr
            for _ in range(30):
                line = p.stderr.readline()
                if "trycloudflare.com" in line:
                    for part in line.split():
                        if "trycloudflare.com" in part:
                            self.public_url = part.strip()
                            print(f"\n=======================================================")
                            print(f"🔥 PUBLIC LIVE STREAM LINK: {self.public_url}")
                            print(f"=======================================================\n")
                            break
                    if self.public_url:
                        break
                time.sleep(0.5)
        except Exception as e:
            print(f"[Streamer] Note: Cloudflare tunnel setup info: {e}")
