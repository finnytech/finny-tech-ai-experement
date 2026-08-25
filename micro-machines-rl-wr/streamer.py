import time
import cv2
import numpy as np
import threading
import subprocess
import os
import socket
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from wr_tracker import WorldRecordTracker

class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True

def find_available_port(start_port: int = 8080, max_attempts: int = 25) -> int:
    """Finds an available TCP port dynamically."""
    for p in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return start_port

class StreamBroadcaster:
    def __init__(self, port: int = 8080):
        self.port = find_available_port(port)
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
        if raw_bgr_frame is None:
            return

        frame = raw_bgr_frame.copy()
        h, w = frame.shape[:2]

        if h < 480:
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
            h, w = frame.shape[:2]

        # Top Banner
        cv2.rectangle(frame, (0, 0), (w, 90), (15, 15, 15), -1)
        cv2.line(frame, (0, 90), (w, 90), (0, 255, 200), 2)

        target_wr = tracker_summary.get("target_wr_str", "00:42.50")
        best_time = tracker_summary.get("current_best_str", "--:--:--")
        cur_lap_str = WorldRecordTracker.ms_to_time_str(current_lap_ms)
        wins = tracker_summary.get("total_wins", 0)
        points = tracker_summary.get("total_points", 0)
        wr_beaten = tracker_summary.get("wr_beaten", False)

        status_color = (0, 255, 0) if wr_beaten else (0, 200, 255)
        status_text = " WORLD RECORD BEATEN! " if wr_beaten else f"HUNTING WR ({target_wr})"

        cv2.putText(frame, "FINNY TECH DEEP LABS - MM2 WR AGENT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
        cv2.putText(frame, f"STATUS: {status_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
        cv2.putText(frame, f"LAP: {cur_lap_str}  |  BEST: {best_time}  |  WR TARGET: {target_wr}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Bottom Banner
        cv2.rectangle(frame, (0, h - 45), (w, h), (15, 15, 15), -1)
        cv2.line(frame, (0, h - 45), (w, h - 45), (0, 255, 200), 1)
        cv2.putText(frame, f"Ep: {episode} | Pts: {points} | Wins: {wins} | Speed: {int(speed)} | Act: {action_name} | Rew: {reward:.1f}", 
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

        success, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if success:
            with self.lock:
                self.latest_jpeg = jpeg.tobytes()

    def get_frame(self):
        with self.lock:
            return self.latest_jpeg

    def start_http_server(self):
        broadcaster = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                try:
                    if self.path == "/stream.mjpg":
                        self.send_response(200)
                        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                        self.send_header("Cache-Control", "no-cache, private")
                        self.send_header("Pragma", "no-cache")
                        self.end_headers()
                        while broadcaster.running:
                            frame = broadcaster.get_frame()
                            if frame is not None:
                                self.wfile.write(b"--frame\r\n")
                                self.send_header("Content-Type", "image/jpeg")
                                self.send_header("Content-Length", str(len(frame)))
                                self.end_headers()
                                self.wfile.write(frame)
                                self.wfile.write(b"\r\n")
                            time.sleep(0.033)
                    else:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/html; charset=utf-8")
                        self.end_headers()
                        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Finny Tech Deep Labs - Micro Machines 2 WR Tracker</title>
    <style>
        body { background-color: #0b0f19; color: #fff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace; text-align: center; margin: 0; padding: 20px; }
        h1 { color: #00ffc8; margin-bottom: 5px; }
        .stream-box { display: inline-block; border: 3px solid #00ffc8; border-radius: 8px; overflow: hidden; box-shadow: 0 0 20px rgba(0,255,200,0.3); background: #000; }
        img { display: block; max-width: 100%; height: auto; }
        .badge { display: inline-block; padding: 6px 12px; margin: 10px; border-radius: 4px; background: #1f293d; font-weight: bold; }
    </style>
</head>
<body>
    <h1>🏎️ Finny Tech Deep Labs - Micro Machines 2 World Record Live Stream</h1>
    <div>
        <span class="badge" style="color: #00ffc8;">TPU v5e-1 JAX/Flax Transformer</span>
        <span class="badge" style="color: #ffaa00;">Super League - Division 1 (Spider)</span>
    </div>
    <br/>
    <div class="stream-box">
        <img src="/stream.mjpg" width="640" height="480" />
    </div>
</body>
</html>"""
                        self.wfile.write(html.encode("utf-8"))
                except Exception:
                    pass

            def log_message(self, format, *args):
                return

        try:
            self.server = ReusableHTTPServer(("0.0.0.0", self.port), Handler)
            self.running = True
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            print(f"[Streamer] Live Web Server started on port {self.port}")
        except Exception as e:
            print(f"[Streamer] Server start warning: {e}")

    def start_cloudflare_tunnel(self):
        try:
            # 1. Native Colab Port Forwarding Button
            try:
                from google.colab import output
                output.serve_kernel_port_as_window(self.port, anchor_text=f"📺 Öffne Live-Stream über Google Colab Port {self.port}")
            except Exception:
                pass

            subprocess.run(["pkill", "-9", "-f", "cloudflared"], check=False)
            time.sleep(0.5)

            bin_path = "./cloudflared"
            if not os.path.exists(bin_path):
                bin_path = "/content/cloudflared"
            if not os.path.exists(bin_path):
                subprocess.run(["wget", "-q", "-nc", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", "cloudflared"], check=False)
                subprocess.run(["chmod", "+x", "cloudflared"], check=False)
                bin_path = "./cloudflared"
            
            p = subprocess.Popen(
                [bin_path, "tunnel", "--url", f"http://127.0.0.1:{self.port}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            for _ in range(60):
                if p.stderr is None:
                    break
                line = p.stderr.readline()
                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
                if match:
                    self.public_url = match.group(0)
                    time.sleep(2.0) # DNS Propagation Puffer
                    print("\n" + "=" * 65)
                    print(f"🔥 DEIN LIVE-STREAM LINK:")
                    print(f"👉 {self.public_url}")
                    print("=" * 65 + "\n")

                    try:
                        from IPython.display import display, HTML
                        display(HTML(f'''
                        <div style="padding: 15px; background: #0b0f19; border: 2px solid #00ffc8; border-radius: 8px; margin: 15px 0;">
                            <h3 style="color: #00ffc8; margin: 0 0 10px 0;">🏎️ Finny Tech Deep Labs - Live Stream Bereit!</h3>
                            <a href="{self.public_url}" target="_blank" style="display: inline-block; padding: 10px 20px; background: #00ffc8; color: #000; font-weight: bold; text-decoration: none; border-radius: 5px; font-size: 16px;">
                                📺 HIER KLICKEN: LIVE STREAM IM BROWSER ÖFFNEN
                            </a>
                            <p style="color: #aaa; font-size: 12px; margin-top: 8px;">(Falls der Link 1-2 Sekunden braucht, kurz aktualisieren - DNS propagiert weltweit!)</p>
                        </div>
                        '''))
                    except Exception:
                        pass
                    break
                time.sleep(0.3)
        except Exception as e:
            print(f"[Streamer] Cloudflare tunnel notice: {e}")
