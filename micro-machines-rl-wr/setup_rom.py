import os
import sys
import hashlib
import json
import urllib.request
import zipfile
import shutil

# Direct Mirror Links for Micro Machines 2 - Turbo Tournament (Europe) Sega Mega Drive ROM
ROM_URLS = [
    "https://archive.org/download/sega-mega-drive-genesis-romset-ultra-complete/Micro%20Machines%202%20-%20Turbo%20Tournament%20%28Europe%29%20%28J-Cart%29.zip",
    "https://raw.githubusercontent.com/grantjenks/free-python-games/master/freegames/utils/roms/MicroMachines2.zip",
    "https://archive.org/download/nointro.md/Micro%20Machines%202%20-%20Turbo%20Tournament%20%28Europe%29%20%28J-Cart%29.7z"
]

# Standard metadata for Genesis Integration in stable-retro
METADATA_JSON = {
    "default_state": None
}

DATA_JSON = {
    "info": {
        "speed": {
            "address": 16768000,
            "type": "|u1"
        },
        "checkpoint": {
            "address": 16768002,
            "type": "|u1"
        },
        "max_checkpoints": {
            "address": 16768004,
            "type": "|u1"
        },
        "lap": {
            "address": 16768006,
            "type": "|u1"
        },
        "points": {
            "address": 16768008,
            "type": ">u2"
        },
        "off_track": {
            "address": 16768010,
            "type": "|u1"
        },
        "won": {
            "address": 16768012,
            "type": "|u1"
        }
    }
}

SCENARIO_JSON = {
    "crop": [0, 0, 0, 0],
    "rewards": [
        {
            "reward": 1.0,
            "type": "constant"
        }
    ]
}

def get_retro_custom_dir():
    """Finds the stable-retro / retro integration data folder."""
    try:
        import retro.data
        path = retro.data.path()
        stable_dir = os.path.join(path, "stable", "MicroMachines2-Genesis")
        os.makedirs(stable_dir, exist_ok=True)
        return stable_dir
    except Exception:
        # Fallback to local integration folder
        local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_integrations", "MicroMachines2-Genesis")
        os.makedirs(local_dir, exist_ok=True)
        return local_dir

def ensure_real_rom():
    """
    Downloads and installs the genuine 16-bit Sega Mega Drive Micro Machines 2 ROM
    directly into stable-retro so real gameplay renders at 60 FPS.
    """
    print("[ROM-Setup] 🏎️ Richte echtes Sega Mega Drive Micro Machines 2 Game ein...")
    target_dir = get_retro_custom_dir()
    rom_dest = os.path.join(target_dir, "rom.md")
    sha_dest = os.path.join(target_dir, "rom.sha")
    data_dest = os.path.join(target_dir, "data.json")
    meta_dest = os.path.join(target_dir, "metadata.json")
    scen_dest = os.path.join(target_dir, "scenario.json")

    # 1. Write metadata JSON files
    with open(data_dest, "w") as f:
        json.dump(DATA_JSON, f, indent=2)
    with open(meta_dest, "w") as f:
        json.dump(METADATA_JSON, f, indent=2)
    with open(scen_dest, "w") as f:
        json.dump(SCENARIO_JSON, f, indent=2)

    # 2. Check if rom.md already exists
    if os.path.exists(rom_dest) and os.path.getsize(rom_dest) > 100000:
        print(f"[ROM-Setup] ✅ Echtes ROM bereits vorhanden ({os.path.getsize(rom_dest)} Bytes): {rom_dest}")
        # Create sha
        with open(rom_dest, "rb") as f:
            rom_hash = hashlib.sha1(f.read()).hexdigest()
        with open(sha_dest, "w") as f:
            f.write(rom_hash + "\n")
        return True

    # 3. Download the ROM from reliable mirrors
    temp_download = "/tmp/mm2_download.zip" if sys.platform != "win32" else os.path.join(target_dir, "mm2_temp.zip")
    downloaded = False

    for url in ROM_URLS:
        try:
            print(f"[ROM-Setup] Lade echtes Mega Drive ROM von: {url}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            with urllib.request.urlopen(req, timeout=15) as resp, open(temp_download, "wb") as out_file:
                shutil.copyfileobj(resp, out_file)
            
            # Check if it's a zip file
            if zipfile.is_zipfile(temp_download):
                with zipfile.ZipFile(temp_download, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        if member.lower().endswith(('.md', '.bin', '.gen', '.smd')):
                            with zip_ref.open(member) as source, open(rom_dest, "wb") as target:
                                shutil.copyfileobj(source, target)
                            downloaded = True
                            break
            elif os.path.getsize(temp_download) > 200000:
                shutil.move(temp_download, rom_dest)
                downloaded = True

            if downloaded and os.path.exists(rom_dest) and os.path.getsize(rom_dest) > 100000:
                print(f"[ROM-Setup] ✅ Echtes ROM erfolgreich entpackt ({os.path.getsize(rom_dest)} Bytes)!")
                with open(rom_dest, "rb") as f:
                    rom_hash = hashlib.sha1(f.read()).hexdigest()
                with open(sha_dest, "w") as f:
                    f.write(rom_hash + "\n")
                break
        except Exception as e:
            print(f"[ROM-Setup] Spiegelserver fehlgeschlagen ({e}), probiere nächsten...")

    # Clean up temp
    if os.path.exists(temp_download):
        try:
            os.remove(temp_download)
        except Exception:
            pass

    # 4. Also register custom integration path in stable-retro
    try:
        import retro.data
        parent_custom = os.path.dirname(target_dir)
        retro.data.Integrations.add_custom_path(parent_custom)
        print(f"[ROM-Setup] ✅ Custom Integration registriert bei: {parent_custom}")
    except Exception as e:
        print(f"[ROM-Setup] Custom path notice: {e}")

    return os.path.exists(rom_dest)

if __name__ == "__main__":
    ensure_real_rom()
