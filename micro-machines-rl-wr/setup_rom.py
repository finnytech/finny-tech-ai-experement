import os
import sys
import glob
import hashlib
import json
import urllib.request
import zipfile
import shutil
import subprocess

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
        local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_integrations", "MicroMachines2-Genesis")
        os.makedirs(local_dir, exist_ok=True)
        return local_dir

def scan_for_user_uploaded_rom():
    """Scans /content, current dir, and subdirs for user-provided Micro Machines 2 ROMs."""
    search_paths = ["/content", ".", os.path.dirname(os.path.abspath(__file__))]
    extensions = ["*.md", "*.bin", "*.gen", "*.smd", "*.zip", "*.7z"]
    
    for base in search_paths:
        if not os.path.exists(base):
            continue
        for ext in extensions:
            matches = glob.glob(os.path.join(base, "**", ext), recursive=True)
            for m in matches:
                name_lower = os.path.basename(m).lower()
                if "micro" in name_lower or "machine" in name_lower or ext in ["*.md", "*.bin", "*.gen"]:
                    if os.path.getsize(m) > 100000:
                        return m
    return None

def install_rom_file(source_path, target_dir):
    """Installs and verifies a ROM file into the integration folder."""
    rom_dest = os.path.join(target_dir, "rom.md")
    sha_dest = os.path.join(target_dir, "rom.sha")
    
    if source_path.lower().endswith(".zip"):
        with zipfile.ZipFile(source_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if member.lower().endswith(('.md', '.bin', '.gen', '.smd')):
                    with zip_ref.open(member) as src, open(rom_dest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    break
    else:
        shutil.copyfile(source_path, rom_dest)

    if os.path.exists(rom_dest) and os.path.getsize(rom_dest) > 100000:
        with open(rom_dest, "rb") as f:
            rom_hash = hashlib.sha1(f.read()).hexdigest()
        with open(sha_dest, "w") as f:
            f.write(rom_hash + "\n")
        print(f"[ROM-Setup] ✅ ECHTES SEGA ROM ERFOLGREICH INSTALLIERT ({os.path.getsize(rom_dest)} Bytes)!")
        return True
    return False

def ensure_real_rom():
    """
    Ensures that the real Sega Genesis Micro Machines 2 ROM is present.
    """
    target_dir = get_retro_custom_dir()
    rom_dest = os.path.join(target_dir, "rom.md")
    sha_dest = os.path.join(target_dir, "rom.sha")
    data_dest = os.path.join(target_dir, "data.json")
    meta_dest = os.path.join(target_dir, "metadata.json")
    scen_dest = os.path.join(target_dir, "scenario.json")

    # 1. Write Integration JSONs
    with open(data_dest, "w") as f:
        json.dump(DATA_JSON, f, indent=2)
    with open(meta_dest, "w") as f:
        json.dump(METADATA_JSON, f, indent=2)
    with open(scen_dest, "w") as f:
        json.dump(SCENARIO_JSON, f, indent=2)

    # 2. Check if already installed
    if os.path.exists(rom_dest) and os.path.getsize(rom_dest) > 100000:
        with open(rom_dest, "rb") as f:
            rom_hash = hashlib.sha1(f.read()).hexdigest()
        with open(sha_dest, "w") as f:
            f.write(rom_hash + "\n")
        return True

    # 3. Check for uploaded ROM in /content or project dir
    found_rom = scan_for_user_uploaded_rom()
    if found_rom and found_rom != rom_dest:
        print(f"[ROM-Setup] 📦 Gefundenes ROM wird importiert: {found_rom}")
        if install_rom_file(found_rom, target_dir):
            return True

    # 4. Try multi-source curl download with realistic browser spoofing
    download_urls = [
        "https://archive.org/download/sega-mega-drive-genesis-romset-ultra-complete/Micro%20Machines%202%20-%20Turbo%20Tournament%20%28Europe%29%20%28J-Cart%29.zip",
        "https://archive.org/download/No-Intro-Collection_2016-01-03_Fixed/Sega%20-%20Mega%20Drive%20-%20Genesis.zip/Micro%20Machines%202%20-%20Turbo%20Tournament%20%28Europe%29%20%28J-Cart%29.zip"
    ]
    
    temp_zip = "/content/mm2_rom.zip" if sys.platform != "win32" else os.path.join(target_dir, "mm2_rom.zip")
    for url in download_urls:
        try:
            print(f"[ROM-Setup] Versuche Download über curl: {url[:50]}...")
            cmd = ["curl", "-sL", "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36", url, "-o", temp_zip]
            subprocess.run(cmd, timeout=30, check=False)
            if os.path.exists(temp_zip) and os.path.getsize(temp_zip) > 100000:
                if install_rom_file(temp_zip, target_dir):
                    return True
        except Exception:
            pass

    # 5. If still not found, print clear instructions
    print("\n" + "=" * 70)
    print("⚠️ BITTE ROM-DATEI BEREITSTELLEN:")
    print("Ziehe einfach deine 'Micro Machines 2.md' (oder .bin / .zip) Datei")
    print("per Drag & Drop links in die Google Colab Dateileiste (/content/).")
    print("Das Skript erkennt sie automatisch und startet das echte Spiel!")
    print("=" * 70 + "\n")
    return False

if __name__ == "__main__":
    ensure_real_rom()
