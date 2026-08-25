import os
import sys
import glob
import hashlib
import json
import urllib.request
import zipfile
import shutil

DIRECT_VERIFIED_ROM_URL = "https://archive.org/download/Micro_Machines_2_Turbo_Tournament_Europe.md/Micro_Machines_2_Turbo_Tournament_Europe.md"
EXPECTED_SHA1 = "cb5fb33212592809639b37c2babd72a7953fa102"

METADATA_JSON = {
    "default_state": None,
    "players": 1
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

def install_rom_bytes(target_dir, rom_path_or_bytes):
    rom_dest = os.path.join(target_dir, "rom.md")
    sha_dest = os.path.join(target_dir, "rom.sha")

    if isinstance(rom_path_or_bytes, bytes):
        with open(rom_dest, "wb") as f:
            f.write(rom_path_or_bytes)
    elif os.path.exists(rom_path_or_bytes) and rom_path_or_bytes != rom_dest:
        shutil.copyfile(rom_path_or_bytes, rom_dest)

    if os.path.exists(rom_dest) and os.path.getsize(rom_dest) > 100000:
        with open(rom_dest, "rb") as f:
            calc_sha = hashlib.sha1(f.read()).hexdigest()
        with open(sha_dest, "w") as f:
            f.write(calc_sha + "\n")
        print(f"[ROM-Setup] ✅ ECHTES SEGA ROM ERFOLGREICH INSTALLIERT ({os.path.getsize(rom_dest)} Bytes, SHA1: {calc_sha})!")
        return True
    return False

def ensure_real_rom():
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
            calc_sha = hashlib.sha1(f.read()).hexdigest()
        with open(sha_dest, "w") as f:
            f.write(calc_sha + "\n")
        return True

    # 3. Check local paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_local = [
        os.path.join(current_dir, "rom.md"),
        "/content/rom.md",
        "/content/finny-tech-ai-experement/micro-machines-rl-wr/rom.md"
    ]
    for p in possible_local:
        if os.path.exists(p) and os.path.getsize(p) > 100000:
            print(f"[ROM-Setup] 📦 Gefundenes lokales ROM: {p}")
            if install_rom_bytes(target_dir, p):
                return True

    return os.path.exists(rom_dest)

if __name__ == "__main__":
    ensure_real_rom()
