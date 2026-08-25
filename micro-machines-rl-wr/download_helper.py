import urllib.request
import re
import os
import zipfile

def download_rom():
    dest_dir = r"D:\finny tech deep labs\micro-machines-rl-wr"
    zip_path = os.path.join(dest_dir, "Micro_Machines_2_Turbo_Tournament_Europe.zip")
    rom_dest = os.path.join(dest_dir, "rom.md")

    urls = [
        "https://www.emu-land.net/consoles/genesis/roms?act=showonly&id=1747",
        "https://archive.org/download/sega-mega-drive-genesis-romset-ultra-complete/Micro%20Machines%202%20-%20Turbo%20Tournament%20%28Europe%29%20%28J-Cart%29.zip",
        "https://files.retrostic.com/sega-genesis/Micro%20Machines%202%20-%20Turbo%20Tournament%20(Europe)%20(J-Cart).zip"
    ]

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Referer': 'https://www.google.com/'
    }

    # Try downloading from retrostic / direct
    test_urls = [
        "https://archive.org/download/No-Intro-Collection_2016-01-03_Fixed/Sega%20-%20Mega%20Drive%20-%20Genesis.zip/Micro%20Machines%202%20-%20Turbo%20Tournament%20%28Europe%29%20%28J-Cart%29.zip"
    ]

    for u in test_urls:
        try:
            print(f"Trying: {u}")
            req = urllib.request.Request(u, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = resp.read()
                print(f"Downloaded {len(data)} bytes")
                if len(data) > 50000:
                    with open(zip_path, "wb") as f:
                        f.write(data)
                    print("Saved to zip!")
                    return True
        except Exception as e:
            print(f"Failed: {e}")

    return False

if __name__ == "__main__":
    download_rom()
