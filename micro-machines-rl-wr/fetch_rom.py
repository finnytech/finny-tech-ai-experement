import urllib.request
import re
import os
import zipfile

def try_fetch():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
    }

    # Test downloading from open directories or direct endpoints
    test_urls = [
        "https://ia800908.us.archive.org/view_archive.php?archive=/11/items/sega_genesis_romset/Micro%20Machines%202%20-%20Turbo%20Tournament%20%28Europe%29%20%28J-Cart%29.zip",
        "https://archive.org/download/sega-genesis-romset/Micro%20Machines%202%20-%20Turbo%20Tournament%20%28Europe%29%20%28J-Cart%29.zip",
        "https://myrient.erista.me/files/No-Intro/Sega%20-%20Mega%20Drive%20-%20Genesis/Micro%20Machines%202%20-%20Turbo%20Tournament%20%28Europe%29%20%28J-Cart%29.zip"
    ]

    dest_zip = r"D:\finny tech deep labs\micro-machines-rl-wr\rom.zip"
    dest_md = r"D:\finny tech deep labs\micro-machines-rl-wr\rom.md"

    for url in test_urls:
        try:
            print(f"Testing URL: {url}")
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=12) as response:
                content = response.read()
                print(f"Received {len(content)} bytes from {url}")
                if len(content) > 50000 and not content.startswith(b"<!DOCTYPE") and not content.startswith(b"<html"):
                    with open(dest_zip, "wb") as f:
                        f.write(content)
                    print(f"Saved {dest_zip}")
                    
                    # Unzip
                    if zipfile.is_zipfile(dest_zip):
                        with zipfile.ZipFile(dest_zip, "r") as z:
                            for name in z.namelist():
                                print(f"Found in zip: {name}")
                                if name.lower().endswith((".md", ".bin", ".gen", ".smd")):
                                    with z.open(name) as src, open(dest_md, "wb") as dst:
                                        dst.write(src.read())
                                    print(f"Successfully extracted {dest_md} ({os.path.getsize(dest_md)} bytes)!")
                                    return True
                    else:
                        with open(dest_md, "wb") as f:
                            f.write(content)
                        print(f"Saved directly as {dest_md}")
                        return True
        except Exception as e:
            print(f"Failed {url}: {e}")

    return False

if __name__ == "__main__":
    try_fetch()
