import urllib.request
import ssl
import re

ctx = ssl._create_unverified_context()
req = urllib.request.Request('https://tassig.com/downloads/roms/genesis/', headers={'User-Agent': 'Mozilla/5.0'})
try:
    html = urllib.request.urlopen(req, context=ctx, timeout=10).read().decode('utf-8', errors='ignore')
    links = re.findall(r'href=[\'"]([^\'"]+)[\'"]', html)
    print("All Micro matches:")
    for l in links:
        if 'micro' in l.lower() or 'turbo' in l.lower():
            print(l)
except Exception as e:
    print(f"Error: {e}")
